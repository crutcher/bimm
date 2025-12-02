use crate::models::swin::v2::window_attention::{
    OffsetGridRelativePositionBias, RelativePositionBiasConfig, RelativePositionBiasMeta,
    apply_attention_mask,
};
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::config::Config;
use burn::module::{Module, Param, ParamId};
use burn::nn::activation::ActivationConfig;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Tensor};
use burn::tensor::activation::softmax;
use burn::tensor::linalg::{Norm, vector_normalize};

/// EPS is a small constant used to avoid numerical instability in calculations.
pub const EPS: f64 = 1e-12;

/// Common introspection interface for `WindowAttention`.
pub trait WindowAttentionMeta {
    /// Get the input/channel dimension size.
    fn d_input(&self) -> usize;

    /// Get the window shape ``[height, width]``.
    fn window_shape(&self) -> [usize; 2];

    /// Get the height of the window.
    fn window_height(&self) -> usize {
        self.window_shape()[0]
    }

    /// Get the width of the window.
    fn window_width(&self) -> usize {
        self.window_shape()[1]
    }

    /// Get the number of attention heads.
    fn num_heads(&self) -> usize;

    /// Get the drop rate for attention.
    fn attn_drop(&self) -> f64;

    /// Get the drop rate for projection.
    fn proj_drop(&self) -> f64;

    /// Is the QKV bias enabled?
    fn enable_qkv_bias(&self) -> bool;
}

impl WindowAttentionMeta for WindowAttentionConfig {
    /// Get the input/channel dimension size.
    fn d_input(&self) -> usize {
        self.d_input
    }

    /// Get the window shape.
    fn window_shape(&self) -> [usize; 2] {
        self.window_shape
    }

    /// Get the number of heads.
    fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Get the drop rate for attention.
    fn attn_drop(&self) -> f64 {
        self.attn_drop
    }

    /// Get the drop rate for projection.
    fn proj_drop(&self) -> f64 {
        self.proj_drop
    }

    /// Check if QKV bias is enabled.
    fn enable_qkv_bias(&self) -> bool {
        self.enable_qkv_bias
    }
}

impl<B: Backend> WindowAttentionMeta for WindowAttention<B> {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn window_shape(&self) -> [usize; 2] {
        self.rpb_module.window_shape()
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn attn_drop(&self) -> f64 {
        self.attn_drop.prob
    }

    fn proj_drop(&self) -> f64 {
        self.proj_drop.prob
    }

    fn enable_qkv_bias(&self) -> bool {
        self.q_linear.bias.is_some()
    }
}

#[cfg(test)]
mod tests {
    use crate::models::swin::v2::window_attention::*;
    use bimm_contracts::assert_shape_contract;
    use burn::backend::NdArray;
    use burn::prelude::Tensor;
    use burn::tensor::Distribution;

    #[test]
    fn test_window_attention_meta() {
        let window_shape = [4, 4];
        let num_heads = 8;
        let channels = num_heads * 3; // Assuming cph = 3

        let config = WindowAttentionConfig::new(channels, window_shape, num_heads);

        assert_eq!(config.d_input(), channels);
        assert_eq!(config.window_shape(), window_shape);
        assert_eq!(config.num_heads(), num_heads);
        assert!(config.enable_qkv_bias());
        assert_eq!(config.attn_drop(), 0.0);
        assert_eq!(config.proj_drop(), 0.0);
        assert_eq!(config.window_height(), 4);
        assert_eq!(config.window_width(), 4);

        let device = Default::default();
        let attn_mod = config.init::<NdArray>(&device);

        assert_eq!(attn_mod.d_input(), channels);
        assert_eq!(attn_mod.window_shape(), window_shape);
        assert_eq!(attn_mod.num_heads(), num_heads);
        assert!(attn_mod.enable_qkv_bias());
        assert_eq!(attn_mod.attn_drop(), 0.0);
        assert_eq!(attn_mod.proj_drop(), 0.0);
    }

    #[test]
    fn test_wa() {
        let b = 3;
        let num_windows = 2;

        let window_size = 4;

        let num_heads = 5;
        let cph = 3;
        let channels = num_heads * cph;

        let config = WindowAttentionConfig::new(channels, [window_size, window_size], num_heads);

        let device = Default::default();
        let attn_mod = config.init::<NdArray>(&device);

        assert_eq!(attn_mod.d_input(), channels);
        assert_eq!(attn_mod.window_shape(), [window_size, window_size]);
        assert_eq!(attn_mod.num_heads(), num_heads);

        let distribution = Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<NdArray, 3>::random(
            [b * num_windows, window_size * window_size, channels],
            distribution,
            &device,
        );

        let res = attn_mod.forward(input, None);
        assert_shape_contract!(
            [
                "bn" = "batch" * "num_windows",
                "window_size" ^ 2,
                "channels"
            ],
            &res.dims(),
            &[
                ("batch", b),
                ("num_windows", num_windows),
                ("window_size", window_size),
                ("channels", channels),
            ],
        );
    }
}

/// Configuration for the `WindowAttention` module.
#[derive(Config, Debug)]
pub struct WindowAttentionConfig {
    /// Input dimension size.
    pub d_input: usize,

    /// Window shape as [height, width].
    pub window_shape: [usize; 2],

    /// Number of attention heads.
    pub num_heads: usize,

    /// Whether to enable bias for Q, K, V linear layers.
    #[config(default = true)]
    pub enable_qkv_bias: bool,

    /// Dropout rate for attention.
    #[config(default = 0.)]
    pub attn_drop: f64,

    /// Dropout rate for projection.
    #[config(default = 0.)]
    pub proj_drop: f64,

    /// The hidden dimension of the MLP.
    #[config(default = 512)]
    pub rpb_mlp_hidden_dim: usize,

    /// The activation layer configuration.
    #[config(default = "ActivationConfig::Relu")]
    pub rpb_mlp_activation: ActivationConfig,
}

/// The `WindowAttention` module.
#[derive(Module, Debug)]
pub struct WindowAttention<B: Backend> {
    /// Input dimension size.
    pub d_input: usize,

    /// Number of attention heads.
    pub num_heads: usize,

    /// Linear layer for Q.
    pub q_linear: Linear<B>,

    /// Linear layer for K.
    pub k_linear: Linear<B>,

    /// Linear layer for V.
    pub v_linear: Linear<B>,

    /// Learnable logit scale.
    pub logit_scale: Param<Tensor<B, 3>>,

    /// Relative position bias module.
    pub rpb_module: OffsetGridRelativePositionBias<B>,

    /// Linear layer for projection.
    pub proj: Linear<B>,

    /// Dropout for attention.
    pub attn_drop: Dropout,

    /// Dropout for projection.
    pub proj_drop: Dropout,
}

impl<B: Backend> WindowAttention<B> {
    /// Forward pass of the `WindowAttention` module.
    ///
    /// # Arguments
    ///
    /// - `x`: Input tensor of shape ``[batch * num_windows, Wh*Ww, channels]``.
    /// - `mask`: Optional mask tensor of shape ``[num_windows, Wh*Ww, Wh*Ww]``.
    ///
    /// # Returns
    ///
    /// Output tensor of shape ``[batch * num_windows, Wh*Ww, channels]``.
    ///
    /// # Panics
    ///
    /// On shape contract failure.
    #[must_use]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [wh, ww] = self.window_shape();

        let [b_nw, n, c] = unpack_shape_contract!(
            ["b_nw", "n" = "wh" * "ww", "c"],
            &x.dims(),
            &["b_nw", "n", "c"],
            &[("wh", wh), ("ww", ww)],
        );

        self.window_shape();

        // n = ws * ws

        let q = self.q_linear.forward(x.clone());
        let k = self.k_linear.forward(x.clone());
        let v = self.v_linear.forward(x);
        // (b_nw, ws*ws, c)

        let c_per_head = c / self.num_heads;
        let qkv_shape = [b_nw, n, self.num_heads, c_per_head];

        let q = q.reshape(qkv_shape).swap_dims(1, 2);
        let k = k.reshape(qkv_shape).swap_dims(1, 2);
        let v = v.reshape(qkv_shape).swap_dims(1, 2);
        // (b_nw, num_heads, ws*ws, c_per_head)

        let attn = self.attention(b_nw, n, q, k, mask);

        let x = attn.matmul(v);
        let x = x.swap_dims(1, 2).reshape([b_nw, n, c]);
        // (b_nw, ws*ws, c)

        let x = self.proj.forward(x);
        self.proj_drop.forward(x)
        // (b_nw, ws*ws, c)
    }

    /// Compute the attention.
    ///
    /// # Arguments
    ///
    /// - `b_nw`: Batch size times number of windows.
    /// - `n`: Number of elements in the input tensor.
    /// - `q`: Query tensor of shape (`b_nw`, `num_heads`, ws*ws, `c_per_head`).
    /// - `k`: Key tensor of shape (`b_nw`, `num_heads`, ws*ws, `c_per_head`).
    /// - `mask`: Optional mask tensor of shape (`num_windows`, ws*ws, ws*ws).
    ///
    /// # Returns
    ///
    /// - Output attention tensor of shape (`b_nw`, `num_heads`, ws*ws, ws*ws).
    #[must_use]
    fn attention(
        &self,
        b_nw: usize,
        n: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 4> {
        // cosine attention
        let q = vector_normalize(q, Norm::L2, 3, EPS);
        // (b_nw, num_heads, ws*ws, c_per_head)

        let k = vector_normalize(k, Norm::L2, 3, EPS).swap_dims(2, 3);
        // (b_nw, num_heads, c_per_head, ws*ws)

        let attn = q.matmul(k);
        // (b_nw, num_heads, ws*ws, ws*ws)

        let attn = self.encode_attention(attn);
        // (b_nw, num_heads, Wh*Ww, Wh*Ww)

        let attn = self.attn_drop.forward(attn);

        let attn = match mask {
            None => attn,
            Some(mask) => apply_attention_mask(b_nw, n, self.num_heads, attn, mask),
        };

        // (b_nw, num_heads, Wh*Ww, Wh*Ww)
        let attn = softmax(attn, 3);
        assert_shape_contract_periodically!(
            ["b_nw", "num_heads", "Wh" * "Ww", "Wh" * "Ww"],
            &attn.dims(),
            &[
                ("b_nw", b_nw),
                ("num_heads", self.num_heads()),
                ("Wh", self.window_shape()[0]),
                ("Ww", self.window_shape()[1]),
            ],
        );

        attn
    }

    /// Get the learnable logit scale.
    ///
    /// # Returns
    ///
    /// - Output tensor of shape (`num_heads`, 1, 1).
    #[must_use]
    fn logit_scale(&self) -> Tensor<B, 3> {
        // TODO(crutcher): I suspect this is a bug in the original code.
        // I *think* the authors thought this was log_10; and not log_e;
        // it doesn't make sense to use log_e with a scale of 10.0 here.
        self.logit_scale.val().clamp_max((1.0f64 / 0.01).ln()).exp()
    }

    /// Get the learnable relative position bias.
    ///
    /// # Returns
    ///
    /// - Output tensor of shape (`num_heads`, Wh*Ww, Wh*Ww).
    #[inline(always)]
    #[must_use]
    fn relative_pos_bias(&self) -> Tensor<B, 3> {
        self.rpb_module.forward()
    }

    /// Encode the attention logits with the logit scale and relative position bias.
    ///
    /// # Arguments
    ///
    /// - `attn`: Attention logits tensor of shape (`b_nw`, `num_heads`, Wh*Ww, Wh*Ww).
    ///
    /// # Returns
    ///
    /// - Output tensor of shape (`b_nw`, `num_heads`, Wh*Ww, Wh*Ww).
    #[inline(always)]
    #[must_use]
    fn encode_attention(
        &self,
        attn: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        attn * self.logit_scale().unsqueeze() + self.relative_pos_bias().unsqueeze()
    }
}

impl WindowAttentionConfig {
    /// Create a new `WindowAttentionConfig`.
    ///
    /// # Arguments
    ///
    /// - `device`: The backend device to use.
    ///
    /// # Returns
    ///
    /// A new instance of `WindowAttentionConfig`.
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> WindowAttention<B> {
        let d_input = self.d_input();
        let num_heads = self.num_heads();
        let window_size = self.window_shape();

        WindowAttention {
            d_input,
            num_heads,
            q_linear: LinearConfig::new(d_input, d_input)
                .with_bias(self.enable_qkv_bias)
                .init(device),
            k_linear: LinearConfig::new(d_input, d_input).init(device),
            v_linear: LinearConfig::new(d_input, d_input)
                .with_bias(self.enable_qkv_bias)
                .init(device),
            logit_scale: Param::initialized(
                ParamId::new(),
                // TODO(crutcher): I suspect this is a bug in the original code.
                // I *think* the authors thought this was log_10; and not log_e;
                // it doesn't make sense to use log_e with a scale of 10.0 here.
                Tensor::<B, 3>::ones([num_heads, 1, 1], device)
                    .mul_scalar(10.0)
                    .log(),
            ),
            attn_drop: DropoutConfig {
                prob: self.attn_drop,
            }
            .init(),
            rpb_module: RelativePositionBiasConfig::new(num_heads, window_size)
                .with_mlp_hidden_dim(self.rpb_mlp_hidden_dim)
                .with_mlp_activation(self.rpb_mlp_activation.clone())
                .init_offset_grid_rpb(device),
            proj: LinearConfig::new(d_input, d_input)
                .with_bias(false)
                .init(device),
            proj_drop: DropoutConfig {
                prob: self.proj_drop,
            }
            .init(),
        }
    }
}
