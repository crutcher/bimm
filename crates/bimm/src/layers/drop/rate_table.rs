//! Common rate table for `DropPath` regularization.

use crate::compat::ops::float_vec_linspace;
use alloc::vec::Vec;

/// Computes a progressive incremental path drop rate for stochastic depth.
///
/// Used by SWIN Transformer V2 to apply a gradual increase in drop path rate.
///
/// # Arguments
///
/// * `drop_path_rate`: The final drop path rate to be achieved.
/// * `depth`: The total number of layers in the model.
///
/// # Returns
///
/// A vector of drop path rates, one for each layer, starting from 0.0 and ending at `drop_path_rate`.
#[inline(always)]
#[must_use]
pub fn progressive_dpr(
    drop_path_rate: f64,
    depth: usize,
) -> Vec<f64> {
    float_vec_linspace(
        0.0,
        drop_path_rate,
        // Total number of layers
        depth,
    )
}

/// Represents a table of progressive drop path rates for each layer in a model.
pub struct DropPathRateDepthTable {
    progressive_dpr: Vec<f64>,
    layer_depths: Vec<usize>,
}

impl DropPathRateDepthTable {
    /// Creates a new `DropPathRateDepthTable` with the specified drop path rate and layer depths.
    ///
    /// # Arguments
    ///
    /// * `drop_path_rate`: The final drop path rate to be achieved.
    /// * `layer_depths`: A slice of layer depths, where each element represents the depth of a specific layer.
    ///
    /// # Returns
    ///
    /// A new `DropPathRateDepthTable` instance containing the progressive drop path rates for each layer.
    #[must_use]
    pub fn new(
        drop_path_rate: f64,
        layer_depths: &[usize],
    ) -> Self {
        let layer_depths = layer_depths.to_vec();
        let progressive_dpr = progressive_dpr(drop_path_rate, layer_depths.iter().sum());
        Self {
            progressive_dpr,
            layer_depths,
        }
    }

    /// Returns the depth of each layer.
    pub fn layer_depths(&self) -> &[usize] {
        &self.layer_depths
    }

    /// Returns the total depth of all layers.
    pub fn total_depth(&self) -> usize {
        self.layer_depths.iter().sum()
    }

    /// Returns the total number of layers.
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.layer_depths.len()
    }

    /// Returns the drop path rates for a specific layer.
    ///
    /// This will be a vector of size equal to the depth of the specified layer;
    /// corresponding to the progressive drop path rates for that layer.
    ///
    /// # Arguments
    ///
    /// * `layer_i`: The index of the layer for which to retrieve the drop path rates.
    ///
    /// # Panics
    ///
    /// If the layer index is out of bounds, it will panic with a message indicating the issue.
    #[must_use]
    pub fn layer_dprs(
        &self,
        layer_i: usize,
    ) -> Vec<f64> {
        if layer_i >= self.num_layers() {
            panic!(
                "Layer index {} out of bounds for {} layers",
                layer_i,
                self.num_layers()
            );
        }
        let depths = &self.layer_depths;
        let progressive_dpr1 = &self.progressive_dpr;
        let start = depths[..layer_i].iter().sum::<usize>();
        let end = start + depths[layer_i];

        progressive_dpr1[start..end].to_vec()
    }

    /// Returns the `layer_dprs` for all layers as a vector of vectors.
    #[inline(always)]
    #[must_use]
    pub fn layer_rates(&self) -> Vec<Vec<f64>> {
        (0..self.num_layers()).map(|i| self.layer_dprs(i)).collect()
    }

    /// Returns the progressive drop path rates for layer with the given depths.
    ///
    /// This is a convenience function that creates a new `DropPathRateDepthTable` and returns the layer rates.
    ///
    /// # Arguments
    ///
    /// * `drop_path_rate`: The final drop path rate to be achieved.
    /// * `layer_depths`: A slice of layer depths, where each element represents the depth of a specific layer.
    ///
    /// # Returns
    ///
    /// A vector of vectors, where each inner vector contains the progressive drop path rates for a specific layer.
    #[must_use]
    pub fn dpr_layer_rates(
        drop_path_rate: f64,
        layer_depths: &[usize],
    ) -> Vec<Vec<f64>> {
        let dpr_table = DropPathRateDepthTable::new(drop_path_rate, layer_depths);
        dpr_table.layer_rates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::drop::rate_table::progressive_dpr;
    use crate::testing::assert_close_to_vec;
    use alloc::vec;
    use hamcrest::prelude::*;

    #[test]
    fn test_incremental_drop_rate() {
        let drop_path_rate = 0.1;
        let depth = 9;
        let rates = progressive_dpr(drop_path_rate, depth);
        assert_close_to_vec(
            &rates,
            &[0.0, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1],
            0.001,
        );
    }

    #[test]
    fn test_table() {
        let depths = vec![2, 3, 4];
        let dpr_table = DropPathRateDepthTable::new(0.1, &depths);

        assert_eq!(dpr_table.total_depth(), 9);

        assert_that!(
            &dpr_table.layer_depths().to_vec(),
            contains(depths.clone()).exactly()
        );

        assert_close_to_vec(&dpr_table.layer_dprs(0), &[0.0, 0.0125], 0.001);

        assert_close_to_vec(&dpr_table.layer_dprs(1), &[0.025, 0.0375, 0.05], 0.001);

        let rates = dpr_table.layer_rates();

        assert_eq!(rates.len(), 3);
        assert_close_to_vec(&rates[0], &[0.0, 0.0125], 0.001);
        assert_close_to_vec(&rates[1], &[0.025, 0.0375, 0.05], 0.001);
        assert_close_to_vec(&rates[2], &[0.0625, 0.075, 0.0875, 0.1], 0.001);
    }

    #[should_panic(expected = "Layer index 3 out of bounds for 3 layers")]
    #[test]
    fn test_layer_dprs_out_of_bounds() {
        let depths = vec![2, 3, 4];
        let dpr_table = DropPathRateDepthTable::new(0.1, &depths);
        // This should panic because there are only 3 layers (0, 1, 2)
        let _d = dpr_table.layer_dprs(3);
    }

    #[test]
    fn test_dpr_layer_rates() {
        let drop_path_rate = 0.1;
        let layer_depths = vec![2, 3, 4];
        let rates = DropPathRateDepthTable::dpr_layer_rates(drop_path_rate, &layer_depths);

        assert_eq!(rates.len(), 3);
        assert_close_to_vec(&rates[0], &[0.0, 0.0125], 0.001);
        assert_close_to_vec(&rates[1], &[0.025, 0.0375, 0.05], 0.001);
        assert_close_to_vec(&rates[2], &[0.0625, 0.075, 0.0875, 0.1], 0.001);
    }
}
