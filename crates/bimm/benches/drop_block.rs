use std::hint::black_box;

use bimm::{
    layers::drop::drop_block::{
        DropBlockOptions,
        drop_block_2d,
    },
    utility::burn::noise::NoiseConfig,
};
use burn::{
    backend::NdArray,
    prelude::Tensor,
};
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn bench_drop_block_10x32x32x3_7_normalise(c: &mut Criterion) {
    type B = NdArray<f32>;
    let device = Default::default();

    let batch_size = 10;
    let height = 32;
    let width = height;
    let channels = 3;

    let shape = [batch_size, height, width, channels];
    let tensor: Tensor<B, 4> = Tensor::ones(shape, &device);

    for noise in [None, Some(NoiseConfig::default())] {
        for batchwise in [false, true] {
            for couple_channels in [false, true] {
                for partial_edge_blocks in [false, true] {
                    let config = DropBlockOptions::default()
                        .with_batchwise(batchwise)
                        .with_couple_channels(couple_channels)
                        .with_partial_edge_blocks(partial_edge_blocks)
                        .with_noise(noise.clone());

                    c.bench_function(
                        format!(
                            "drop_block_2d: {:?} norm noise={} b={batchwise} c={couple_channels} p={partial_edge_blocks}",
                            shape,
                            noise.is_some(),
                        )
                            .as_str(),
                        |b| {
                            b.iter(|| {
                                black_box(drop_block_2d(tensor.clone(), &config));
                            })
                        },
                    );
                }
            }
        }
    }
}

criterion_group!(benches, bench_drop_block_10x32x32x3_7_normalise,);
criterion_main!(benches);
