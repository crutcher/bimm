#![recursion_limit = "256"]

extern crate core;

use anyhow::bail;
use bimm::models::resnet::resnet_model::{ResNet, ResNetAbstractConfig};
use burn::backend::Cuda;
use burn::data::network::downloader;
use clap::{Parser, arg};
use core::clone::Clone;
use core::option::Option;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Directory to save the artifacts.
    #[arg(
        long,
        default_value = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    )]
    weights: String,

    #[arg(long, default_value = "1000")]
    num_classes: usize,
}

#[allow(dead_code, unused)]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    type B = Cuda;
    let device = Default::default();

    let weights = fetch_model_weights(&args.weights)?;

    let model: ResNet<B> = ResNetAbstractConfig::resnet18(args.num_classes)
        .to_structure()
        .init(&device)
        .load_pytorch_weights(weights)?;

    Ok(())
}

/// Returns a local path to model weights file.
/// If the file does not exist, it will be downloaded from the given URL.
pub fn fetch_model_weights<S: AsRef<str>>(model_url: S) -> anyhow::Result<PathBuf> {
    let model_url = model_url.as_ref();

    // Model cache directory
    let model_dir = dirs::home_dir()
        .expect("Should be able to get home directory")
        .join(".cache")
        .join("burn-model-cache");

    if !model_dir.exists() {
        create_dir_all(&model_dir)?;
    }

    let file_base_name = model_url.rsplit_once('/').unwrap().1;
    let file_name = model_dir.join(file_base_name);
    if !file_name.exists() {
        // Download file content
        let bytes = downloader::download_file_as_bytes(model_url, file_base_name);

        // Write content to file
        let mut output_file = File::create(&file_name)?;
        let bytes_written = output_file.write(&bytes)?;

        if bytes_written != bytes.len() {
            bail!("Failed to write the whole model weights file.",);
        }
    }

    Ok(file_name)
}

/*
pub fn dump_key_shapes(weights_path: &str) -> anyhow::Result<BTreeMap<String, Vec<usize>>> {
    use candle_core::pickle;
    use std::collections::BTreeMap;

    // Read the pickle file and return a map of names to Candle tensors
    let key_shapes: BTreeMap<String, Vec<usize>> = pickle::read_all_with_key(weights_path, None)?
        .into_iter()
        .map(|(key, tensor)| (key, tensor.shape().clone().into_dims()))
        .collect();

    Ok(key_shapes)
}
 */
