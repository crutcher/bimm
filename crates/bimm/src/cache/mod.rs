//! # Module / Weight Caches

use anyhow::bail;
use burn::data::network::downloader;
use std::fs::{File, create_dir_all};
use std::io::Write;
use std::path::PathBuf;

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
