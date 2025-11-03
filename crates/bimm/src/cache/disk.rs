//! # Cache Policy

use crate::cache::disk;
use alloc::string::String;
use anyhow::bail;
use burn::config::Config;
use burn::data::network::downloader;
use std::fs::{File, remove_file};
use std::io::Write;
use std::path::PathBuf;

/// Cache Policy
#[derive(Config, Debug)]
pub struct DiskCacheConfig {
    /// Key for the root cache directory.
    #[config(default = "\"bimm\".to_string()")]
    pub root_cache_key: String,
}

impl Default for DiskCacheConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl DiskCacheConfig {
    /// Fetch the base cache directory.
    ///
    /// If the cache directory does not exist, does not create it.
    pub fn base_cache_dir(&self) -> anyhow::Result<PathBuf> {
        Ok(dirs::home_dir()
            .expect("Should be able to get home directory")
            .join(".cache")
            .join(&self.root_cache_key))
    }

    /// Fetch the base cache directory.
    ///
    /// If the cache directory does not exist, creates it.
    pub fn ensure_base_cache_dir(&self) -> anyhow::Result<PathBuf> {
        let dir = self.base_cache_dir()?;
        if !dir.exists() {
            std::fs::create_dir_all(&dir)?;
        }
        Ok(dir)
    }

    /// Map a resource key to a cache path.
    ///
    /// Does not ensure that the path (or any of the parents) exist.
    pub fn resource_to_path(
        &self,
        resource_key: &[String],
    ) -> anyhow::Result<PathBuf> {
        let path = self.base_cache_dir()?;
        Ok(resource_key.iter().fold(path, |acc, s| acc.join(s)))
    }

    /// Map a resource key to a cache path and ensure the parent directory exists.
    pub fn ensure_resource_parent_dir(
        &self,
        resource_key: &[String],
    ) -> anyhow::Result<PathBuf> {
        let path = self.resource_to_path(resource_key)?;
        if !path.exists() {
            std::fs::create_dir_all(path.parent().unwrap())?;
        }
        Ok(path)
    }

    /// Fetch a Resource to the Cache.
    pub fn fetch_resource(
        &self,
        url: &str,
        resource: &[String],
    ) -> anyhow::Result<PathBuf> {
        let cache_file_path = self.ensure_resource_parent_dir(resource)?;
        disk::try_cache_download_to_path(url, cache_file_path)
    }
}

/// Download a URL resource to a given path.
///
/// If the path already exists, does nothing.
///
/// # Returns
///
/// The cache path.
pub fn try_cache_download_to_path(
    url: &str,
    cache_file_path: PathBuf,
) -> anyhow::Result<PathBuf> {
    if !cache_file_path.exists() {
        let file_name = cache_file_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string();

        // TODO: download-to-file instead of download-to-memory.
        // Download file content
        let bytes = downloader::download_file_as_bytes(url, &file_name);

        // Write content to file
        let mut output_file = File::create(&cache_file_path)?;
        let bytes_written = output_file.write(&bytes)?;

        if bytes_written != bytes.len() {
            remove_file(cache_file_path)?;
            bail!("Failed to write the whole model weights file.");
        }
    }

    Ok(cache_file_path)
}
