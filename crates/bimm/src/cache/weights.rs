//! # Module / Weight Caches

use crate::cache::disk::DiskCacheConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

const X25: crc::Crc<u16> = crc::Crc::<u16>::new(&crc::CRC_16_IBM_SDLC);

/// Returns a local path to model weights file.
/// If the file does not exist, it will be downloaded from the given URL.
pub fn fetch_model_weights<S: AsRef<str>>(url: S) -> anyhow::Result<PathBuf> {
    let cache_key = url_to_cache_key(Some("model"), url.as_ref());
    let resource = pretrained_weights_resource_key(&cache_key);

    let disk_cache = DiskCacheConfig::default();
    disk_cache.fetch_resource(url.as_ref(), &resource)
}

/// Build a cache key (bare cache file name) from a name and URL.
pub fn url_to_cache_key(
    name: Option<&str>,
    url: &str,
) -> String {
    let hash = X25.checksum(url.as_bytes()).to_string();
    let base_name = url.rsplit_once('/').unwrap().1;
    match name {
        Some(n) => format!("{}-{}-{}", n, hash, base_name),
        None => format!("{}-{}", hash, base_name),
    }
}

/// Get the cache resource key for a pretrained weights file.
///
/// # Arguments
///
/// - `cache_key`: the cache key (the bare cache file name).
///
/// # Returns
///
/// The cache resource key.
pub fn pretrained_weights_resource_key(cache_key: &str) -> Vec<String> {
    vec!["weights".to_string(), cache_key.to_string()]
}

/// Static [`PretrainedWeightsDescriptor`] provider.
#[derive(Debug)]
pub struct StaticPretrainedWeightsDescriptor<'a> {
    /// Name of the model.
    pub name: &'a str,

    /// Description of the model.
    pub description: &'a str,

    /// URL to download the weights from.
    pub urls: &'a [&'a str],
}

impl<'a> StaticPretrainedWeightsDescriptor<'a> {
    /// Convert to a [`PretrainedWeightsDescriptor`].
    pub fn to_descriptor(&self) -> PretrainedWeightsDescriptor {
        PretrainedWeightsDescriptor {
            name: self.name.to_string(),
            description: self.description.to_string(),
            urls: self.urls.iter().map(|s| s.to_string()).collect(),
        }
    }
}

impl From<&StaticPretrainedWeightsDescriptor<'_>> for PretrainedWeightsDescriptor {
    fn from(descriptor: &StaticPretrainedWeightsDescriptor) -> Self {
        descriptor.to_descriptor()
    }
}

/// A descriptor for a pretrained weights file.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct PretrainedWeightsDescriptor {
    /// Name of the model.
    pub name: String,

    /// Description of the model.
    pub description: String,

    /// URL to download the weights from.
    pub urls: Vec<String>,
}

impl PretrainedWeightsDescriptor {
    /// Cache Key
    ///
    /// The key is ``{name}-{url crc hash}-{url basename}``.
    pub fn cache_key(&self) -> String {
        url_to_cache_key(Some(&self.name), self.urls.first().unwrap())
    }

    /// Read-Through Cache the Model Weights
    ///
    /// # Returns
    ///
    /// The disk location of the cached weights.
    pub fn fetch_weights_to_disk_cache(
        &self,
        disk_cache: &DiskCacheConfig,
    ) -> anyhow::Result<PathBuf> {
        let url = self.urls.first().unwrap();
        let cache_key = &self.cache_key();
        let resource = pretrained_weights_resource_key(cache_key);

        disk_cache.fetch_resource(url, &resource)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_descriptor_to_descriptor() {
        let s_desc = StaticPretrainedWeightsDescriptor {
            name: "my_model",
            description: "some description of my model.",
            urls: &["foo", "bar"],
        };
        let d_desc = s_desc.to_descriptor();

        assert_eq!(d_desc.name, s_desc.name.to_string());
        assert_eq!(d_desc.description, s_desc.description.to_string());
        assert_eq!(
            d_desc.urls,
            s_desc
                .urls
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
        );
    }
}
