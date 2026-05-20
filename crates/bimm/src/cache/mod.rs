//! # Cache Implementations

use bunsen::cache as bcache;

mod prefabs;
pub use bcache::{
    DiskCacheConfig,
    PretrainedWeightsDescriptor,
    PretrainedWeightsMap,
    StaticPretrainedWeightsDescriptor,
    StaticPretrainedWeightsMap,
    fetch_model_weights,
    pretrained_weights_resource_key,
    try_cache_download_to_path,
    url_to_cache_key,
};
#[doc(inline)]
pub use prefabs::*;
