//! # Config Prefabs for Well-Known Model Configurations

use crate::cache::weights::{
    PretrainedWeightsDescriptor, PretrainedWeightsMap, StaticPretrainedWeightsMap,
};
use anyhow::bail;
use burn::config::Config;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

/// Static builder for a [`PreFabConfig`]
pub struct StaticPreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Name of the model config pre-fab.
    pub name: &'static str,

    /// Description of the model config pre-fab.
    pub description: &'static str,

    /// Builder function for the config.
    pub builder: fn() -> C,

    /// Pretrained weights map.
    pub weights: Option<&'static StaticPretrainedWeightsMap<'static>>,
}

impl<C> StaticPreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Convert to a [`PreFabConfig<C>`].
    pub fn to_prefab(&self) -> PreFabConfig<C> {
        let builder = self.builder;
        PreFabConfig {
            name: self.name.to_string(),
            description: self.description.to_string(),
            builder: Arc::new(builder),
            weights: self.weights.map(|w| w.to_directory()),
        }
    }

    /// Build a new config.
    pub fn new_config(&self) -> C {
        (self.builder)()
    }
}

impl<C> From<&StaticPreFabConfig<C>> for PreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    fn from(config: &StaticPreFabConfig<C>) -> Self {
        config.to_prefab()
    }
}

impl<C> Debug for StaticPreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        self.to_prefab().fmt(f)
    }
}

/// A [`Config`] Well-Known Pre-Fab.
#[derive(Clone)]
pub struct PreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Name of the model config pre-fab.
    pub name: String,

    /// Description of the model config pre-fab.
    pub description: String,

    /// Builder function for the config.
    pub builder: Arc<dyn Fn() -> C + Send + Sync>,

    /// Pretrained weights map.
    pub weights: Option<PretrainedWeightsMap>,
}

impl<C> Debug for PreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let pretty = f.alternate();

        let type_name = std::any::type_name::<C>();
        let mut handle = f.debug_struct(&format!("PreFabConfig<{}>", type_name));

        handle
            .field("name", &self.name)
            .field("description", &self.description);

        if pretty {
            handle.field("config", &self.new_config());
        }

        handle.finish()
    }
}

impl<C> PreFabConfig<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Build a new config.
    pub fn new_config(&self) -> C {
        (self.builder)()
    }

    /// Lookup a descriptor.
    pub fn lookup_weights(
        &self,
        name: &str,
    ) -> Option<PretrainedWeightsDescriptor> {
        match &self.weights {
            None => None,
            Some(m) => m.lookup_by_name(name),
        }
    }

    /// Lookup a descriptor.
    pub fn try_lookup_weights(
        &self,
        name: &str,
    ) -> anyhow::Result<PretrainedWeightsDescriptor> {
        match self.lookup_weights(name) {
            Some(d) => Ok(d),
            None => bail!("Descriptor not found: {}", name),
        }
    }

    /// Lookup a descriptor.
    pub fn expect_lookup_weights(
        &self,
        name: &str,
    ) -> PretrainedWeightsDescriptor {
        match self.try_lookup_weights(name) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        }
    }
}

/// Static builder for a [`PreFabMap`].
#[derive(Debug)]
pub struct StaticPreFabMap<'a, C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Name of the prefab map.
    pub name: &'a str,

    /// Description of the prefab map.
    pub description: &'a str,

    /// List of prefabs.
    pub items: &'a [&'a StaticPreFabConfig<C>],
}

impl<'a, C> StaticPreFabMap<'a, C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Convert to a [`PreFabMap`].
    pub fn to_prefab_map(&self) -> PreFabMap<C> {
        PreFabMap {
            name: self.name.to_string(),
            description: self.description.to_string(),
            items: self
                .items
                .iter()
                .map(|c| (c.name.to_string(), c.to_prefab()))
                .collect(),
        }
    }

    /// Lookup a prefab.
    pub fn lookup_prefab(
        &self,
        name: &str,
    ) -> Option<PreFabConfig<C>> {
        self.items
            .iter()
            .find(|c| c.name == name)
            .map(|c| c.to_prefab())
    }

    /// Lookup a prefab.
    pub fn try_lookup_prefab(
        &self,
        name: &str,
    ) -> anyhow::Result<PreFabConfig<C>> {
        match self.lookup_prefab(name) {
            Some(d) => Ok(d),
            None => bail!("PreFab not found: {}", name),
        }
    }

    /// Lookup a prefab.
    pub fn expect_lookup_prefab(
        &self,
        name: &str,
    ) -> PreFabConfig<C> {
        match self.try_lookup_prefab(name) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        }
    }
}

/// A map of [`PreFabConfig`]s.
#[derive(Debug, Clone)]
pub struct PreFabMap<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Name of the prefab map.
    pub name: String,

    /// Description of the prefab map.
    pub description: String,

    /// Map of prefabs.
    pub items: BTreeMap<String, PreFabConfig<C>>,
}

impl<C> PreFabMap<C>
where
    C: 'static + Config + Debug + Clone,
{
    /// Lookup a prefab.
    pub fn lookup_by_name(
        &self,
        name: &str,
    ) -> Option<PreFabConfig<C>> {
        self.items.get(name).cloned()
    }

    /// Lookup a prefab.
    pub fn try_lookup_by_name(
        &self,
        name: &str,
    ) -> anyhow::Result<PreFabConfig<C>> {
        match self.lookup_by_name(name) {
            Some(d) => Ok(d),
            None => bail!("PreFab not found: {}", name),
        }
    }

    /// Lookup a prefab.
    pub fn expect_lookup_by_name(
        &self,
        name: &str,
    ) -> PreFabConfig<C> {
        match self.try_lookup_by_name(name) {
            Ok(p) => p,
            Err(e) => panic!("{}", e),
        }
    }
}
