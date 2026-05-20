#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//!# bimm-firehose - Burn-based Data Pipeline

/// New Data Table module.
pub mod core;

/// Namespace of common operators.
pub mod ops;

/// Burn Integration Module.
pub mod burn;
pub mod utility;

/// Experimental Data Pipeline module; non-public.
#[allow(dead_code)]
mod pipeline;

/// Define a self-referential ID.
///
/// The id will be defined as a static string constant that refers to its own
/// namespace path.
///
/// # Arguments
///
/// * `$name`: The final path name of the ID to define; the rest of the name
///   will be taken from the module context.
///
/// # Example
/// ```
/// // In module "foo::bar"
/// bimm_firehose::define_self_referential_id!(ID);
/// // pub static ID: &str = "foo::bar::ID";
/// ```
#[macro_export]
macro_rules! define_self_referential_id {
    ($name:ident) => {
        /// Self-referential ID.
        pub static $name: &str = concat!(module_path!(), "::", stringify!($name),);
    };
    ($schema:literal, $name:ident) => {
        #[doc = concat!("Self-referential '", $schema, "' ID.")]
        pub static $name: &str = concat!($schema, "://", module_path!(), "::", stringify!($name),);
    };
}
