//! Data source connectors. Each connector turns one kind of source into
//! [`synk_core::Document`]s.

mod fs;
pub mod readers;

pub use fs::FsConnector;
pub use readers::{SUPPORTED_EXTENSIONS, is_supported};
