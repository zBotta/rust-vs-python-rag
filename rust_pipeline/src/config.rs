//! Configuration loader for the RAG benchmark.
//!
//! Reads `benchmark_config.toml`, validates required keys, and returns a
//! typed `BenchmarkConfig` struct. Missing required keys produce a
//! descriptive `ConfigError` that names the absent key.

use serde::Deserialize;
use std::path::Path;
use thiserror::Error;

/// All errors that can arise during configuration loading.
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("Missing required configuration key: '{0}'")]
    MissingKey(String),

    #[error("Failed to read config file '{path}': {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },

    #[error("Failed to parse config file: {0}")]
    Parse(#[from] toml::de::Error),
}

/// Raw TOML representation — all fields are `Option<T>` so we can detect
/// missing keys and emit a descriptive error rather than a generic parse error.
#[derive(Debug, Deserialize)]
struct RawConfig {
    dataset_name: Option<String>,
    dataset_subset: Option<String>,
    num_documents: Option<u64>,
    chunk_size: Option<u64>,
    chunk_overlap: Option<u64>,
    embedding_model: Option<String>,
    top_k: Option<u64>,
    llm_model: Option<String>,
    llm_host: Option<String>,
    query_set_path: Option<String>,
    output_dir: Option<String>,
}

/// Validated benchmark configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkConfig {
    pub dataset_name: String,
    pub dataset_subset: String,
    pub num_documents: usize,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub embedding_model: String,
    pub top_k: usize,
    pub llm_model: String,
    pub llm_host: String,
    pub query_set_path: String,
    pub output_dir: String,
}

/// Helper: extract a required `Option<String>` field or return `ConfigError::MissingKey`.
macro_rules! require_str {
    ($raw:expr, $field:ident) => {
        $raw.$field
            .ok_or_else(|| ConfigError::MissingKey(stringify!($field).to_string()))?
    };
}

/// Helper: extract a required `Option<u64>` field or return `ConfigError::MissingKey`.
macro_rules! require_u64 {
    ($raw:expr, $field:ident) => {
        $raw.$field
            .ok_or_else(|| ConfigError::MissingKey(stringify!($field).to_string()))? as usize
    };
}

/// Load and validate benchmark configuration from a TOML file.
///
/// # Errors
///
/// Returns [`ConfigError::MissingKey`] (naming the absent key) when any
/// required key is absent from the file.
pub fn load_config(config_path: impl AsRef<Path>) -> Result<BenchmarkConfig, ConfigError> {
    let path = config_path.as_ref();
    let contents = std::fs::read_to_string(path).map_err(|e| ConfigError::Io {
        path: path.display().to_string(),
        source: e,
    })?;

    let raw: RawConfig = toml::from_str(&contents)?;

    Ok(BenchmarkConfig {
        dataset_name: require_str!(raw, dataset_name),
        dataset_subset: require_str!(raw, dataset_subset),
        num_documents: require_u64!(raw, num_documents),
        chunk_size: require_u64!(raw, chunk_size),
        chunk_overlap: require_u64!(raw, chunk_overlap),
        embedding_model: require_str!(raw, embedding_model),
        top_k: require_u64!(raw, top_k),
        llm_model: require_str!(raw, llm_model),
        llm_host: require_str!(raw, llm_host),
        query_set_path: require_str!(raw, query_set_path),
        output_dir: require_str!(raw, output_dir),
    })
}

// ---------------------------------------------------------------------------
// Property-based tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// All required configuration keys.
    const REQUIRED_KEYS: &[&str] = &[
        "dataset_name",
        "dataset_subset",
        "num_documents",
        "chunk_size",
        "chunk_overlap",
        "embedding_model",
        "top_k",
        "llm_model",
        "llm_host",
        "query_set_path",
        "output_dir",
    ];

    /// Build a complete valid TOML string.
    fn full_toml() -> String {
        r#"
dataset_name    = "wikipedia"
dataset_subset  = "20220301.simple"
num_documents   = 1000
chunk_size      = 512
chunk_overlap   = 64
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
top_k           = 5
llm_model       = "llama3.2:3b"
llm_host        = "http://localhost:11434"
query_set_path  = "query_set.json"
output_dir      = "output/"
"#
        .to_string()
    }

    /// Write `content` to a temp file and return the file handle.
    fn write_temp(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    // -----------------------------------------------------------------------
    // Property 15: Missing required config key produces descriptive error
    // Feature: rust-vs-python-rag-benchmark, Property 15: Missing required config key produces descriptive error
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// For any required key that is removed from the config, load_config must
        /// return a ConfigError::MissingKey whose message contains the key name.
        ///
        /// Feature: rust-vs-python-rag-benchmark, Property 15: Missing required config key produces descriptive error
        #[test]
        fn prop_missing_required_key_produces_descriptive_error(
            key_idx in 0usize..REQUIRED_KEYS.len()
        ) {
            let missing_key = REQUIRED_KEYS[key_idx];

            // Build a TOML that omits exactly one required key.
            let toml_content: String = full_toml()
                .lines()
                .filter(|line| {
                    // Drop the line that starts with the key name.
                    let trimmed = line.trim_start();
                    !trimmed.starts_with(missing_key)
                })
                .collect::<Vec<_>>()
                .join("\n");

            let f = write_temp(&toml_content);
            let result = load_config(f.path());

            prop_assert!(
                result.is_err(),
                "Expected error for missing key '{}', but got Ok",
                missing_key
            );

            let err = result.unwrap_err();
            let err_msg = err.to_string();
            prop_assert!(
                err_msg.contains(missing_key),
                "Error message '{}' does not mention missing key '{}'",
                err_msg,
                missing_key
            );
        }
    }

    // -----------------------------------------------------------------------
    // Property 16: Absent optional config key uses documented default
    // Feature: rust-vs-python-rag-benchmark, Property 16: Absent optional config key uses documented default
    // -----------------------------------------------------------------------
    //
    // Per the current design all keys are required; there are no optional keys
    // with defaults yet. This property test verifies the invariant that a
    // complete config (no keys absent) loads successfully — acting as a
    // baseline for when optional keys are added in future tasks.
    //
    // When optional keys with defaults are introduced, this test will be
    // extended to enumerate them and assert the default values.

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// A complete config with all required keys present must load without error.
        ///
        /// Feature: rust-vs-python-rag-benchmark, Property 16: Absent optional config key uses documented default
        #[test]
        fn prop_complete_config_loads_successfully(_seed in 0u32..100) {
            // There are currently no optional keys; verify the full config always loads.
            let f = write_temp(&full_toml());
            let result = load_config(f.path());
            prop_assert!(result.is_ok(), "Full config should load without error: {:?}", result);
        }
    }

    // -----------------------------------------------------------------------
    // Unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn full_config_loads_correctly() {
        let f = write_temp(&full_toml());
        let cfg = load_config(f.path()).unwrap();
        assert_eq!(cfg.dataset_name, "wikipedia");
        assert_eq!(cfg.num_documents, 1000);
        assert_eq!(cfg.chunk_size, 512);
        assert_eq!(cfg.top_k, 5);
    }

    #[test]
    fn missing_each_required_key_names_it_in_error() {
        for &key in REQUIRED_KEYS {
            let toml_content: String = full_toml()
                .lines()
                .filter(|line| !line.trim_start().starts_with(key))
                .collect::<Vec<_>>()
                .join("\n");

            let f = write_temp(&toml_content);
            let err = load_config(f.path()).unwrap_err();
            assert!(
                err.to_string().contains(key),
                "Error for missing '{}' should mention the key; got: {}",
                key,
                err
            );
        }
    }
}
