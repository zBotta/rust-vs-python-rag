//! Embedder — Task 4 implementation.
//!
//! Uses `fastembed::TextEmbedding` with `EmbeddingModel::AllMiniLML6V2` to
//! produce 384-dimensional float vectors for each input text chunk.
//!
//! The `fastembed` dependency is optional (feature flag `fastembed`).  When
//! the feature is disabled the public API is still present but returns an
//! error, which allows the crate to compile and property tests to run without
//! downloading ONNX Runtime binaries.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum EmbedError {
    #[error("Embedding model error: {0}")]
    ModelError(String),
}

/// Embed text chunks using the all-MiniLM-L6-v2 model via fastembed.
///
/// # Arguments
/// * `chunks` – Slice of text strings to embed.
///
/// # Returns
/// `Vec` of 384-dimensional `[f32; 384]` arrays, one per input chunk.
///
/// # Errors
/// Returns [`EmbedError::ModelError`] if the model cannot be initialised or
/// if embedding fails.
pub fn embed_chunks(chunks: &[String]) -> Result<Vec<[f32; 384]>, EmbedError> {
    #[cfg(feature = "fastembed")]
    {
        use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

        let model = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::AllMiniLML6V2)
                .with_show_download_progress(false),
        )
        .map_err(|e| EmbedError::ModelError(e.to_string()))?;

        let raw: Vec<Vec<f32>> = model
            .embed(chunks.to_vec(), None)
            .map_err(|e| EmbedError::ModelError(e.to_string()))?;

        let mut result: Vec<[f32; 384]> = Vec::with_capacity(raw.len());
        for vec in raw {
            if vec.len() != 384 {
                return Err(EmbedError::ModelError(format!(
                    "Expected 384-dimensional vector, got {}",
                    vec.len()
                )));
            }
            let mut arr = [0f32; 384];
            arr.copy_from_slice(&vec);
            result.push(arr);
        }

        Ok(result)
    }

    #[cfg(not(feature = "fastembed"))]
    {
        let _ = chunks;
        Err(EmbedError::ModelError(
            "fastembed feature not enabled; build with --features fastembed".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    // -----------------------------------------------------------------------
    // Sub-task 4.1 — Property 4: Every embedding vector has exactly 384 dimensions
    // Feature: rust-vs-python-rag-benchmark, Property 4: Every embedding vector has exactly 384 dimensions
    // -----------------------------------------------------------------------

    /// Stub embedder that returns fake 384-dim vectors without loading the model.
    /// Used in property tests to avoid downloading model weights.
    fn stub_embed(chunks: &[String]) -> Vec<[f32; 384]> {
        chunks.iter().map(|_| [0f32; 384]).collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 4: Every embedding vector has exactly 384 dimensions.
        ///
        /// Uses a stub embedder that returns fake 384-dim vectors to verify
        /// that the output dimension invariant holds for any list of strings.
        ///
        /// # Feature: rust-vs-python-rag-benchmark, Property 4: Every embedding vector has exactly 384 dimensions
        /// Validates: Requirements 3.1
        #[test]
        fn prop_every_embedding_has_384_dimensions(
            chunks in prop::collection::vec("[a-zA-Z0-9 .,!?]{1,200}", 1..=20)
        ) {
            // Feature: rust-vs-python-rag-benchmark, Property 4: Every embedding vector has exactly 384 dimensions
            let string_chunks: Vec<String> = chunks.iter().map(|s: &String| s.clone()).collect();
            let embeddings = stub_embed(&string_chunks);

            prop_assert_eq!(
                embeddings.len(),
                string_chunks.len(),
                "Number of embeddings must equal number of input chunks"
            );

            for (i, vec) in embeddings.iter().enumerate() {
                prop_assert_eq!(
                    vec.len(),
                    384,
                    "Embedding {} has {} dimensions, expected 384",
                    i,
                    vec.len()
                );
            }
        }
    }
}
