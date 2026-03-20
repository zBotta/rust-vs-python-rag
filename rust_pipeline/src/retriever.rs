//! Retriever: queries the vector store and returns top-k chunk texts.

use thiserror::Error;

use crate::vector_store::{VectorStore, VectorStoreError};

#[derive(Debug, Error)]
pub enum RetrieverError {
    #[error("vector store error: {0}")]
    VectorStore(#[from] VectorStoreError),
}

pub struct Retriever {
    chunks: Vec<String>,
    vector_store: VectorStore,
}

impl Retriever {
    pub fn new(chunks: Vec<String>, vector_store: VectorStore) -> Self {
        Self { chunks, vector_store }
    }

    pub fn retrieve(
        &self,
        query_embedding: &[f32; 384],
        top_k: usize,
    ) -> Result<Vec<String>, RetrieverError> {
        let results = self.vector_store.query(query_embedding, top_k)?;
        let texts = results
            .into_iter()
            .map(|(id, _score)| self.chunks[id].clone())
            .collect();
        Ok(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_store::VectorStore;
    use proptest::prelude::*;

    // Feature: rust-vs-python-rag-benchmark, Property 5: Retriever returns exactly top_k results when index has >= top_k entries

    /// Generate a random 384-dim unit vector.
    fn random_embedding(seed: u64) -> [f32; 384] {
        // Simple deterministic pseudo-random using seed
        let mut arr = [0f32; 384];
        let mut state = seed.wrapping_add(1);
        for x in arr.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *x = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        }
        arr
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_retriever_returns_exactly_top_k(
            // n_extra: how many extra embeddings beyond top_k (0..=20)
            n_extra in 0usize..=20usize,
            top_k in 1usize..=10usize,
            query_seed in 0u64..1000u64,
        ) {
            let n = top_k + n_extra; // n >= top_k
            let mut embeddings: Vec<[f32; 384]> = (0..n)
                .map(|i| random_embedding(i as u64 + 1))
                .collect();

            let chunks: Vec<String> = (0..n).map(|i| format!("chunk_{}", i)).collect();

            let mut vs = VectorStore::new(384);
            vs.build_index(&embeddings).expect("build_index failed");

            let retriever = Retriever::new(chunks, vs);
            let query_emb = random_embedding(query_seed + 1000);
            let results = retriever.retrieve(&query_emb, top_k).expect("retrieve failed");

            prop_assert_eq!(results.len(), top_k);
        }
    }
}
