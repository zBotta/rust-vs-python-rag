//! In-memory HNSW vector store using instant-distance.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum VectorStoreError {
    #[error("index not built — call build_index first")]
    NotBuilt,
    #[error("embeddings list is empty")]
    EmptyEmbeddings,
    #[error("top_k ({top_k}) exceeds number of indexed items ({count})")]
    TopKExceedsCount { top_k: usize, count: usize },
}

/// A single 384-dimensional point stored in the index.
#[derive(Clone, Debug)]
struct Point([f32; 384]);

impl instant_distance::Point for Point {
    fn distance(&self, other: &Self) -> f32 {
        // Cosine distance = 1 - cosine_similarity
        let dot: f32 = self.0.iter().zip(other.0.iter()).map(|(a, b)| a * b).sum();
        let norm_a: f32 = self.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.0.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }
        1.0 - (dot / (norm_a * norm_b))
    }
}

pub struct VectorStore {
    dim: usize,
    /// Stored embeddings (used to rebuild search after construction).
    embeddings: Vec<[f32; 384]>,
    /// The built HNSW map, if any.
    hnsw: Option<instant_distance::HnswMap<Point, usize>>,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            embeddings: Vec::new(),
            hnsw: None,
        }
    }

    pub fn build_index(&mut self, embeddings: &[[f32; 384]]) -> Result<(), VectorStoreError> {
        if embeddings.is_empty() {
            return Err(VectorStoreError::EmptyEmbeddings);
        }
        let points: Vec<Point> = embeddings.iter().map(|e| Point(*e)).collect();
        let values: Vec<usize> = (0..points.len()).collect();
        let mut search = instant_distance::Search::default();
        let hnsw = instant_distance::Builder::default().build(points, values);
        self.embeddings = embeddings.to_vec();
        self.hnsw = Some(hnsw);
        Ok(())
    }

    pub fn query(
        &self,
        embedding: &[f32; 384],
        top_k: usize,
    ) -> Result<Vec<(usize, f32)>, VectorStoreError> {
        let hnsw = self.hnsw.as_ref().ok_or(VectorStoreError::NotBuilt)?;
        let count = self.embeddings.len();
        if top_k > count {
            return Err(VectorStoreError::TopKExceedsCount { top_k, count });
        }
        let query_point = Point(*embedding);
        let mut search = instant_distance::Search::default();
        let results: Vec<(usize, f32)> = hnsw
            .search(&query_point, &mut search)
            .take(top_k)
            .map(|item| (*item.value, item.distance))
            .collect();
        Ok(results)
    }
}
