//! Dataset loader — downloads Wikipedia Simple dataset from HuggingFace Hub
//! and returns the first `num_docs` document text strings.

use std::fs::File;

use arrow::array::{Array, StringArray};
use hf_hub::api::sync::Api;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum DatasetError {
    #[error("Failed to load dataset '{dataset}/{subset}': {reason}")]
    LoadError {
        dataset: String,
        subset: String,
        reason: String,
    },
    #[error("Network error while fetching dataset '{dataset}/{subset}': {reason}")]
    NetworkError {
        dataset: String,
        subset: String,
        reason: String,
    },
}

/// Load the first `num_docs` documents from a HuggingFace dataset.
///
/// Downloads the dataset parquet files via `hf-hub` and extracts the `text`
/// column from each row batch until `num_docs` documents have been collected.
///
/// # Errors
/// Returns [`DatasetError::NetworkError`] if the hub API call fails (e.g. no
/// network), or [`DatasetError::LoadError`] if the parquet file cannot be
/// parsed.
pub fn load_documents(
    dataset_name: &str,
    subset: &str,
    num_docs: usize,
) -> Result<Vec<String>, DatasetError> {
    let api = Api::new().map_err(|e| DatasetError::NetworkError {
        dataset: dataset_name.to_string(),
        subset: subset.to_string(),
        reason: format!("Failed to initialise HuggingFace Hub API: {e}"),
    })?;

    // The wikipedia dataset stores parquet files under
    // `data/<subset>/train-*.parquet` inside the dataset repo.
    let repo = api.dataset(dataset_name.to_string());

    // List available parquet files for this subset.
    let info = repo.info().map_err(|e| DatasetError::NetworkError {
        dataset: dataset_name.to_string(),
        subset: subset.to_string(),
        reason: format!("Failed to fetch dataset info: {e}"),
    })?;

    // Filter siblings to find parquet files for the requested subset.
    let parquet_files: Vec<String> = info
        .siblings
        .iter()
        .filter_map(|s| {
            let name = &s.rfilename;
            if name.contains(subset) && name.ends_with(".parquet") {
                Some(name.clone())
            } else {
                None
            }
        })
        .collect();

    if parquet_files.is_empty() {
        return Err(DatasetError::LoadError {
            dataset: dataset_name.to_string(),
            subset: subset.to_string(),
            reason: format!("No parquet files found for subset '{subset}'"),
        });
    }

    let mut documents: Vec<String> = Vec::with_capacity(num_docs);

    'outer: for file_name in &parquet_files {
        let local_path = repo.get(file_name).map_err(|e| DatasetError::NetworkError {
            dataset: dataset_name.to_string(),
            subset: subset.to_string(),
            reason: format!("Failed to download '{file_name}': {e}"),
        })?;

        let file = File::open(&local_path).map_err(|e| DatasetError::LoadError {
            dataset: dataset_name.to_string(),
            subset: subset.to_string(),
            reason: format!("Failed to open parquet file '{}': {e}", local_path.display()),
        })?;

        let builder =
            ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| DatasetError::LoadError {
                dataset: dataset_name.to_string(),
                subset: subset.to_string(),
                reason: format!("Failed to read parquet metadata: {e}"),
            })?;

        let reader = builder.build().map_err(|e| DatasetError::LoadError {
            dataset: dataset_name.to_string(),
            subset: subset.to_string(),
            reason: format!("Failed to build parquet reader: {e}"),
        })?;

        for batch_result in reader {
            let batch = batch_result.map_err(|e| DatasetError::LoadError {
                dataset: dataset_name.to_string(),
                subset: subset.to_string(),
                reason: format!("Failed to read record batch: {e}"),
            })?;

            let text_col_idx = batch.schema().index_of("text").map_err(|_| {
                DatasetError::LoadError {
                    dataset: dataset_name.to_string(),
                    subset: subset.to_string(),
                    reason: "Column 'text' not found in dataset schema".to_string(),
                }
            })?;

            let text_array = batch
                .column(text_col_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| DatasetError::LoadError {
                    dataset: dataset_name.to_string(),
                    subset: subset.to_string(),
                    reason: "Column 'text' is not a string array".to_string(),
                })?;

            for i in 0..text_array.len() {
                if documents.len() >= num_docs {
                    break 'outer;
                }
                if !text_array.is_null(i) {
                    documents.push(text_array.value(i).to_string());
                }
            }
        }
    }

    Ok(documents)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal in-memory parquet file with a `text` column,
    /// returned as `bytes::Bytes` (which implements `ChunkReader`).
    fn make_parquet_bytes(texts: &[&str]) -> bytes::Bytes {
        use arrow::array::StringArray;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![Field::new("text", DataType::Utf8, false)]));
        let array = Arc::new(StringArray::from(texts.to_vec())) as Arc<dyn arrow::array::Array>;
        let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();

        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        bytes::Bytes::from(buf)
    }

    /// Helper: read documents from in-memory parquet bytes, limiting to num_docs.
    fn read_docs_from_bytes(parquet_bytes: bytes::Bytes, num_docs: usize) -> Vec<String> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(parquet_bytes).unwrap();
        let reader = builder.build().unwrap();

        let mut documents: Vec<String> = Vec::new();

        'outer: for batch_result in reader {
            let batch: arrow::record_batch::RecordBatch = batch_result.unwrap();
            let text_col_idx = batch.schema().index_of("text").unwrap();
            let text_array = batch
                .column(text_col_idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            for i in 0..text_array.len() {
                if documents.len() >= num_docs {
                    break 'outer;
                }
                if !text_array.is_null(i) {
                    documents.push(text_array.value(i).to_string());
                }
            }
        }

        documents
    }

    /// Test that the correct document count is returned when reading from a
    /// local parquet file (no network call).
    #[test]
    fn test_correct_document_count_from_parquet() {
        let texts: Vec<&str> = (0..20).map(|_| "sample text").collect();
        let parquet_bytes = make_parquet_bytes(&texts);

        let documents = read_docs_from_bytes(parquet_bytes, 10);

        assert_eq!(documents.len(), 10);
    }

    /// Test that a DatasetError is produced when the Hub API cannot be reached.
    /// We construct the error directly to verify the Display message is descriptive.
    #[test]
    fn test_network_error_produces_descriptive_error() {
        let err = DatasetError::NetworkError {
            dataset: "wikipedia".to_string(),
            subset: "20220301.simple".to_string(),
            reason: "Connection refused".to_string(),
        };

        let msg = err.to_string();
        assert!(
            msg.contains("wikipedia"),
            "Error message should contain dataset name: {msg}"
        );
        assert!(
            msg.contains("20220301.simple"),
            "Error message should contain subset: {msg}"
        );
        assert!(
            msg.contains("Connection refused"),
            "Error message should contain reason: {msg}"
        );
    }

    /// Test that LoadError also produces a descriptive message.
    #[test]
    fn test_load_error_produces_descriptive_error() {
        let err = DatasetError::LoadError {
            dataset: "wikipedia".to_string(),
            subset: "20220301.simple".to_string(),
            reason: "Column 'text' not found".to_string(),
        };

        let msg = err.to_string();
        assert!(msg.contains("wikipedia"), "Should contain dataset: {msg}");
        assert!(msg.contains("20220301.simple"), "Should contain subset: {msg}");
        assert!(msg.contains("Column 'text' not found"), "Should contain reason: {msg}");
    }

    /// Test that all documents are returned when the dataset has fewer rows
    /// than num_docs.
    #[test]
    fn test_returns_all_docs_when_fewer_than_num_docs() {
        let texts = vec!["doc1", "doc2", "doc3"];
        let parquet_bytes = make_parquet_bytes(&texts);

        let documents = read_docs_from_bytes(parquet_bytes, 1000);

        assert_eq!(documents.len(), 3);
    }

    /// Test that document text content is preserved correctly.
    #[test]
    fn test_document_text_content_preserved() {
        let texts = vec!["Hello world", "Foo bar", "Baz qux"];
        let parquet_bytes = make_parquet_bytes(&texts);

        let documents = read_docs_from_bytes(parquet_bytes, 10);

        assert_eq!(documents, vec!["Hello world", "Foo bar", "Baz qux"]);
    }
}
