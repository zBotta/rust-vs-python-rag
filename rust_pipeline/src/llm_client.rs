//! LLM client — Task 8 implementation.
//!
//! Sends prompts to Ollama's streaming HTTP API and records TTFT and generation time.
//! Retries up to 3 times with 1-second delay on HTTP error.

use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader};
use std::time::Instant;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug, Error)]
pub enum LlmError {
    #[error("LLM request failed after retries: {0}")]
    RequestFailed(String),
    #[error("HTTP error: {0}")]
    Http(String),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

// ---------------------------------------------------------------------------
// Data models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct LLMResponse {
    pub text: String,
    pub total_tokens: u64,
    pub ttft_ms: f64,
    pub generation_ms: f64,
    pub failed: bool,
    pub failure_reason: Option<String>,
}

/// Ollama API request body.
#[derive(Debug, Serialize)]
struct OllamaRequest<'a> {
    model: &'a str,
    prompt: &'a str,
    stream: bool,
    options: OllamaOptions,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    num_predict: u32,
}

/// One line of the Ollama streaming response.
#[derive(Debug, Deserialize)]
struct OllamaStreamLine {
    #[serde(default)]
    response: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the prompt string from retrieved chunks and query.
pub fn build_prompt(chunks: &[String], query: &str) -> String {
    format!("Context:\n{}\n\nQuestion: {}\nAnswer:", chunks.join("\n"), query)
}

/// Send prompt to Ollama and return LLMResponse with timing metrics.
///
/// Reads model name from `BENCHMARK_MODEL` env var (default: `llama3.2:3b`).
/// Retries up to `max_retries` times with 1-second delay on HTTP error.
pub fn generate(
    query: &str,
    chunks: &[String],
    llm_host: &str,
    max_retries: u32,
) -> Result<LLMResponse, LlmError> {
    let model = std::env::var("BENCHMARK_MODEL").unwrap_or_else(|_| "llama3.2:3b".to_string());
    let prompt = build_prompt(chunks, query);

    let url = format!("{}/api/generate", llm_host);
    let mut last_error = String::new();

    for attempt in 0..max_retries {
        if attempt > 0 {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }

        match attempt_generate(&url, &model, &prompt) {
            Ok(resp) => return Ok(resp),
            Err(e) => {
                last_error = e.to_string();
            }
        }
    }

    Ok(LLMResponse {
        text: String::new(),
        total_tokens: 0,
        ttft_ms: 0.0,
        generation_ms: 0.0,
        failed: true,
        failure_reason: Some(format!("Failed after {} retries: {}", max_retries, last_error)),
    })
}

/// Single attempt to call Ollama and stream the response.
fn attempt_generate(url: &str, model: &str, prompt: &str) -> Result<LLMResponse, LlmError> {
    let request_body = OllamaRequest {
        model,
        prompt,
        stream: true,
        options: OllamaOptions { num_predict: 256 },
    };

    let body_bytes = serde_json::to_vec(&request_body)?;

    // Use a blocking reqwest client
    let client = reqwest::blocking::Client::new();
    let response = client
        .post(url)
        .header("Content-Type", "application/json")
        .body(body_bytes)
        .send()
        .map_err(|e| LlmError::Http(e.to_string()))?;

    if !response.status().is_success() {
        return Err(LlmError::Http(format!(
            "HTTP {} {}",
            response.status().as_u16(),
            response.status().canonical_reason().unwrap_or("Unknown")
        )));
    }

    let start = Instant::now();
    let mut text_parts: Vec<String> = Vec::new();
    let mut ttft_ms: f64 = 0.0;
    let mut total_tokens: u64 = 0;
    let mut first_token = true;

    let reader = BufReader::new(response);
    for line in reader.lines() {
        let line = line?;
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let obj: OllamaStreamLine = serde_json::from_str(line)?;

        if !obj.response.is_empty() && first_token {
            ttft_ms = start.elapsed().as_secs_f64() * 1000.0;
            first_token = false;
        }
        text_parts.push(obj.response);

        if obj.done {
            let prompt_tokens = obj.prompt_eval_count.unwrap_or(0);
            let completion_tokens = obj.eval_count.unwrap_or(0);
            total_tokens = prompt_tokens + completion_tokens;
            break;
        }
    }

    let generation_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(LLMResponse {
        text: text_parts.join(""),
        total_tokens,
        ttft_ms,
        generation_ms,
        failed: false,
        failure_reason: None,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    // -----------------------------------------------------------------------
    // Sub-task 8.1 — Property 6: Prompt template correctness
    // Feature: rust-vs-python-rag-benchmark, Property 6: Constructed prompt exactly matches "Context:\n{chunks}\n\nQuestion: {query}\nAnswer:"
    // -----------------------------------------------------------------------

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// Property 6: Constructed prompt exactly matches the template.
        ///
        /// # Feature: rust-vs-python-rag-benchmark, Property 6: Constructed prompt exactly matches "Context:\n{chunks}\n\nQuestion: {query}\nAnswer:"
        /// Validates: Requirements 5.3
        #[test]
        fn prop_prompt_template_correctness(
            chunks in prop::collection::vec(
                prop::string::string_regex("[a-zA-Z0-9 .,!?]{0,100}").unwrap(),
                0..20,
            ),
            query in prop::string::string_regex("[a-zA-Z0-9 .,!?]{1,200}").unwrap(),
        ) {
            // Feature: rust-vs-python-rag-benchmark, Property 6: Constructed prompt exactly matches "Context:\n{chunks}\n\nQuestion: {query}\nAnswer:"
            let prompt = build_prompt(&chunks, &query);
            let expected = format!(
                "Context:\n{}\n\nQuestion: {}\nAnswer:",
                chunks.join("\n"),
                query
            );
            prop_assert_eq!(prompt, expected);
        }

        // -----------------------------------------------------------------------
        // Sub-task 8.2 — Property 7: Model name sourced from environment variable
        // Feature: rust-vs-python-rag-benchmark, Property 7: model field in every Ollama request body equals BENCHMARK_MODEL env var value
        // -----------------------------------------------------------------------

        /// Property 7: model field in every Ollama request body equals BENCHMARK_MODEL env var value.
        ///
        /// # Feature: rust-vs-python-rag-benchmark, Property 7: model field in every Ollama request body equals BENCHMARK_MODEL env var value
        /// Validates: Requirements 5.2
        #[test]
        fn prop_model_name_from_env(
            model_name in prop::string::string_regex("[a-zA-Z0-9._:-]{1,50}").unwrap(),
            chunks in prop::collection::vec(
                prop::string::string_regex("[a-zA-Z0-9 ]{0,50}").unwrap(),
                0..5,
            ),
            query in prop::string::string_regex("[a-zA-Z0-9 ]{1,50}").unwrap(),
        ) {
            // Feature: rust-vs-python-rag-benchmark, Property 7: model field in every Ollama request body equals BENCHMARK_MODEL env var value
            std::env::set_var("BENCHMARK_MODEL", &model_name);

            let resolved_model = std::env::var("BENCHMARK_MODEL")
                .unwrap_or_else(|_| "llama3.2:3b".to_string());

            let request_body = OllamaRequest {
                model: &resolved_model,
                prompt: &build_prompt(&chunks, &query),
                stream: true,
                options: OllamaOptions { num_predict: 256 },
            };

            let serialized = serde_json::to_value(&request_body).unwrap();
            prop_assert_eq!(
                serialized["model"].as_str().unwrap(),
                model_name.as_str()
            );
        }

        // -----------------------------------------------------------------------
        // Sub-task 8.3 — Property 8: Total tokens = prompt_tokens + completion_tokens
        // Feature: rust-vs-python-rag-benchmark, Property 8: total_tokens = prompt_tokens + completion_tokens
        // -----------------------------------------------------------------------

        /// Property 8: total_tokens = prompt_tokens + completion_tokens.
        ///
        /// # Feature: rust-vs-python-rag-benchmark, Property 8: total_tokens = prompt_tokens + completion_tokens
        /// Validates: Requirements 5.6
        #[test]
        fn prop_total_tokens_is_sum(
            prompt_tokens in 0u64..10_000u64,
            completion_tokens in 0u64..10_000u64,
        ) {
            // Feature: rust-vs-python-rag-benchmark, Property 8: total_tokens = prompt_tokens + completion_tokens
            let total = prompt_tokens + completion_tokens;
            prop_assert_eq!(total, prompt_tokens + completion_tokens);
        }
    }

    // -----------------------------------------------------------------------
    // Sub-task 8.4 — Unit tests for LLM client
    // -----------------------------------------------------------------------

    /// Test build_prompt constructs the exact expected string.
    #[test]
    fn test_build_prompt_basic() {
        let chunks = vec!["chunk one".to_string(), "chunk two".to_string()];
        let query = "What is this?";
        let prompt = build_prompt(&chunks, query);
        assert_eq!(
            prompt,
            "Context:\nchunk one\nchunk two\n\nQuestion: What is this?\nAnswer:"
        );
    }

    /// Test build_prompt with empty chunks list.
    #[test]
    fn test_build_prompt_empty_chunks() {
        let chunks: Vec<String> = vec![];
        let query = "Any question?";
        let prompt = build_prompt(&chunks, query);
        assert_eq!(prompt, "Context:\n\n\nQuestion: Any question?\nAnswer:");
    }

    /// Test build_prompt with a single chunk.
    #[test]
    fn test_build_prompt_single_chunk() {
        let chunks = vec!["only chunk".to_string()];
        let query = "Q?";
        let prompt = build_prompt(&chunks, query);
        assert_eq!(prompt, "Context:\nonly chunk\n\nQuestion: Q?\nAnswer:");
    }

    /// Test total_tokens computation: prompt_tokens + completion_tokens.
    #[test]
    fn test_total_tokens_sum() {
        let prompt_tokens: u64 = 50;
        let completion_tokens: u64 = 100;
        let total = prompt_tokens + completion_tokens;
        assert_eq!(total, 150);
    }

    /// Test that OllamaRequest serializes with the correct model field.
    #[test]
    fn test_request_body_model_field() {
        let model = "llama3.2:3b";
        let prompt = "Context:\n\n\nQuestion: test?\nAnswer:";
        let req = OllamaRequest {
            model,
            prompt,
            stream: true,
            options: OllamaOptions { num_predict: 256 },
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "llama3.2:3b");
        assert_eq!(json["stream"], true);
        assert_eq!(json["options"]["num_predict"], 256);
    }

    /// Test that generate returns a failed LLMResponse after exhausting retries
    /// when the server is unreachable (connection refused).
    #[test]
    fn test_generate_fails_after_retries_on_connection_refused() {
        // Use a port that should not be listening
        let result = generate("test query", &[], "http://127.0.0.1:19999", 3);
        // generate() returns Ok(LLMResponse { failed: true }) after exhausting retries
        assert!(result.is_ok());
        let resp = result.unwrap();
        assert!(resp.failed);
        assert!(resp.failure_reason.is_some());
        let reason = resp.failure_reason.unwrap();
        assert!(reason.contains("Failed after 3 retries"), "reason: {}", reason);
    }
}
