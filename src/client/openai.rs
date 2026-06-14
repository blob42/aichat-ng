use super::*;

use crate::utils::strip_think_tag;

use anyhow::{bail, Context, Result};
use reqwest::RequestBuilder;
use serde::Deserialize;
use serde_json::{json, Value};

const API_BASE: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone, Deserialize, Default)]
pub struct OpenAIConfig {
    pub name: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub organization_id: Option<String>,
    #[serde(default)]
    pub models: Vec<ModelData>,
    pub patch: Option<RequestPatch>,
    pub extra: Option<ExtraConfig>,
}

impl OpenAIClient {
    config_get_fn!(api_key, get_api_key);
    config_get_fn!(api_base, get_api_base);

    pub const PROMPTS: [PromptAction<'static>; 1] = [("api_key", "API Key", None)];
}

impl_client_trait!(
    OpenAIClient,
    (
        prepare_chat_completions,
        openai_chat_completions,
        openai_chat_completions_streaming
    ),
    (prepare_embeddings, openai_embeddings),
    (noop_prepare_rerank, noop_rerank),
    (openai_audio_transcriptions),
);

fn prepare_chat_completions(
    self_: &OpenAIClient,
    data: ChatCompletionsData,
) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());

    let url = format!("{}/chat/completions", api_base.trim_end_matches('/'));

    let body = openai_build_chat_completions_body(data, &self_.model, true);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

fn prepare_embeddings(self_: &OpenAIClient, data: &EmbeddingsData) -> Result<RequestData> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());

    let url = format!("{api_base}/embeddings");

    let body = openai_build_embeddings_body(data, &self_.model);

    let mut request_data = RequestData::new(url, body);

    request_data.bearer_auth(api_key);
    if let Some(organization_id) = &self_.config.organization_id {
        request_data.header("OpenAI-Organization", organization_id);
    }

    Ok(request_data)
}

pub async fn openai_chat_completions(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<ChatCompletionsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }

    debug!("non-stream-data: {data}");
    openai_extract_chat_completions(&data)
}

pub async fn openai_chat_completions_streaming(
    builder: RequestBuilder,
    handler: &mut SseHandler,
    _model: &Model,
) -> Result<()> {
    let mut call_id = String::new();
    let mut function_name = String::new();
    let mut function_arguments = String::new();
    let mut function_id = String::new();
    let mut reasoning_state = 0;
    let handle = |message: SseMmessage| -> Result<bool> {
        if message.data == "[DONE]" {
            if !function_name.is_empty() {
                if function_arguments.is_empty() {
                    function_arguments = String::from("{}");
                }
                let arguments: Value = function_arguments.parse().with_context(|| {
                    format!("Tool call '{function_name}' have non-JSON arguments '{function_arguments}'")
                })?;
                handler.tool_call(ToolCall::new(
                    function_name.clone(),
                    arguments,
                    normalize_function_id(&function_id),
                ))?;
            }
            return Ok(true);
        }
        let data: Value = serde_json::from_str(&message.data)?;
        debug!("stream-data: {data}");
        if let Some(text) = data["choices"][0]["delta"]["content"]
            .as_str()
            .filter(|v| !v.is_empty())
        {
            if reasoning_state == 1 {
                handler.text("\n</think>\n\n")?;
                reasoning_state = 0;
            }
            handler.text(text)?;
        } else if let Some(text) = data["choices"][0]["delta"]["reasoning_content"]
            .as_str()
            .or_else(|| data["choices"][0]["delta"]["reasoning"].as_str())
            .filter(|v| !v.is_empty())
        {
            if reasoning_state == 0 {
                handler.text("<think>\n")?;
                reasoning_state = 1;
            }
            handler.text(text)?;
        }
        if let (Some(function), index, id) = (
            data["choices"][0]["delta"]["tool_calls"][0]["function"].as_object(),
            data["choices"][0]["delta"]["tool_calls"][0]["index"].as_u64(),
            data["choices"][0]["delta"]["tool_calls"][0]["id"]
                .as_str()
                .filter(|v| !v.is_empty()),
        ) {
            if reasoning_state == 1 {
                handler.text("\n</think>\n\n")?;
                reasoning_state = 0;
            }
            let maybe_call_id = format!("{}/{}", id.unwrap_or_default(), index.unwrap_or_default());
            if maybe_call_id != call_id && maybe_call_id.len() >= call_id.len() {
                if !function_name.is_empty() {
                    if function_arguments.is_empty() {
                        function_arguments = String::from("{}");
                    }
                    let arguments: Value = function_arguments.parse().with_context(|| {
                        format!("Tool call '{function_name}' have non-JSON arguments '{function_arguments}'")
                    })?;
                    handler.tool_call(ToolCall::new(
                        function_name.clone(),
                        arguments,
                        normalize_function_id(&function_id),
                    ))?;
                }
                function_name.clear();
                function_arguments.clear();
                function_id.clear();
                call_id = maybe_call_id;
            }
            if let Some(name) = function.get("name").and_then(|v| v.as_str()) {
                if name.starts_with(&function_name) {
                    function_name = name.to_string();
                } else {
                    function_name.push_str(name);
                }
            }
            if let Some(arguments) = function.get("arguments").and_then(|v| v.as_str()) {
                function_arguments.push_str(arguments);
            }
            if let Some(id) = id {
                function_id = id.to_string();
            }
        }
        Ok(false)
    };

    sse_stream(builder, handle).await
}

pub async fn openai_embeddings(
    builder: RequestBuilder,
    _model: &Model,
) -> Result<EmbeddingsOutput> {
    let res = builder.send().await?;
    let status = res.status();
    let data: Value = res.json().await?;
    if !status.is_success() {
        catch_error(&data, status.as_u16())?;
    }
    let res_body: EmbeddingsResBody =
        serde_json::from_value(data).context("Invalid embeddings data")?;
    let output = res_body.data.into_iter().map(|v| v.embedding).collect();
    Ok(output)
}

#[derive(Deserialize)]
struct EmbeddingsResBody {
    data: Vec<EmbeddingsResBodyEmbedding>,
}

#[derive(Deserialize)]
struct EmbeddingsResBodyEmbedding {
    embedding: Vec<f32>,
}

pub fn openai_build_chat_completions_body(
    data: ChatCompletionsData,
    model: &Model,
    native_audio: bool // true -> "input_audio", false - "audio_url"
) -> Value {
    let ChatCompletionsData {
        messages,
        temperature,
        top_p,
        functions,
        stream,
    } = data;

    let messages_len = messages.len();
    let messages: Vec<Value> = messages
        .into_iter()
        .enumerate()
        .flat_map(|(i, message)| {
            let Message { role, content } = message;
            let content_value = if native_audio {
                serialize_content_for_openai(&content)
            } else {
                serde_json::to_value(&content).unwrap_or_default()
            };
            match &content {
                MessageContent::ToolCalls(MessageContentToolCalls {
                    tool_results,
                    text: _,
                    sequence,
                }) => {
                    if !sequence {
                        let tool_calls: Vec<_> = tool_results
                            .iter()
                            .map(|tool_result| {
                                json!({
                                    "id": tool_result.call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_result.call.name,
                                        "arguments": tool_result.call.arguments.to_string(),
                                    },
                                })
                            })
                            .collect();
                        let mut messages = vec![
                            json!({ "role": MessageRole::Assistant, "tool_calls": tool_calls }),
                        ];
                        for tool_result in tool_results {
                            messages.push(json!({
                                "role": "tool",
                                "content": tool_result.output.to_string(),
                                "tool_call_id": tool_result.call.id,
                            }));
                        }
                        messages
                    } else {
                        tool_results.iter().flat_map(|tool_result| {
                            vec![
                                json!({
                                    "role": MessageRole::Assistant,
                                    "tool_calls": [
                                        {
                                            "id": tool_result.call.id,
                                            "type": "function",
                                            "function": {
                                                "name": tool_result.call.name,
                                                "arguments": tool_result.call.arguments.to_string(),
                                            },
                                        }
                                    ]
                                }),
                                json!({
                                    "role": "tool",
                                    "content": tool_result.output.to_string(),
                                    "tool_call_id": tool_result.call.id,
                                })
                            ]
                        }).collect()
                    }
                }
                MessageContent::Text(text) if role.is_assistant() && i != messages_len - 1 => {
                    vec![json!({ "role": role, "content": strip_think_tag(text) }
                    )]
                }
                _ => vec![json!({ "role": role, "content": content_value })],
            }
        })
        .collect();

    let mut body = json!({
        "model": &model.real_name(),
        "messages": messages,
    });

    if let Some(v) = model.max_tokens_param() {
        if model
            .patch()
            .and_then(|v| v.get("body").and_then(|v| v.get("max_tokens")))
            == Some(&Value::Null)
        {
            body["max_completion_tokens"] = v.into();
        } else {
            body["max_tokens"] = v.into();
        }
    }
    if let Some(v) = temperature {
        body["temperature"] = v.into();
    }
    if let Some(v) = top_p {
        body["top_p"] = v.into();
    }
    if stream {
        body["stream"] = true.into();
    }
    if let Some(functions) = functions {
        body["tools"] = functions
            .iter()
            .map(|v| {
                json!({
                    "type": "function",
                    "function": v,
                })
            })
            .collect();
    }
    body
}

/// Serialize MessageContent for OpenAI native format.
/// AudioUrl -> {"type": "input_audio", "input_audio": {"data": base64, "format": fmt}}
/// VideoUrl -> {"type": "image_url", "image_url": {"url": data_url}}
/// Other variants use standard serde serialization.
fn serialize_content_for_openai(content: &MessageContent) -> Value {
    match content {
        MessageContent::Text(text) => json!(text),
        MessageContent::Array(parts) => {
            let items: Vec<Value> = parts
                .iter()
                .map(|part| match part {
                    MessageContentPart::Text { text } => json!({ "type": "text", "text": text }),
                    MessageContentPart::ImageUrl { image_url } => {
                        json!({ "type": "image_url", "image_url": { "url": image_url.url } })
                    }
                    MessageContentPart::AudioUrl { audio_url } => {
                        let base64_data = extract_base64(&audio_url.url);
                        let format = extract_audio_format(&audio_url.url, &audio_url.mime_type);
                        json!({
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64_data,
                                "format": format,
                            },
                        })
                    }
                    MessageContentPart::VideoUrl { video_url } => {
                        json!({ "type": "image_url", "image_url": { "url": video_url.url } })
                    }
                })
                .collect();
            json!(items)
        }
        MessageContent::ToolCalls(_) => {
            serde_json::to_value(content).unwrap_or_default()
        }
    }
}

/// Extract base64 data from a data URL (e.g., "data:audio/mpeg;base64,abc123" -> "abc123")
fn extract_base64(url: &str) -> String {
    if let Some(stripped) = url.strip_prefix("data:") {
        if let Some(comma_pos) = stripped.find(',') {
            return stripped[comma_pos + 1..].to_string();
        }
    }
    url.to_string()
}

/// Extract the audio format from a data URL or mime_type (e.g., "mp3", "wav", "ogg")
fn extract_audio_format(url: &str, mime_type: &Option<String>) -> String {
    if let Some(mime) = mime_type {
        return mime_to_audio_format(mime);
    }
    if let Some(stripped) = url.strip_prefix("data:") {
        if let Some(semi_pos) = stripped.find(';') {
            let type_part = &stripped[..semi_pos];
            return mime_to_audio_format(type_part);
        }
    }
    "mp3".to_string()
}

/// Convert a MIME type to an OpenAI audio format string
fn mime_to_audio_format(mime: &str) -> String {
    match mime {
        "audio/mpeg" | "audio/mp3" => "mp3".to_string(),
        "audio/wav" | "audio/x-wav" => "wav".to_string(),
        "audio/ogg" | "audio/oga" => "ogg".to_string(),
        "audio/flac" => "flac".to_string(),
        "audio/m4a" | "audio/mp4" => "m4a".to_string(),
        "audio/webm" => "webm".to_string(),
        _ => "mp3".to_string(),
    }
}

pub fn openai_build_embeddings_body(data: &EmbeddingsData, model: &Model) -> Value {
    json!({
        "input": data.texts,
        "model": model.real_name()
    })
}

pub fn openai_extract_chat_completions(data: &Value) -> Result<ChatCompletionsOutput> {
    let text = data["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default();

    let reasoning = data["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .or_else(|| data["choices"][0]["message"]["reasoning"].as_str())
        .unwrap_or_default()
        .trim();

    let mut tool_calls = vec![];
    if let Some(calls) = data["choices"][0]["message"]["tool_calls"].as_array() {
        for call in calls {
            if let (Some(name), Some(arguments), Some(id)) = (
                call["function"]["name"].as_str(),
                call["function"]["arguments"].as_str(),
                call["id"].as_str(),
            ) {
                let arguments: Value = arguments.parse().with_context(|| {
                    format!("Tool call '{name}' have non-JSON arguments '{arguments}'")
                })?;
                tool_calls.push(ToolCall::new(
                    name.to_string(),
                    arguments,
                    Some(id.to_string()),
                ));
            }
        }
    };

    if text.is_empty() && tool_calls.is_empty() {
        bail!("Invalid response data: {data}");
    }
    let text = if !reasoning.is_empty() {
        format!("<think>\n{reasoning}\n</think>\n\n{text}")
    } else {
        text.to_string()
    };
    let output = ChatCompletionsOutput {
        text,
        tool_calls,
        id: data["id"].as_str().map(|v| v.to_string()),
        input_tokens: data["usage"]["prompt_tokens"].as_u64(),
        output_tokens: data["usage"]["completion_tokens"].as_u64(),
    };
    Ok(output)
}

fn normalize_function_id(value: &str) -> Option<String> {
    if value.is_empty() {
        None
    } else {
        Some(value.to_string())
    }
}

// Adapted from sigoden/aichat#1508 (authored by simon3z + Claude)
pub async fn openai_audio_transcriptions(
    self_: &OpenAIClient,
    client: &reqwest::Client,
    data: TranscriptionData,
) -> Result<String> {
    let api_key = self_.get_api_key()?;
    let api_base = self_
        .get_api_base()
        .unwrap_or_else(|_| API_BASE.to_string());
    let model_name = self_.model.real_name().to_string();
    openai_compatible_audio_transcriptions(client, &api_base, Some(&api_key), &model_name, data)
        .await
}

// Adapted from sigoden/aichat#1508 (authored by simon3z + Claude)
pub async fn openai_compatible_audio_transcriptions(
    client: &reqwest::Client,
    api_base: &str,
    api_key: Option<&str>,
    model_name: &str,
    data: TranscriptionData,
) -> Result<String> {
    let url = format!("{}/audio/transcriptions", api_base.trim_end_matches('/'));

    let file_bytes = tokio::fs::read(&data.path)
        .await
        .with_context(|| format!("Failed to read audio file '{}'", data.path.display()))?;

    let file_name = data
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio")
        .to_string();

    let mime_type = audio_mime_type(&data.path);

    let file_part = reqwest::multipart::Part::bytes(file_bytes)
        .file_name(file_name)
        .mime_str(&mime_type)
        .context("Invalid MIME type for audio file")?;

    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("model", model_name.to_string());

    if let Some(prompt) = data.prompt {
        form = form.text("prompt", prompt);
    }

    let mut request = client.post(&url).multipart(form);
    if let Some(key) = api_key {
        request = request.bearer_auth(key);
    }

    let res = request.send().await?;
    let status = res.status();
    let body: serde_json::Value = res.json().await?;

    if !status.is_success() {
        catch_error(&body, status.as_u16())?;
    }

    body["text"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow::anyhow!("Invalid transcription response: {body}"))
}

// Adapted from sigoden/aichat#1508 (authored by simon3z + Claude)
fn audio_mime_type(path: &std::path::Path) -> String {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "mp3" | "mpeg" | "mpga" => "audio/mpeg",
        "mp4" => "audio/mp4",
        "m4a" => "audio/m4a",
        "ogg" | "oga" => "audio/ogg",
        "wav" => "audio/wav",
        "webm" => "audio/webm",
        "flac" => "audio/flac",
        _ => "application/octet-stream",
    }
    .to_string()
}

#[cfg(test)]
mod openai_audio_video_tests {
    use super::*;
    use crate::client::message::{MessageContent, MessageContentPart, MediaUrl};

    #[test]
    fn test_build_body_with_audio_native() {
        let audio_url = MediaUrl {
            url: "data:audio/mpeg;base64,abc123".to_string(),
            mime_type: Some("audio/mpeg".to_string()),
        };
        let content = MessageContent::Array(vec![MessageContentPart::AudioUrl {
            audio_url,
        }]);
        let value = serialize_content_for_openai(&content);
        let arr = value.as_array().unwrap();
        assert_eq!(arr[0]["type"], "input_audio");
        assert_eq!(arr[0]["input_audio"]["data"], "abc123");
        assert_eq!(arr[0]["input_audio"]["format"], "mp3");
    }

    #[test]
    fn test_build_body_with_audio_compatible() {
        let audio_url = MediaUrl {
            url: "data:audio/mpeg;base64,abc123".to_string(),
            mime_type: Some("audio/mpeg".to_string()),
        };
        let content = MessageContent::Array(vec![MessageContentPart::AudioUrl {
            audio_url,
        }]);
        let value = serde_json::to_value(&content).unwrap();
        let arr = value.as_array().unwrap();
        assert_eq!(arr[0]["type"], "audio_url");
        assert_eq!(
            arr[0]["audio_url"]["url"],
            "data:audio/mpeg;base64,abc123"
        );
    }

    #[test]
    fn test_extract_audio_format_variants() {
        assert_eq!(mime_to_audio_format("audio/mpeg"), "mp3");
        assert_eq!(mime_to_audio_format("audio/mp3"), "mp3");
        assert_eq!(mime_to_audio_format("audio/wav"), "wav");
        assert_eq!(mime_to_audio_format("audio/x-wav"), "wav");
        assert_eq!(mime_to_audio_format("audio/ogg"), "ogg");
        assert_eq!(mime_to_audio_format("audio/flac"), "flac");
        assert_eq!(mime_to_audio_format("audio/m4a"), "m4a");
        assert_eq!(mime_to_audio_format("audio/webm"), "webm");
        assert_eq!(mime_to_audio_format("unknown/type"), "mp3");
    }

    #[test]
    fn test_extract_base64() {
        assert_eq!(extract_base64("data:audio/mpeg;base64,abc123"), "abc123");
        assert_eq!(extract_base64("data:audio/wav;base64,xyz789"), "xyz789");
        assert_eq!(extract_base64("not-a-data-url"), "not-a-data-url");
    }

    #[test]
    fn test_extract_audio_format_from_url() {
        assert_eq!(
            extract_audio_format("data:audio/mpeg;base64,abc", &None),
            "mp3"
        );
        assert_eq!(
            extract_audio_format("data:audio/wav;base64,abc", &None),
            "wav"
        );
        assert_eq!(
            extract_audio_format("data:audio/mpeg;base64,abc", &Some("audio/flac".to_string())),
            "flac"
        );
    }

    #[test]
    fn test_video_as_image_url_native() {
        let video_url = MediaUrl {
            url: "data:video/mp4;base64,xyz789".to_string(),
            mime_type: Some("video/mp4".to_string()),
        };
        let content = MessageContent::Array(vec![MessageContentPart::VideoUrl {
            video_url,
        }]);
        let value = serialize_content_for_openai(&content);
        let arr = value.as_array().unwrap();
        assert_eq!(arr[0]["type"], "image_url");
        assert_eq!(
            arr[0]["image_url"]["url"],
            "data:video/mp4;base64,xyz789"
        );
    }

    #[test]
    fn test_mixed_media_native() {
        let parts = vec![
            MessageContentPart::Text {
                text: "Hello".to_string(),
            },
            MessageContentPart::AudioUrl {
                audio_url: MediaUrl {
                    url: "data:audio/mpeg;base64,abc".to_string(),
                    mime_type: Some("audio/mpeg".to_string()),
                },
            },
        ];
        let content = MessageContent::Array(parts);
        let value = serialize_content_for_openai(&content);
        let arr = value.as_array().unwrap();
        assert_eq!(arr[0]["type"], "text");
        assert_eq!(arr[0]["text"], "Hello");
        assert_eq!(arr[1]["type"], "input_audio");
        assert_eq!(arr[1]["input_audio"]["data"], "abc");
        assert_eq!(arr[1]["input_audio"]["format"], "mp3");
    }
}
