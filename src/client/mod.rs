mod access_token;
mod common;
mod message;
#[macro_use]
mod macros;
mod model;
mod stream;

pub use crate::function::ToolCall;
pub use common::*;
pub use message::*;
pub use model::*;
pub use stream::*;

register_client!(
    (openai, "openai", OpenAIConfig, OpenAIClient),
    (
        openai_compatible,
        "openai-compatible",
        OpenAICompatibleConfig,
        OpenAICompatibleClient
    ),
    (gemini, "gemini", GeminiConfig, GeminiClient),
    (claude, "claude", ClaudeConfig, ClaudeClient),
    (cohere, "cohere", CohereConfig, CohereClient),
    (ollama, "ollama", OllamaConfig, OllamaClient),
    (
        azure_openai,
        "azure-openai",
        AzureOpenAIConfig,
        AzureOpenAIClient
    ),
    (vertexai, "vertexai", VertexAIConfig, VertexAIClient),
    (bedrock, "bedrock", BedrockConfig, BedrockClient),
);

pub const OPENAI_COMPATIBLE_PROVIDERS: [(&str, &str); 18] = [
    ("ai21", "https://api.ai21.com/studio/v1"),
    (
        "cloudflare",
        "https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1",
    ),
    ("deepinfra", "https://api.deepinfra.com/v1/openai"),
    ("deepseek", "https://api.deepseek.com"),
    ("ernie", "https://qianfan.baidubce.com/v2"),
    ("github", "https://models.inference.ai.azure.com"),
    ("groq", "https://api.groq.com/openai/v1"),
    ("hunyuan", "https://api.hunyuan.cloud.tencent.com/v1"),
    ("minimax", "https://api.minimax.chat/v1"),
    ("mistral", "https://api.mistral.ai/v1"),
    ("moonshot", "https://api.moonshot.cn/v1"),
    ("openrouter", "https://openrouter.ai/api/v1"),
    // ("ollama", "http://localhost:11434/v1"),
    ("perplexity", "https://api.perplexity.ai"),
    (
        "qianwen",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    ("xai", "https://api.x.ai/v1"),
    ("zhipuai", "https://open.bigmodel.cn/api/paas/v4"),
    // RAG-dedicated
    ("jina", "https://api.jina.ai/v1"),
    ("voyageai", "https://api.voyageai.com/v1"),
];

#[cfg(test)]
mod integration_audio_video {
    use super::*;
    use crate::client::message::{
        ImageUrl, MediaUrl, Message, MessageContent, MessageContentPart, MessageRole,
    };
    use crate::client::openai::openai_build_chat_completions_body;

    fn make_model(name: &str, supports_audio: bool, supports_video: bool) -> Model {
        let mut m = Model::new("openai", name);
        m.data_mut().supports_audio = supports_audio;
        m.data_mut().supports_video = supports_video;
        m
    }

    fn make_chat_data(messages: Vec<Message>, _model: &Model) -> ChatCompletionsData {
        ChatCompletionsData {
            messages,
            temperature: None,
            top_p: None,
            functions: None,
            stream: false,
        }
    }

    #[test]
    fn test_full_pipeline_audio_openai() {
        let model = make_model("gpt-4o", true, false);
        let audio_url = MediaUrl {
            url: "data:audio/mpeg;base64,abc123".to_string(),
            mime_type: Some("audio/mpeg".to_string()),
        };
        let content = MessageContent::Array(vec![
            MessageContentPart::Text {
                text: "Transcribe this".to_string(),
            },
            MessageContentPart::AudioUrl { audio_url },
        ]);
        let messages = vec![Message::new(MessageRole::User, content)];
        let data = make_chat_data(messages, &model);
        let body = openai_build_chat_completions_body(data, &model, true);

        let msgs = body["messages"].as_array().unwrap();
        let content_arr = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content_arr[0]["type"], "text");
        assert_eq!(content_arr[0]["text"], "Transcribe this");
        assert_eq!(content_arr[1]["type"], "input_audio");
        assert_eq!(content_arr[1]["input_audio"]["data"], "abc123");
        assert_eq!(content_arr[1]["input_audio"]["format"], "mp3");
    }

    #[test]
    fn test_full_pipeline_audio_openai_compatible() {
        let model = make_model("gpt-4o", true, false);
        let audio_url = MediaUrl {
            url: "data:audio/mpeg;base64,abc123".to_string(),
            mime_type: Some("audio/mpeg".to_string()),
        };
        let content = MessageContent::Array(vec![
            MessageContentPart::Text {
                text: "Transcribe this".to_string(),
            },
            MessageContentPart::AudioUrl { audio_url },
        ]);
        let messages = vec![Message::new(MessageRole::User, content)];
        let data = make_chat_data(messages, &model);
        let body = openai_build_chat_completions_body(data, &model, false);

        let msgs = body["messages"].as_array().unwrap();
        let content_arr = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content_arr[0]["type"], "text");
        assert_eq!(content_arr[1]["type"], "audio_url");
        assert_eq!(
            content_arr[1]["audio_url"]["url"],
            "data:audio/mpeg;base64,abc123"
        );
    }

    #[test]
    fn test_full_pipeline_mixed_media() {
        let model = make_model("gpt-4o", true, true);
        let parts = vec![
            MessageContentPart::Text {
                text: "Describe this".to_string(),
            },
            MessageContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: "data:image/png;base64,img123".to_string(),
                },
            },
            MessageContentPart::AudioUrl {
                audio_url: MediaUrl {
                    url: "data:audio/mpeg;base64,audio123".to_string(),
                    mime_type: Some("audio/mpeg".to_string()),
                },
            },
        ];
        let content = MessageContent::Array(parts);
        let messages = vec![Message::new(MessageRole::User, content)];
        let data = make_chat_data(messages, &model);
        let body = openai_build_chat_completions_body(data, &model, true);

        let msgs = body["messages"].as_array().unwrap();
        let content_arr = msgs[0]["content"].as_array().unwrap();
        assert_eq!(content_arr.len(), 3);
        assert_eq!(content_arr[0]["type"], "text");
        assert_eq!(content_arr[1]["type"], "image_url");
        assert_eq!(content_arr[2]["type"], "input_audio");
        assert_eq!(content_arr[2]["input_audio"]["data"], "audio123");
    }

    #[test]
    fn test_serialization_roundtrip() {
        // AudioUrl roundtrip
        let audio_part = MessageContentPart::AudioUrl {
            audio_url: MediaUrl {
                url: "data:audio/mpeg;base64,abc".to_string(),
                mime_type: Some("audio/mpeg".to_string()),
            },
        };
        let json = serde_json::to_value(&audio_part).unwrap();
        let deserialized: MessageContentPart = serde_json::from_value(json.clone()).unwrap();
        match deserialized {
            MessageContentPart::AudioUrl { audio_url } => {
                assert_eq!(audio_url.url, "data:audio/mpeg;base64,abc");
                assert_eq!(audio_url.mime_type, Some("audio/mpeg".to_string()));
            }
            _ => panic!("Expected AudioUrl"),
        }

        // VideoUrl roundtrip
        let video_part = MessageContentPart::VideoUrl {
            video_url: MediaUrl {
                url: "data:video/mp4;base64,xyz".to_string(),
                mime_type: None,
            },
        };
        let json = serde_json::to_value(&video_part).unwrap();
        let deserialized: MessageContentPart = serde_json::from_value(json).unwrap();
        match deserialized {
            MessageContentPart::VideoUrl { video_url } => {
                assert_eq!(video_url.url, "data:video/mp4;base64,xyz");
                assert_eq!(video_url.mime_type, None);
            }
            _ => panic!("Expected VideoUrl"),
        }

        // Full content roundtrip
        let content = MessageContent::Array(vec![
            MessageContentPart::Text {
                text: "Hello".to_string(),
            },
            MessageContentPart::AudioUrl {
                audio_url: MediaUrl {
                    url: "data:audio/mpeg;base64,abc".to_string(),
                    mime_type: None,
                },
            },
        ]);
        let json = serde_json::to_value(&content).unwrap();
        let deserialized: MessageContent = serde_json::from_value(json).unwrap();
        match deserialized {
            MessageContent::Array(parts) => {
                assert_eq!(parts.len(), 2);
                match &parts[0] {
                    MessageContentPart::Text { text } => assert_eq!(text, "Hello"),
                    _ => panic!("Expected Text"),
                }
                matches!(parts[1], MessageContentPart::AudioUrl { .. });
            }
            _ => panic!("Expected Array content"),
        }
    }
}
