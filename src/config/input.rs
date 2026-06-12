use super::*;

use crate::client::{
    init_client, patch_messages, ChatCompletionsData, Client, ImageUrl, MediaUrl, Message,
    MessageContent, MessageContentPart, MessageContentToolCalls, MessageRole, Model,
};
use crate::function::ToolResult;
use crate::utils::{base64_encode, is_loader_protocol, sha256, AbortSignal};

use anyhow::{bail, Context, Result};
use indexmap::IndexSet;
use std::{collections::HashMap, env, fs::File, io::Read};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

const IMAGE_EXTS: [&str; 5] = ["png", "jpeg", "jpg", "webp", "gif"];
const AUDIO_EXTS: [&str; 7] = ["mp3", "wav", "ogg", "flac", "m4a", "webm", "mp4"];
const VIDEO_EXTS: [&str; 5] = ["mp4", "webm", "avi", "mov", "mkv"];
const SUMMARY_MAX_WIDTH: usize = 80;
const DEFAULT_MAX_MEDIA_SIZE_MB: usize = 25;

#[derive(Debug, Clone, PartialEq)]
enum MediaType {
    Image,
    Audio,
    Video,
}

#[derive(Debug, Clone)]
pub enum Regenerate {
    // Simply regenerate last reponse
    Simple,

    // Edit LLM response
    Edit(String),
}

#[derive(Debug, Clone)]
pub struct Input {
    config: GlobalConfig,
    text: String,
    raw: (String, Vec<String>),
    patched_text: Option<String>,
    last_reply: Option<String>,
    continue_output: Option<String>,
    regenerate: Option<Regenerate>,
    media_parts: Vec<MessageContentPart>,
    data_urls: HashMap<String, String>,
    tool_calls: Option<MessageContentToolCalls>,
    role: Role,
    rag_name: Option<String>,
    with_session: bool,
    with_agent: bool,
}

impl Input {
    pub fn from_str(config: &GlobalConfig, text: &str, role: Option<Role>) -> Self {
        let (role, with_session, with_agent) = resolve_role(&config.read(), role);
        Self {
            config: config.clone(),
            text: text.to_string(),
            raw: (text.to_string(), vec![]),
            patched_text: None,
            last_reply: None,
            continue_output: None,
            regenerate: None,
            media_parts: Default::default(),
            data_urls: Default::default(),
            tool_calls: None,
            role,
            rag_name: None,
            with_session,
            with_agent,
        }
    }

    pub async fn from_files(
        config: &GlobalConfig,
        raw_text: &str,
        paths: Vec<String>,
        role: Option<Role>,
    ) -> Result<Self> {
        let loaders = config.read().document_loaders.clone();
        let (raw_paths, local_paths, remote_urls, external_cmds, protocol_paths, with_last_reply) =
            resolve_paths(&loaders, paths)?;
        let mut last_reply = None;
        let (documents, media_parts, data_urls) = load_documents(
            &loaders,
            local_paths,
            remote_urls,
            external_cmds,
            protocol_paths,
        )
        .await
        .context("Failed to load files")?;
        let mut texts = vec![];
        if !raw_text.is_empty() {
            texts.push(raw_text.to_string());
        };
        if with_last_reply {
            if let Some(LastMessage { input, output, .. }) = config.read().last_message.as_ref() {
                if !output.is_empty() {
                    last_reply = Some(output.clone())
                } else if let Some(v) = input.last_reply.as_ref() {
                    last_reply = Some(v.clone());
                }
                if let Some(v) = last_reply.clone() {
                    texts.push(format!("\n{v}"));
                }
            }
            if last_reply.is_none() && documents.is_empty() && media_parts.is_empty() {
                bail!("No last reply found");
            }
        }
        let documents_len = documents.len();
        for (kind, path, contents) in documents {
            if documents_len == 1 && raw_text.is_empty() {
                texts.push(format!("\n{contents}"));
            } else {
                texts.push(format!(
                    "\n============ {kind}: {path} ============\n{contents}"
                ));
            }
        }
        let (role, with_session, with_agent) = resolve_role(&config.read(), role);
        Ok(Self {
            config: config.clone(),
            text: texts.join("\n"),
            raw: (raw_text.to_string(), raw_paths),
            patched_text: None,
            last_reply,
            continue_output: None,
            regenerate: None,
            media_parts,
            data_urls,
            tool_calls: Default::default(),
            role,
            rag_name: None,
            with_session,
            with_agent,
        })
    }

    pub async fn from_files_with_spinner(
        config: &GlobalConfig,
        raw_text: &str,
        paths: Vec<String>,
        role: Option<Role>,
        abort_signal: AbortSignal,
    ) -> Result<Self> {
        abortable_run_with_spinner(
            Input::from_files(config, raw_text, paths, role),
            "Loading files",
            abort_signal,
        )
        .await
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.media_parts.is_empty()
    }
    pub fn data_urls(&self) -> HashMap<String, String> {
        self.data_urls.clone()
    }

    pub fn tool_calls(&self) -> &Option<MessageContentToolCalls> {
        &self.tool_calls
    }

    pub fn text(&self) -> String {
        match self.patched_text.clone() {
            Some(text) => text,
            None => self.text.clone(),
        }
    }

    pub fn clear_patch(&mut self) {
        self.patched_text = None;
    }

    pub fn set_text(&mut self, text: String) {
        self.text = text;
    }

    pub fn stream(&self) -> bool {
        self.config.read().stream && !self.role().model().no_stream()
    }

    pub fn continue_output(&self) -> Option<&str> {
        self.continue_output.as_deref()
    }

    pub fn set_continue_output(&mut self, output: &str) {
        let output = match &self.continue_output {
            Some(v) => format!("{v}{output}"),
            None => output.to_string(),
        };
        self.continue_output = Some(output);
    }

    pub fn regenerate(&self) -> Option<Regenerate> {
        self.regenerate.clone()
    }

    pub fn set_regenerate(&mut self, output: Option<String>) {
        let role = self.config.read().extract_role();
        if role.name() == self.role().name() {
            self.role = role;
        }

        self.regenerate = match output {
            Some(output) => Some(Regenerate::Edit(output)),
            None => Some(Regenerate::Simple),
        };
        self.tool_calls = None;
    }

    pub async fn use_embeddings(&mut self, abort_signal: AbortSignal) -> Result<()> {
        if self.text.is_empty() {
            return Ok(());
        }
        let rag = self.config.read().rag.clone();
        if let Some(rag) = rag {
            let result = Config::search_rag(&self.config, &rag, &self.text, abort_signal).await?;
            self.patched_text = Some(result);
            self.rag_name = Some(rag.name().to_string());
        }
        Ok(())
    }

    pub fn rag_name(&self) -> Option<&str> {
        self.rag_name.as_deref()
    }

    pub fn merge_tool_results(mut self, output: String, tool_results: Vec<ToolResult>) -> Self {
        match self.tool_calls.as_mut() {
            Some(exist_tool_results) => {
                exist_tool_results.merge(tool_results, output);
            }
            None => self.tool_calls = Some(MessageContentToolCalls::new(tool_results, output)),
        }
        self
    }

    pub fn create_client(&self) -> Result<Box<dyn Client>> {
        init_client(&self.config, Some(self.role().model().clone()))
    }

    pub async fn fetch_chat_text(&self) -> Result<String> {
        let client = self.create_client()?;
        let text = client.chat_completions(self.clone()).await?.text;
        let text = strip_think_tag(&text).to_string();
        Ok(text)
    }

    fn guard_media_capabilities(&self, model: &Model) -> Result<()> {
        for part in &self.media_parts {
            match part {
                MessageContentPart::AudioUrl { .. } => {
                    if !model.supports_audio() {
                        bail!(
                            "Model '{}' does not support audio input. Use a model with audio support (e.g., gpt-4o) or use .transcript to transcribe first.",
                            model.id()
                        )
                    }
                }
                MessageContentPart::VideoUrl { .. } => {
                    if !model.supports_video() {
                        bail!(
                            "Model '{}' does not support video input. Use a model with video support.",
                            model.id()
                        )
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn prepare_completion_data(
        &self,
        model: &Model,
        stream: bool,
    ) -> Result<ChatCompletionsData> {
        self.guard_media_capabilities(model)?;
        let mut messages = self.build_messages()?;
        patch_messages(&mut messages, model);
        model.guard_max_input_tokens(&messages)?;
        let (temperature, top_p) = (self.role().temperature(), self.role().top_p());
        let functions = self.config.read().select_functions(self.role());
        Ok(ChatCompletionsData {
            messages,
            temperature,
            top_p,
            functions,
            stream,
        })
    }

    pub fn build_messages(&self) -> Result<Vec<Message>> {
        let mut messages = if let Some(session) = self.session(&self.config.read().session) {
            session.build_messages(self)
        } else {
            self.role().build_messages(self)
        };
        if let Some(tool_calls) = &self.tool_calls {
            messages.push(Message::new(
                MessageRole::Assistant,
                MessageContent::ToolCalls(tool_calls.clone()),
            ))
        }
        Ok(messages)
    }

    pub fn echo_messages(&self) -> String {
        if let Some(session) = self.session(&self.config.read().session) {
            session.echo_messages(self)
        } else {
            self.role().echo_messages(self)
        }
    }

    pub fn role(&self) -> &Role {
        &self.role
    }

    pub fn session<'a>(&self, session: &'a Option<Session>) -> Option<&'a Session> {
        if self.with_session {
            session.as_ref()
        } else {
            None
        }
    }

    pub fn session_mut<'a>(&self, session: &'a mut Option<Session>) -> Option<&'a mut Session> {
        if self.with_session {
            session.as_mut()
        } else {
            None
        }
    }

    pub fn with_agent(&self) -> bool {
        self.with_agent
    }

    pub fn summary(&self) -> String {
        let text: String = self
            .text
            .trim()
            .chars()
            .map(|c| if c.is_control() { ' ' } else { c })
            .collect();
        if text.width_cjk() > SUMMARY_MAX_WIDTH {
            let mut sum_width = 0;
            let mut chars = vec![];
            for c in text.chars() {
                sum_width += c.width_cjk().unwrap_or(1);
                if sum_width > SUMMARY_MAX_WIDTH - 3 {
                    chars.extend(['.', '.', '.']);
                    break;
                }
                chars.push(c);
            }
            chars.into_iter().collect()
        } else {
            text
        }
    }

    pub fn raw(&self) -> String {
        let (text, files) = &self.raw;
        let mut segments = files.to_vec();
        if !segments.is_empty() {
            segments.insert(0, ".file".into());
        }
        if !text.is_empty() {
            if !segments.is_empty() {
                segments.push("--".into());
            }
            segments.push(text.clone());
        }
        segments.join(" ")
    }

    pub fn render(&self) -> String {
        let text = self.text();
        if self.media_parts.is_empty() {
            return text;
        }
        let tail_text = if text.is_empty() {
            String::new()
        } else {
            format!(" -- {text}")
        };
        let files: Vec<String> = self
            .media_parts
            .iter()
            .map(|part| match part {
                MessageContentPart::ImageUrl { image_url } => image_url.url.clone(),
                MessageContentPart::AudioUrl { audio_url } => audio_url.url.clone(),
                MessageContentPart::VideoUrl { video_url } => video_url.url.clone(),
                MessageContentPart::Text { .. } => unreachable!(),
            })
            .map(|url| resolve_data_url(&self.data_urls, url))
            .collect();
        format!(".file {}{}", files.join(" "), tail_text)
    }

    pub fn message_content(&self) -> MessageContent {
        if self.media_parts.is_empty() {
            MessageContent::Text(self.text())
        } else {
            let mut list = self.media_parts.clone();
            if !self.text.is_empty() {
                list.insert(0, MessageContentPart::Text { text: self.text() });
            }
            MessageContent::Array(list)
        }
    }
}

fn resolve_role(config: &Config, role: Option<Role>) -> (Role, bool, bool) {
    match role {
        Some(v) => (v, false, false),
        None => (
            config.extract_role(),
            config.session.is_some(),
            config.agent.is_some(),
        ),
    }
}

type ResolvePathsOutput = (
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    Vec<String>,
    bool,
);

fn resolve_paths(
    loaders: &HashMap<String, String>,
    paths: Vec<String>,
) -> Result<ResolvePathsOutput> {
    let mut raw_paths = IndexSet::new();
    let mut local_paths = IndexSet::new();
    let mut remote_urls = IndexSet::new();
    let mut external_cmds = IndexSet::new();
    let mut protocol_paths = IndexSet::new();
    let mut with_last_reply = false;
    for path in paths {
        if path == "%%" {
            with_last_reply = true;
            raw_paths.insert(path);
        } else if path.starts_with('`') && path.len() > 2 && path.ends_with('`') {
            external_cmds.insert(path[1..path.len() - 1].to_string());
            raw_paths.insert(path);
        } else if is_url(&path) {
            if path.strip_suffix("**").is_some() {
                bail!("Invalid website '{path}'");
            }
            remote_urls.insert(path.clone());
            raw_paths.insert(path);
        } else if is_loader_protocol(loaders, &path) {
            protocol_paths.insert(path.clone());
            raw_paths.insert(path);
        } else {
            let resolved_path = resolve_home_dir(&path);
            let absolute_path = to_absolute_path(&resolved_path)
                .with_context(|| format!("Invalid path '{path}'"))?;
            local_paths.insert(resolved_path);
            raw_paths.insert(absolute_path);
        }
    }
    Ok((
        raw_paths.into_iter().collect(),
        local_paths.into_iter().collect(),
        remote_urls.into_iter().collect(),
        external_cmds.into_iter().collect(),
        protocol_paths.into_iter().collect(),
        with_last_reply,
    ))
}

async fn load_documents(
    loaders: &HashMap<String, String>,
    local_paths: Vec<String>,
    remote_urls: Vec<String>,
    external_cmds: Vec<String>,
    protocol_paths: Vec<String>,
) -> Result<(
    Vec<(&'static str, String, String)>,
    Vec<MessageContentPart>,
    HashMap<String, String>,
)> {
    let mut files = vec![];
    let mut media_parts = vec![];
    let mut data_urls = HashMap::new();

    for cmd in external_cmds {
        let output = duct::cmd(&SHELL.cmd, &[&SHELL.arg, &cmd])
            .stderr_to_stdout()
            .unchecked()
            .read()
            .unwrap_or_else(|err| err.to_string());
        files.push(("CMD", cmd, output));
    }

    let local_files = expand_glob_paths(&local_paths, true).await?;
    for file_path in local_files {
        check_media_size(&file_path)?;
        let media_type = detect_media_type(&file_path);
        if media_type != MediaType::Image {
            let (contents, part_type) = read_media_with_type(&file_path)
                .with_context(|| format!("Unable to read media '{file_path}'"))?;
            data_urls.insert(sha256(&contents), file_path.clone());
            let part = match part_type {
                MediaType::Audio => MessageContentPart::AudioUrl {
                    audio_url: MediaUrl {
                        url: contents,
                        mime_type: None,
                    },
                },
                MediaType::Video => MessageContentPart::VideoUrl {
                    video_url: MediaUrl {
                        url: contents,
                        mime_type: None,
                    },
                },
                MediaType::Image => unreachable!(),
            };
            media_parts.push(part)
        } else if is_image(&file_path) {
            let contents = read_media_to_data_url(&file_path)
                .with_context(|| format!("Unable to read media '{file_path}'"))?;
            data_urls.insert(sha256(&contents), file_path.clone());
            media_parts.push(MessageContentPart::ImageUrl {
                image_url: ImageUrl {
                    url: contents,
                },
            })
        } else {
            let document = load_file(loaders, &file_path)
                .await
                .with_context(|| format!("Unable to read file '{file_path}'"))?;
            files.push(("FILE", file_path, document.contents));
        }
    }

    for file_url in remote_urls {
        let (contents, extension) = fetch_with_loaders(loaders, &file_url, true)
            .await
            .with_context(|| format!("Failed to load url '{file_url}'"))?;
        if extension == MEDIA_URL_EXTENSION {
            data_urls.insert(sha256(&contents), file_url.clone());
            let media_type = media_type_from_mime(&contents);
            let part = match media_type {
                MediaType::Audio => MessageContentPart::AudioUrl {
                    audio_url: MediaUrl {
                        url: contents,
                        mime_type: None,
                    },
                },
                MediaType::Video => MessageContentPart::VideoUrl {
                    video_url: MediaUrl {
                        url: contents,
                        mime_type: None,
                    },
                },
                MediaType::Image => MessageContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: contents,
                    },
                },
            };
            media_parts.push(part)
        } else {
            files.push(("URL", file_url, contents));
        }
    }

    for protocol_path in protocol_paths {
        let documents = load_protocol_path(loaders, &protocol_path)
            .with_context(|| format!("Failed to load from '{protocol_path}'"))?;
        files.extend(
            documents
                .into_iter()
                .map(|document| ("FROM", document.path, document.contents)),
        );
    }

    Ok((files, media_parts, data_urls))
}

pub fn resolve_data_url(data_urls: &HashMap<String, String>, data_url: String) -> String {
    if data_url.starts_with("data:") {
        let hash = sha256(&data_url);
        if let Some(path) = data_urls.get(&hash) {
            return path.to_string();
        }
        data_url
    } else {
        data_url
    }
}

fn is_image(path: &str) -> bool {
    get_patch_extension(path)
        .map(|v| IMAGE_EXTS.contains(&v.as_str()))
        .unwrap_or_default()
}

fn is_audio(path: &str) -> bool {
    get_patch_extension(path)
        .map(|v| AUDIO_EXTS.contains(&v.as_str()))
        .unwrap_or_default()
}

fn is_video(path: &str) -> bool {
    get_patch_extension(path)
        .map(|v| VIDEO_EXTS.contains(&v.as_str()))
        .unwrap_or_default()
}

fn detect_media_type(path: &str) -> MediaType {
    if is_image(path) {
        MediaType::Image
    } else if is_audio(path) {
        // .mp4 and .webm default to audio
        MediaType::Audio
    } else if is_video(path) {
        MediaType::Video
    } else {
        MediaType::Image
    }
}

fn media_mime_type(path: &str) -> &'static str {
    match get_patch_extension(path).as_deref().unwrap_or("") {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "mp3" | "mpga" | "mpeg" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" | "oga" => "audio/ogg",
        "flac" => "audio/flac",
        "m4a" => "audio/mp4",
        "webm" => "audio/webm",
        "mp4" => "audio/mp4",
        _ => "application/octet-stream",
    }
}

fn max_media_size() -> usize {
    env::var(get_env_name("max_media_size_mb"))
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_MEDIA_SIZE_MB)
        * 1024
        * 1024
}

fn check_media_size(path: &str) -> Result<()> {
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("Unable to access '{path}'"))?;
    let size = metadata.len() as usize;
    if size > max_media_size() {
        bail!(
            "File '{}' is too large ({:.1}MB), max size is {}MB",
            path,
            size as f64 / 1024.0 / 1024.0,
            max_media_size() / 1024 / 1024
        )
    }
    Ok(())
}

fn read_media_with_type(path: &str) -> Result<(String, MediaType)> {
    check_media_size(path)?;
    let media_type = detect_media_type(path);
    let mime_type = media_mime_type(path);
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let encoded = base64_encode(buffer);
    let data_url = format!("data:{mime_type};base64,{encoded}");
    Ok((data_url, media_type))
}

fn read_media_to_data_url(image_path: &str) -> Result<String> {
    let extension = get_patch_extension(image_path).unwrap_or_default();
    let mime_type = match extension.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => bail!("Unexpected media type"),
    };
    let mut file = File::open(image_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;
    let encoded_image = base64_encode(buffer);
    let data_url = format!("data:{mime_type};base64,{encoded_image}");
    Ok(data_url)
}

fn media_type_from_mime(data_url: &str) -> MediaType {
    if let Some(mime) = data_url.strip_prefix("data:").and_then(|s| s.split(';').next()) {
        if mime.starts_with("audio/") {
            return MediaType::Audio;
        }
        if mime.starts_with("video/") {
            return MediaType::Video;
        }
        if mime.starts_with("image/") {
            return MediaType::Image;
        }
    }
    MediaType::Image
}

#[cfg(test)]
mod input_audio_video_tests {
    use super::*;

    #[test]
    fn test_is_audio_extensions() {
        assert!(is_audio("recording.mp3"));
        assert!(is_audio("voice.wav"));
        assert!(is_audio("song.ogg"));
        assert!(is_audio("music.flac"));
        assert!(is_audio("audio.m4a"));
        assert!(is_audio("track.webm"));
        assert!(is_audio("clip.mp4"));
        assert!(!is_audio("photo.png"));
        assert!(!is_audio("document.txt"));
    }

    #[test]
    fn test_is_video_extensions() {
        assert!(is_video("clip.mp4"));
        assert!(is_video("screen.webm"));
        assert!(is_video("movie.avi"));
        assert!(is_video("video.mov"));
        assert!(is_video("recording.mkv"));
        assert!(!is_video("song.mp3"));
        assert!(!is_video("photo.png"));
    }

    #[test]
    fn test_detect_media_type() {
        assert_eq!(detect_media_type("photo.png"), MediaType::Image);
        assert_eq!(detect_media_type("recording.mp3"), MediaType::Audio);
        assert_eq!(detect_media_type("movie.avi"), MediaType::Video);
        // .mp4 and .webm default to audio
        assert_eq!(detect_media_type("clip.mp4"), MediaType::Audio);
        assert_eq!(detect_media_type("screen.webm"), MediaType::Audio);
    }

    #[test]
    fn test_media_type_from_mime() {
        assert_eq!(
            media_type_from_mime("data:audio/mpeg;base64,abc"),
            MediaType::Audio
        );
        assert_eq!(
            media_type_from_mime("data:video/mp4;base64,xyz"),
            MediaType::Video
        );
        assert_eq!(
            media_type_from_mime("data:image/png;base64,def"),
            MediaType::Image
        );
    }

    #[test]
    fn test_input_message_content_with_mixed_media() {
        let input = Input {
            text: "Hello world".to_string(),
            media_parts: vec![
                MessageContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: "data:image/png;base64,abc".to_string(),
                    },
                },
                MessageContentPart::AudioUrl {
                    audio_url: MediaUrl {
                        url: "data:audio/mpeg;base64,xyz".to_string(),
                        mime_type: None,
                    },
                },
            ],
            config: GlobalConfig::default(),
            raw: ("Hello world".to_string(), vec![]),
            patched_text: None,
            last_reply: None,
            continue_output: None,
            regenerate: None,
            data_urls: HashMap::new(),
            tool_calls: None,
            role: Role::default(),
            rag_name: None,
            with_session: false,
            with_agent: false,
        };
        let content = input.message_content();
        match content {
            MessageContent::Array(parts) => {
                assert_eq!(parts.len(), 3);
                match &parts[0] {
                    MessageContentPart::Text { text } => assert_eq!(text, "Hello world"),
                    _ => panic!("Expected Text part"),
                }
                matches!(parts[1], MessageContentPart::ImageUrl { .. });
                matches!(parts[2], MessageContentPart::AudioUrl { .. });
            }
            _ => panic!("Expected Array content"),
        }
    }

    #[test]
    fn test_guard_rejects_audio_without_support() {
        let input = Input {
            text: "transcribe this".to_string(),
            media_parts: vec![MessageContentPart::AudioUrl {
                audio_url: MediaUrl {
                    url: "data:audio/mpeg;base64,xyz".to_string(),
                    mime_type: None,
                },
            }],
            config: GlobalConfig::default(),
            raw: ("transcribe this".to_string(), vec![]),
            patched_text: None,
            last_reply: None,
            continue_output: None,
            regenerate: None,
            data_urls: HashMap::new(),
            tool_calls: None,
            role: Role::default(),
            rag_name: None,
            with_session: false,
            with_agent: false,
        };
        let model = Model::new("test", "gpt-4");
        let result = input.prepare_completion_data(&model, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not support audio"));
    }

    #[test]
    fn test_guard_rejects_video_without_support() {
        let input = Input {
            text: "describe this".to_string(),
            media_parts: vec![MessageContentPart::VideoUrl {
                video_url: MediaUrl {
                    url: "data:video/mp4;base64,xyz".to_string(),
                    mime_type: None,
                },
            }],
            config: GlobalConfig::default(),
            raw: ("describe this".to_string(), vec![]),
            patched_text: None,
            last_reply: None,
            continue_output: None,
            regenerate: None,
            data_urls: HashMap::new(),
            tool_calls: None,
            role: Role::default(),
            rag_name: None,
            with_session: false,
            with_agent: false,
        };
        let model = Model::new("test", "gpt-4");
        let result = input.prepare_completion_data(&model, false);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not support video"));
    }

    #[test]
    fn test_guard_allows_audio_with_support() {
        let input = Input {
            text: "transcribe this".to_string(),
            media_parts: vec![MessageContentPart::AudioUrl {
                audio_url: MediaUrl {
                    url: "data:audio/mpeg;base64,xyz".to_string(),
                    mime_type: None,
                },
            }],
            config: GlobalConfig::default(),
            raw: ("transcribe this".to_string(), vec![]),
            patched_text: None,
            last_reply: None,
            continue_output: None,
            regenerate: None,
            data_urls: HashMap::new(),
            tool_calls: None,
            role: Role::default(),
            rag_name: None,
            with_session: false,
            with_agent: false,
        };
        let mut model = Model::new("test", "gpt-4o");
        model.data_mut().supports_audio = true;
        let result = input.prepare_completion_data(&model, false);
        assert!(result.is_ok(), "should allow audio when model supports it");
    }

    #[test]
    fn test_guard_allows_video_with_support() {
        let input = Input {
            text: "describe this".to_string(),
            media_parts: vec![MessageContentPart::VideoUrl {
                video_url: MediaUrl {
                    url: "data:video/mp4;base64,xyz".to_string(),
                    mime_type: None,
                },
            }],
            config: GlobalConfig::default(),
            raw: ("describe this".to_string(), vec![]),
            patched_text: None,
            last_reply: None,
            continue_output: None,
            regenerate: None,
            data_urls: HashMap::new(),
            tool_calls: None,
            role: Role::default(),
            rag_name: None,
            with_session: false,
            with_agent: false,
        };
        let mut model = Model::new("test", "gpt-4o");
        model.data_mut().supports_video = true;
        let result = input.prepare_completion_data(&model, false);
        assert!(result.is_ok(), "should allow video when model supports it");
    }

}
