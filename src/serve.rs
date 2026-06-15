use crate::{client::*, config::*, function::*, rag::*, utils::*};

use anyhow::{anyhow, bail, Result};
use bytes::Bytes;
use chrono::{Timelike, Utc};
use futures_util::StreamExt;
use http::{Method, Response, StatusCode};
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::{
    body::{Frame, Incoming},
    service::service_fn,
};
use hyper_util::rt::{TokioExecutor, TokioIo};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    convert::Infallible,
    net::IpAddr,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, LazyLock,
    },
};
use tokio::{
    net::TcpListener,
    sync::{
        mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender},
        oneshot,
    },
};
use tokio_graceful::Shutdown;
use tokio_stream::wrappers::UnboundedReceiverStream;

const DEFAULT_MODEL_NAME: &str = "default";
const PLAYGROUND_HTML: &[u8] = include_bytes!("../assets/playground.html");
const ARENA_HTML: &[u8] = include_bytes!("../assets/arena.html");
const MESSAGES_HTML: &[u8] = include_bytes!("../assets/messages.html");

type AppResponse = Response<BoxBody<Bytes, Infallible>>;

pub async fn run(config: GlobalConfig, addr: Option<String>) -> Result<()> {
    let addr = match addr {
        Some(addr) => {
            if let Ok(port) = addr.parse::<u16>() {
                format!("127.0.0.1:{port}")
            } else if let Ok(ip) = addr.parse::<IpAddr>() {
                format!("{ip}:8000")
            } else {
                addr
            }
        }
        None => config.read().serve_addr(),
    };
    let server = Arc::new(Server::new(&config));
    let listener = TcpListener::bind(&addr).await?;
    let stop_server = server.run(listener).await?;
    println!("Chat Completions API: http://{addr}/v1/chat/completions");
    println!("Embeddings API:       http://{addr}/v1/embeddings");
    println!("Rerank API:           http://{addr}/v1/rerank");
    println!("LLM Playground:       http://{addr}/playground");
    println!("LLM Arena:            http://{addr}/arena?num=2");
    println!("Messages Viewer:      http://{addr}/messages");
    shutdown_signal().await;
    let _ = stop_server.send(());
    Ok(())
}

struct Server {
    config: Config,
    models: Vec<Value>,
    roles: Vec<Role>,
    rags: Vec<String>,
}

impl Server {
    fn new(config: &GlobalConfig) -> Self {
        let mut config = config.read().clone();
        config.functions = Functions::default();
        let mut models = list_all_models(&config);
        let mut default_model = config.model.clone();
        default_model.data_mut().name = DEFAULT_MODEL_NAME.into();
        models.insert(0, &default_model);
        let models: Vec<Value> = models
            .into_iter()
            .enumerate()
            .map(|(i, model)| {
                let id = if i == 0 {
                    DEFAULT_MODEL_NAME.into()
                } else {
                    model.id()
                };
                let mut value = json!(model.data());
                if let Some(value_obj) = value.as_object_mut() {
                    value_obj.insert("id".into(), id.into());
                    value_obj.insert("object".into(), "model".into());
                    value_obj.insert("owned_by".into(), model.client_name().into());
                    value_obj.remove("name");
                }
                value
            })
            .collect();
        Self {
            config,
            models,
            roles: Config::all_roles(),
            rags: Config::list_rags(),
        }
    }

    async fn run(self: Arc<Self>, listener: TcpListener) -> Result<oneshot::Sender<()>> {
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let shutdown = Shutdown::new(async { rx.await.unwrap_or_default() });
            let guard = shutdown.guard_weak();

            loop {
                tokio::select! {
                    res = listener.accept() => {
                        let Ok((cnx, _)) = res else {
                            continue;
                        };

                        let stream = TokioIo::new(cnx);
                        let server = self.clone();
                        shutdown.spawn_task(async move {
                            let hyper_service = service_fn(move |request: hyper::Request<Incoming>| {
                                server.clone().handle(request)
                            });
                            let _ = hyper_util::server::conn::auto::Builder::new(TokioExecutor::new())
                                .serve_connection_with_upgrades(stream, hyper_service)
                                .await;
                        });
                    }
                    _ = guard.cancelled() => {
                        break;
                    }
                }
            }
        });
        Ok(tx)
    }

    async fn handle(
        self: Arc<Self>,
        req: hyper::Request<Incoming>,
    ) -> std::result::Result<AppResponse, hyper::Error> {
        let method = req.method().clone();
        let uri = req.uri().clone();
        let path = uri.path();

        if method == Method::OPTIONS {
            let mut res = Response::default();
            *res.status_mut() = StatusCode::NO_CONTENT;
            set_cors_header(&mut res);
            return Ok(res);
        }

        let mut status = StatusCode::OK;
        let res = if path == "/v1/chat/completions" {
            self.chat_completions(req).await
        } else if path == "/v1/embeddings" {
            self.embeddings(req).await
        } else if path == "/v1/rerank" {
            self.rerank(req).await
        } else if path == "/v1/models" {
            self.list_models()
        } else if path == "/v1/roles" {
            self.list_roles()
        } else if path == "/v1/rags" {
            self.list_rags()
        } else if path == "/v1/rags/search" {
            self.search_rag(req).await
        } else if path == "/playground" || path == "/playground.html" {
            self.playground_page()
        } else if path == "/arena" || path == "/arena.html" {
            self.arena_page()
        } else if path == "/messages" || path == "/messages.html" {
            self.messages_page()
        } else if path == "/api/messages/default" {
            self.load_default_messages()
        } else if path == "/api/sessions/list" {
            self.list_sessions()
        } else if path == "/api/sessions/load" {
            self.load_session(req).await
        } else if path == "/api/messages/parse" {
            self.parse_messages_file(req).await
        } else if path == "/api/session/parse" {
            self.parse_session_file(req).await
        } else {
            status = StatusCode::NOT_FOUND;
            Err(anyhow!("Not Found"))
        };
        let mut res = match res {
            Ok(res) => {
                info!("{method} {uri} {}", status.as_u16());
                res
            }
            Err(err) => {
                if status == StatusCode::OK {
                    status = StatusCode::BAD_REQUEST;
                }
                error!("{method} {uri} {} {err}", status.as_u16());
                ret_err(err)
            }
        };
        *res.status_mut() = status;
        set_cors_header(&mut res);
        Ok(res)
    }

    fn playground_page(&self) -> Result<AppResponse> {
        let res = Response::builder()
            .header("Content-Type", "text/html; charset=utf-8")
            .body(Full::new(Bytes::from(PLAYGROUND_HTML)).boxed())?;
        Ok(res)
    }

    fn arena_page(&self) -> Result<AppResponse> {
        let res = Response::builder()
            .header("Content-Type", "text/html; charset=utf-8")
            .body(Full::new(Bytes::from(ARENA_HTML)).boxed())?;
        Ok(res)
    }

    // --- Messages UI Handlers  ---

    /// Serve the Messages UI HTML page
    fn messages_page(&self) -> Result<AppResponse> {
        let res = Response::builder()
            .header("Content-Type", "text/html; charset=utf-8")
            .body(Full::new(Bytes::from(MESSAGES_HTML)).boxed())?;
        Ok(res)
    }

    /// Auto-load messages.md from config, parse and return JSON
    fn load_default_messages(&self) -> Result<AppResponse> {
        let path = self.config.messages_file();
        if !path.exists() {
            return Ok(ret_json(&json!({ "threads": Vec::<UiThread>::new() })));
        }
        let content = std::fs::read_to_string(&path)
            .map_err(|err| anyhow!("Failed to read messages file: {err}"))?;
        let threads = parse_messages_md(&content)?;
        Ok(ret_json(&json!({ "threads": threads })))
    }

    /// List all available session names, merge + deduplicate + sort
    fn list_sessions(&self) -> Result<AppResponse> {
        let sessions = self.config.list_sessions();
        let autoname = self.config.list_autoname_sessions();
        let mut all: Vec<String> = sessions.into_iter().chain(autoname).collect();
        all.sort_unstable();
        all.dedup();
        Ok(ret_json(&json!({ "sessions": all })))
    }

    /// Load a named session by ?name=... query param
    async fn load_session(&self, req: hyper::Request<Incoming>) -> Result<AppResponse> {
        let name = extract_query_param(req.uri(), "name")
            .ok_or_else(|| anyhow!("Missing 'name' query parameter"))?;

        let path = self.config.session_file(&name);
        if !path.exists() {
            bail!("Session '{}' not found at {}", name, path.display());
        }

        let content = std::fs::read_to_string(&path)
            .map_err(|err| anyhow!("Failed to read session file: {err}"))?;

        let session: Session =
            serde_yaml::from_str(&content).map_err(|err| anyhow!("Invalid session YAML: {err}"))?;

        let threads = session_to_ui_threads(&session);
        Ok(ret_json(&json!({ "threads": threads })))
    }

    /// POST endpoint to parse an uploaded messages.md file content
    async fn parse_messages_file(&self, req: hyper::Request<Incoming>) -> Result<AppResponse> {
        let req_body = req.collect().await?.to_bytes();
        let body: ParseRequestBody = serde_json::from_slice(&req_body)
            .map_err(|err| anyhow!("Invalid request body: {err}"))?;
        let threads = parse_messages_md(&body.content)?;
        Ok(ret_json(&json!({ "threads": threads })))
    }

    /// POST endpoint to parse an uploaded session YAML file content
    async fn parse_session_file(&self, req: hyper::Request<Incoming>) -> Result<AppResponse> {
        let req_body = req.collect().await?.to_bytes();
        let body: ParseRequestBody = serde_json::from_slice(&req_body)
            .map_err(|err| anyhow!("Invalid request body: {err}"))?;
        let session: Session = serde_yaml::from_str(&body.content)
            .map_err(|err| anyhow!("Invalid session YAML: {err}"))?;
        let threads = session_to_ui_threads(&session);
        Ok(ret_json(&json!({ "threads": threads })))
    }

    fn list_models(&self) -> Result<AppResponse> {
        let data = json!({ "data": self.models });
        let res = Response::builder()
            .header("Content-Type", "application/json; charset=utf-8")
            .body(Full::new(Bytes::from(data.to_string())).boxed())?;
        Ok(res)
    }

    fn list_roles(&self) -> Result<AppResponse> {
        let data = json!({ "data": self.roles });
        let res = Response::builder()
            .header("Content-Type", "application/json; charset=utf-8")
            .body(Full::new(Bytes::from(data.to_string())).boxed())?;
        Ok(res)
    }

    fn list_rags(&self) -> Result<AppResponse> {
        let data = json!({ "data": self.rags });
        let res = Response::builder()
            .header("Content-Type", "application/json; charset=utf-8")
            .body(Full::new(Bytes::from(data.to_string())).boxed())?;
        Ok(res)
    }

    async fn search_rag(&self, req: hyper::Request<Incoming>) -> Result<AppResponse> {
        let req_body = req.collect().await?.to_bytes();
        let req_body: Value = serde_json::from_slice(&req_body)
            .map_err(|err| anyhow!("Invalid request json, {err}"))?;

        debug!("search rag request: {req_body}");
        let SearchRagReqBody { name, input } = serde_json::from_value(req_body)
            .map_err(|err| anyhow!("Invalid request body, {err}"))?;

        let config = Arc::new(RwLock::new(self.config.clone()));

        let abort_signal = create_abort_signal();

        let rag_path = config.read().rag_file(&name);
        let rag = Rag::load(&config, &name, &rag_path)?;

        let rag_result = Config::search_rag(&config, &rag, &input, abort_signal).await?;

        let data = json!({ "data": rag_result });
        let res = Response::builder()
            .header("Content-Type", "application/json; charset=utf-8")
            .body(Full::new(Bytes::from(data.to_string())).boxed())?;
        Ok(res)
    }

    async fn chat_completions(&self, req: hyper::Request<Incoming>) -> Result<AppResponse> {
        let req_body = req.collect().await?.to_bytes();
        let req_body: Value = serde_json::from_slice(&req_body)
            .map_err(|err| anyhow!("Invalid request json, {err}"))?;

        debug!("chat completions request: {req_body}");
        let req_body = serde_json::from_value(req_body)
            .map_err(|err| anyhow!("Invalid request body, {err}"))?;

        let ChatCompletionsReqBody {
            model,
            messages,
            temperature,
            top_p,
            max_tokens,
            stream,
            tools,
        } = req_body;

        let mut messages =
            parse_messages(messages).map_err(|err| anyhow!("Invalid request body, {err}"))?;

        let functions = parse_tools(tools).map_err(|err| anyhow!("Invalid request body, {err}"))?;

        let config = self.config.clone();

        let default_model = config.model.clone();

        let config = Arc::new(RwLock::new(config));

        let (model_name, change) = if model == DEFAULT_MODEL_NAME {
            (default_model.id(), true)
        } else if default_model.id() == model {
            (model, false)
        } else {
            (model, true)
        };

        if change {
            config.write().set_model(&model_name)?;
        }

        let mut client = init_client(&config, None)?;
        if max_tokens.is_some() {
            client.model_mut().set_max_tokens(max_tokens, true);
        }
        let abort_signal = create_abort_signal();
        let http_client = client.build_client()?;

        let completion_id = generate_completion_id();
        let created = Utc::now().timestamp();

        patch_messages(&mut messages, client.model());

        let data: ChatCompletionsData = ChatCompletionsData {
            messages,
            temperature,
            top_p,
            functions,
            stream,
        };

        if stream {
            let (tx, mut rx) = unbounded_channel();
            tokio::spawn(async move {
                let is_first = Arc::new(AtomicBool::new(true));
                let (sse_tx, sse_rx) = unbounded_channel();
                let mut handler = SseHandler::new(sse_tx, abort_signal);
                async fn map_event(
                    mut sse_rx: UnboundedReceiver<SseEvent>,
                    tx: &UnboundedSender<ResEvent>,
                    is_first: Arc<AtomicBool>,
                ) {
                    while let Some(reply_event) = sse_rx.recv().await {
                        if is_first.load(Ordering::SeqCst) {
                            let _ = tx.send(ResEvent::First(None));
                            is_first.store(false, Ordering::SeqCst)
                        }
                        match reply_event {
                            SseEvent::Text(text) => {
                                let _ = tx.send(ResEvent::Text(text));
                            }
                            SseEvent::Done => {
                                let _ = tx.send(ResEvent::Done);
                                sse_rx.close();
                            }
                        }
                    }
                }
                async fn chat_completions(
                    client: &dyn Client,
                    http_client: &reqwest::Client,
                    handler: &mut SseHandler,
                    mut data: ChatCompletionsData,
                    tx: &UnboundedSender<ResEvent>,
                    is_first: Arc<AtomicBool>,
                ) {
                    if client.model().no_stream() {
                        data.stream = false;
                        let ret = client.chat_completions_inner(http_client, data).await;
                        match ret {
                            Ok(output) => {
                                let ChatCompletionsOutput {
                                    text, tool_calls, ..
                                } = output;
                                let _ = tx.send(ResEvent::First(None));
                                is_first.store(false, Ordering::SeqCst);
                                let _ = tx.send(ResEvent::Text(text));
                                if !tool_calls.is_empty() {
                                    let _ = tx.send(ResEvent::ToolCalls(tool_calls));
                                }
                            }
                            Err(err) => {
                                let _ = tx.send(ResEvent::First(Some(format!("{err:?}"))));
                                is_first.store(false, Ordering::SeqCst)
                            }
                        };
                    } else {
                        let ret = client
                            .chat_completions_streaming_inner(http_client, handler, data)
                            .await;
                        let first = match ret {
                            Ok(()) => None,
                            Err(err) => Some(format!("{err:?}")),
                        };
                        if is_first.load(Ordering::SeqCst) {
                            let _ = tx.send(ResEvent::First(first));
                            is_first.store(false, Ordering::SeqCst)
                        }
                        let tool_calls = handler.tool_calls().to_vec();
                        if !tool_calls.is_empty() {
                            let _ = tx.send(ResEvent::ToolCalls(tool_calls));
                        }
                    }
                    handler.done();
                }
                tokio::join!(
                    map_event(sse_rx, &tx, is_first.clone()),
                    chat_completions(
                        client.as_ref(),
                        &http_client,
                        &mut handler,
                        data,
                        &tx,
                        is_first
                    ),
                );
            });

            let first_event = rx.recv().await;

            if let Some(ResEvent::First(Some(err))) = first_event {
                bail!("{err}");
            }

            let shared: Arc<(String, String, i64, AtomicBool)> =
                Arc::new((completion_id, model_name, created, AtomicBool::new(false)));
            let stream = UnboundedReceiverStream::new(rx);
            let stream = stream.filter_map(move |res_event| {
                let shared = shared.clone();
                async move {
                    let (completion_id, model, created, has_tool_calls) = shared.as_ref();
                    match res_event {
                        ResEvent::Text(text) => {
                            Some(Ok(create_text_frame(completion_id, model, *created, &text)))
                        }
                        ResEvent::ToolCalls(tool_calls) => {
                            has_tool_calls.store(true, Ordering::SeqCst);
                            Some(Ok(create_tool_calls_frame(
                                completion_id,
                                model,
                                *created,
                                &tool_calls,
                            )))
                        }
                        ResEvent::Done => Some(Ok(create_done_frame(
                            completion_id,
                            model,
                            *created,
                            has_tool_calls.load(Ordering::SeqCst),
                        ))),
                        _ => None,
                    }
                }
            });
            let res = Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "text/event-stream")
                .header("Cache-Control", "no-cache")
                .header("Connection", "keep-alive")
                .body(BodyExt::boxed(StreamBody::new(stream)))?;
            Ok(res)
        } else {
            let output = client.chat_completions_inner(&http_client, data).await?;
            let res = Response::builder()
                .header("Content-Type", "application/json")
                .body(
                    Full::new(ret_non_stream(
                        &completion_id,
                        &model_name,
                        created,
                        &output,
                    ))
                    .boxed(),
                )?;
            Ok(res)
        }
    }

    async fn embeddings(&self, req: hyper::Request<Incoming>) -> Result<AppResponse> {
        let req_body = req.collect().await?.to_bytes();
        let req_body: Value = serde_json::from_slice(&req_body)
            .map_err(|err| anyhow!("Invalid request json, {err}"))?;

        debug!("embeddings request: {req_body}");
        let req_body = serde_json::from_value(req_body)
            .map_err(|err| anyhow!("Invalid request body, {err}"))?;

        let EmbeddingsReqBody {
            input,
            model: embedding_model_id,
        } = req_body;

        let config = Arc::new(RwLock::new(self.config.clone()));

        let embedding_model =
            Model::retrieve_model(&config.read(), &embedding_model_id, ModelType::Embedding)?;

        let texts = match input {
            EmbeddingsReqBodyInput::Single(v) => vec![v],
            EmbeddingsReqBodyInput::Multiple(v) => v,
        };
        let client = init_client(&config, Some(embedding_model))?;
        let data = client
            .embeddings(&EmbeddingsData {
                query: false,
                texts,
            })
            .await?;
        let data: Vec<_> = data
            .into_iter()
            .enumerate()
            .map(|(i, v)| {
                json!({
                        "object": "embedding",
                        "embedding": v,
                        "index": i,
                })
            })
            .collect();
        let output = json!({
            "object": "list",
            "data": data,
            "model": embedding_model_id,
            "usage": {
                "prompt_tokens": 0,
                "total_tokens": 0,
            }
        });
        let res = Response::builder()
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(output.to_string())).boxed())?;
        Ok(res)
    }

    async fn rerank(&self, req: hyper::Request<Incoming>) -> Result<AppResponse> {
        let req_body = req.collect().await?.to_bytes();
        let req_body: Value = serde_json::from_slice(&req_body)
            .map_err(|err| anyhow!("Invalid request json, {err}"))?;

        debug!("rerank request: {req_body}");
        let req_body = serde_json::from_value(req_body)
            .map_err(|err| anyhow!("Invalid request body, {err}"))?;

        let RerankReqBody {
            model: reranker_model_id,
            documents,
            query,
            top_n,
        } = req_body;

        let top_n = top_n.unwrap_or(documents.len());

        let config = Arc::new(RwLock::new(self.config.clone()));

        let reranker_model =
            Model::retrieve_model(&config.read(), &reranker_model_id, ModelType::Reranker)?;

        let client = init_client(&config, Some(reranker_model))?;
        let data = client
            .rerank(&RerankData {
                query,
                documents: documents.clone(),
                top_n,
            })
            .await?;

        let results: Vec<_> = data
            .into_iter()
            .map(|v| {
                json!({
                    "index": v.index,
                    "relevance_score": v.relevance_score,
                    "document": documents.get(v.index).map(|v| json!(v)).unwrap_or_default(),
                })
            })
            .collect();
        let output = json!({
            "id": uuid::Uuid::new_v4().to_string(),
            "results": results,
        });
        let res = Response::builder()
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(output.to_string())).boxed())?;
        Ok(res)
    }
}

// Messages UI response structs

#[derive(Debug, Serialize)]
struct UiThread {
    id: usize,
    title: String,
    timestamp: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rag: Option<String>,
    user_message: String,
    tool_calls: Vec<UiToolCall>,
    assistant_message: String,
    is_session: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    meta: Option<UiSessionMeta>,
}

#[derive(Debug, Serialize)]
struct UiToolCall {
    name: String,
    arguments: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
}

#[derive(Debug, Serialize)]
struct UiSessionMeta {
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    use_tools: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    role_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    agent_instructions: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ParseRequestBody {
    content: String,
}

/// Helper for consistent JSON response building
fn ret_json(data: &Value) -> AppResponse {
    Response::builder()
        .header("Content-Type", "application/json; charset=utf-8")
        .body(Full::new(Bytes::from(data.to_string())).boxed())
        .unwrap()
}

// --- Parsing Logic  ---

/// Convert a Session (YAML) to Vec<UiThread> with proper user/assistant pairing, tool call extraction, and session metadata
fn session_to_ui_threads(session: &Session) -> Vec<UiThread> {
    let all_messages: Vec<&Message> = session
        .compressed_messages()
        .iter()
        .chain(session.messages().iter())
        .collect();

    let mut threads = Vec::new();
    let mut i = 0;

    while i < all_messages.len() {
        let msg = &all_messages[i];

        // Skip system and tool messages at top level
        if msg.role.is_system() || msg.role == MessageRole::Tool {
            i += 1;
            continue;
        }

        if msg.role.is_user() {
            let user_text = msg.content.to_text().trim().to_string();
            let mut tool_calls = Vec::new();
            let mut assistant_parts = Vec::new();

            i += 1;
            // Collect assistant response(s) and tool calls following this user message
            while i < all_messages.len() {
                let next = &all_messages[i];
                if next.role.is_assistant() {
                    let text = next.content.to_text().trim().to_string();
                    if !text.is_empty() {
                        assistant_parts.push(text);
                    }
                    // Check for tool calls embedded in assistant message
                    if let MessageContent::ToolCalls(tc) = &next.content {
                        for tr in &tc.tool_results {
                            tool_calls.push(UiToolCall {
                                name: tr.call.name.clone(),
                                arguments: serde_json::to_value(&tr.call.arguments)
                                    .unwrap_or_default(),
                                id: tr.call.id.clone(),
                                result: Some(serde_json::to_value(&tr.output).unwrap_or_default()),
                            });
                        }
                    }
                    i += 1;
                } else if next.role == MessageRole::Tool {
                    // Standalone tool role messages — extract tool call info
                    if let MessageContent::ToolCalls(tc) = &next.content {
                        for tr in &tc.tool_results {
                            tool_calls.push(UiToolCall {
                                name: tr.call.name.clone(),
                                arguments: serde_json::to_value(&tr.call.arguments)
                                    .unwrap_or_default(),
                                id: tr.call.id.clone(),
                                result: Some(serde_json::to_value(&tr.output).unwrap_or_default()),
                            });
                        }
                    }
                    i += 1;
                } else {
                    break; // Next user message or system — end of this thread
                }
            }

            threads.push(UiThread {
                id: threads.len(),
                title: user_text.chars().take(80).collect(),
                timestamp: String::new(),
                role: session.role_name().map(|s| s.to_string()),
                rag: None,
                user_message: user_text,
                tool_calls,
                assistant_message: assistant_parts.join("\n\n"),
                is_session: true,
                meta: Some(UiSessionMeta {
                    model: session.model_id().to_string(),
                    temperature: session.temperature(),
                    top_p: session.top_p(),
                    use_tools: session.use_tools(),
                    role_name: session.role_name().map(|s| s.to_string()),
                    agent_instructions: if session.agent_instructions().is_empty() {
                        None
                    } else {
                        Some(session.agent_instructions().to_string())
                    },
                }),
            });
        } else {
            i += 1;
        }
    }

    threads
}

/// Parse messages.md format using regex header matching, extracts title/timestamp/scope, user message, tool calls block, and assistant message
/// Supports both new format `# CHAT: title [timestamp] (scope)` and old format `# CHAT:[timestamp] (scope)`.
fn parse_messages_md(content: &str) -> Result<Vec<UiThread>> {
    // New format: # CHAT: title [timestamp] (scope)
    static HEADER_RE_NEW: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
        fancy_regex::Regex::new(r"^# CHAT: (.+?) \[(.+?)\](?: \((.+?)\))?\s*$").unwrap()
    });
    // Old format (circa 2023): # CHAT:[timestamp] (scope)
    static HEADER_RE_OLD: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
        fancy_regex::Regex::new(r"^# CHAT: ?\[(.+?)\](?: \((.+?)\))?\s*$").unwrap()
    });

    let lines: Vec<&str> = content.lines().collect();
    let mut threads = Vec::new();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        // Try new format first (has title), then old format (no title)
        let new_match = HEADER_RE_NEW.captures(line).ok().and_then(|c| c);
        let is_new = new_match.is_some();
        let m = new_match.or_else(|| HEADER_RE_OLD.captures(line).ok().and_then(|c| c));
        let Some(m) = m else {
            i += 1;
            continue;
        };

        let title = if is_new {
            m.get(1).map(|m| m.as_str().to_string()).unwrap_or_default()
        } else {
            String::new()
        };
        let timestamp = if is_new {
            // New format: groups are (1=title, 2=timestamp, 3=scope)
            m.get(2).map(|m| m.as_str()).unwrap_or("").to_string()
        } else {
            // Old format: groups are (1=timestamp, 2=scope)
            m.get(1).map(|m| m.as_str()).unwrap_or("").to_string()
        };
        let scope = if is_new {
            m.get(3).map(|m| m.as_str()).unwrap_or("")
        } else {
            m.get(2).map(|m| m.as_str()).unwrap_or("")
        };
        let (role, rag) = parse_scope(scope);

        i += 1;

        // Collect user message until first --------
        let mut user_lines = Vec::new();
        while i < lines.len() && lines[i] != "--------" {
            user_lines.push(lines[i]);
            i += 1;
        }
        if i < lines.len() {
            i += 1;
        } // skip first --------

        // Check for tool calls block, then collect assistant message
        let mut tool_calls = Vec::new();
        let mut assistant_lines = Vec::new();

        // Peek for <tool_calls> block
        let mut scan = i;
        let mut tc_start = None;
        let mut tc_end = None;
        while scan < lines.len() && lines[scan] != "--------" {
            if lines[scan].trim() == "<tool_calls>" {
                tc_start = Some(scan);
            }
            if lines[scan].trim() == "</tool_calls>" {
                tc_end = Some(scan);
                break;
            }
            scan += 1;
        }

        if let (Some(start), Some(end)) = (tc_start, tc_end) {
            if end > start {
                // Parse tool calls from JSON between tags
                let tc_content = lines[start + 1..end].join("\n").trim().to_string();
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&tc_content) {
                    if let Some(arr) = json.as_array() {
                        for item in arr {
                            if let (Some(name), Some(args)) = (
                                item.get("name").and_then(|v| v.as_str()),
                                item.get("arguments"),
                            ) {
                                tool_calls.push(UiToolCall {
                                    name: name.to_string(),
                                    arguments: args.clone(),
                                    id: None,
                                    result: None,
                                });
                            }
                        }
                    }
                }
                // Assistant message is after </tool_calls>
                let mut ai = end + 1;
                while ai < lines.len() && lines[ai] != "--------" {
                    assistant_lines.push(lines[ai]);
                    ai += 1;
                }
                i = ai;
            }
        } else {
            // No tool calls — everything until next -------- is assistant message
            while i < lines.len() && lines[i] != "--------" {
                assistant_lines.push(lines[i]);
                i += 1;
            }
        }

        if i < lines.len() {
            i += 1;
        } // skip closing --------

        let user_message = user_lines.join("\n").trim().to_string();

        // Filter out messages that consist of only the single token "hi"
        if user_message == "hi" {
            continue;
        }

        threads.push(UiThread {
            id: threads.len(),
            title,
            timestamp,
            role,
            rag,
            user_message,
            tool_calls,
            assistant_message: assistant_lines.join("\n").trim().to_string(),
            is_session: false,
            meta: None,
        });

        // Skip blank lines between threads
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
    }

    Ok(threads)
}

/// Split scope string by '#' into (role, rag) tuple, handle '%%' as null role
fn parse_scope(scope: &str) -> (Option<String>, Option<String>) {
    if scope.is_empty() {
        return (None, None);
    }
    let parts: Vec<&str> = scope.split('#').collect();
    let role = if parts[0] == "%%" {
        None
    } else {
        Some(parts[0].to_string())
    };
    let rag = if parts.len() > 1 && !parts[1].is_empty() {
        Some(parts[1].to_string())
    } else {
        None
    };
    (role, rag)
}

/// Extract query parameters from http::Uri for /api/sessions/load?name=...
fn extract_query_param(uri: &http::Uri, key: &str) -> Option<String> {
    uri.query()?
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find_map(|(k, v)| {
            if k == key {
                Some(
                    percent_encoding::percent_decode_str(v)
                        .decode_utf8()
                        .ok()?
                        .into_owned(),
                )
            } else {
                None
            }
        })
}

#[derive(Debug, Deserialize)]
struct SearchRagReqBody {
    name: String,
    input: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsReqBody {
    model: String,
    messages: Vec<Value>,
    temperature: Option<f64>,
    top_p: Option<f64>,
    max_tokens: Option<isize>,
    #[serde(default)]
    stream: bool,
    tools: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsReqBody {
    input: EmbeddingsReqBodyInput,
    model: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum EmbeddingsReqBodyInput {
    Single(String),
    Multiple(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct RerankReqBody {
    documents: Vec<String>,
    query: String,
    model: String,
    top_n: Option<usize>,
}

#[derive(Debug)]
enum ResEvent {
    First(Option<String>),
    Text(String),
    ToolCalls(Vec<ToolCall>),
    Done,
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install CTRL+C signal handler")
}

fn generate_completion_id() -> String {
    let random_id = chrono::Utc::now().nanosecond();
    format!("chatcmpl-{random_id}")
}

fn set_cors_header(res: &mut AppResponse) {
    res.headers_mut().insert(
        hyper::header::ACCESS_CONTROL_ALLOW_ORIGIN,
        hyper::header::HeaderValue::from_static("*"),
    );
    res.headers_mut().insert(
        hyper::header::ACCESS_CONTROL_ALLOW_METHODS,
        hyper::header::HeaderValue::from_static("GET,POST,PUT,PATCH,DELETE"),
    );
    res.headers_mut().insert(
        hyper::header::ACCESS_CONTROL_ALLOW_HEADERS,
        hyper::header::HeaderValue::from_static("Content-Type,Authorization"),
    );
}

fn create_text_frame(id: &str, model: &str, created: i64, content: &str) -> Frame<Bytes> {
    let delta = if content.is_empty() {
        json!({ "role": "assistant", "content": content })
    } else {
        json!({ "content": content })
    };
    let choice = json!({
        "index": 0,
        "delta": delta,
        "finish_reason": null,
    });
    let value = build_chat_completion_chunk_json(id, model, created, &choice);
    Frame::data(Bytes::from(format!("data: {value}\n\n")))
}

fn create_tool_calls_frame(
    id: &str,
    model: &str,
    created: i64,
    tool_calls: &[ToolCall],
) -> Frame<Bytes> {
    let chunks = tool_calls
        .iter()
        .enumerate()
        .flat_map(|(i, call)| {
            let choice1 = json!({
              "index": 0,
              "delta": {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                  {
                    "index": i,
                    "id": call.id,
                    "type": "function",
                    "function": {
                      "name": call.name,
                      "arguments": ""
                    }
                  }
                ]
              },
              "finish_reason": null
            });
            let choice2 = json!({
              "index": 0,
              "delta": {
                "tool_calls": [
                  {
                    "index": i,
                    "function": {
                      "arguments": call.arguments.to_string(),
                    }
                  }
                ]
              },
              "finish_reason": null
            });
            vec![
                build_chat_completion_chunk_json(id, model, created, &choice1),
                build_chat_completion_chunk_json(id, model, created, &choice2),
            ]
        })
        .map(|v| format!("data: {v}\n\n"))
        .collect::<Vec<String>>()
        .join("");
    Frame::data(Bytes::from(chunks))
}

fn create_done_frame(id: &str, model: &str, created: i64, has_tool_calls: bool) -> Frame<Bytes> {
    let finish_reason = if has_tool_calls { "tool_calls" } else { "stop" };
    let choice = json!({
        "index": 0,
        "delta": {},
        "finish_reason": finish_reason,
    });
    let value = build_chat_completion_chunk_json(id, model, created, &choice);
    Frame::data(Bytes::from(format!("data: {value}\n\ndata: [DONE]\n\n")))
}

fn build_chat_completion_chunk_json(id: &str, model: &str, created: i64, choice: &Value) -> Value {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    })
}

fn ret_non_stream(id: &str, model: &str, created: i64, output: &ChatCompletionsOutput) -> Bytes {
    let id = output.id.as_deref().unwrap_or(id);
    let input_tokens = output.input_tokens.unwrap_or_default();
    let output_tokens = output.output_tokens.unwrap_or_default();
    let total_tokens = input_tokens + output_tokens;
    let choice = if output.tool_calls.is_empty() {
        json!({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output.text,
            },
            "logprobs": null,
            "finish_reason": "stop",
        })
    } else {
        let content = if output.text.is_empty() {
            Value::Null
        } else {
            output.text.clone().into()
        };
        let tool_calls: Vec<_> = output
            .tool_calls
            .iter()
            .map(|call| {
                json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments.to_string(),
                    }
                })
            })
            .collect();
        json!({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls,
            },
            "logprobs": null,
            "finish_reason": "tool_calls",
        })
    };
    let res_body = json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [choice],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
    });
    Bytes::from(res_body.to_string())
}

fn ret_err<T: std::fmt::Display>(err: T) -> AppResponse {
    let data = json!({
        "error": {
            "message": err.to_string(),
            "type": "invalid_request_error",
        },
    });
    Response::builder()
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(data.to_string())).boxed())
        .unwrap()
}

fn parse_messages(message: Vec<Value>) -> Result<Vec<Message>> {
    let mut output = vec![];
    let mut tool_results = None;
    for (i, message) in message.into_iter().enumerate() {
        let err = || anyhow!("Failed to parse '.messages[{i}]'");
        let role = message["role"].as_str().ok_or_else(err)?;
        let content = match message.get("content") {
            Some(value) => {
                if let Some(value) = value.as_str() {
                    MessageContent::Text(value.to_string())
                } else if value.is_array() {
                    let value = serde_json::from_value(value.clone()).map_err(|_| err())?;
                    MessageContent::Array(value)
                } else if value.is_null() {
                    MessageContent::Text(String::new())
                } else {
                    return Err(err());
                }
            }
            None => MessageContent::Text(String::new()),
        };
        match role {
            "system" | "user" => {
                let role = match role {
                    "system" => MessageRole::System,
                    "user" => MessageRole::User,
                    _ => unreachable!(),
                };
                output.push(Message::new(role, content))
            }
            "assistant" => {
                let role = MessageRole::Assistant;
                match message["tool_calls"].as_array() {
                    Some(tool_calls) => {
                        if tool_results.is_some() {
                            return Err(err());
                        }
                        let mut list = vec![];
                        for tool_call in tool_calls {
                            if let (id, Some(name), Some(arguments)) = (
                                tool_call["id"].as_str().map(|v| v.to_string()),
                                tool_call["function"]["name"].as_str(),
                                tool_call["function"]["arguments"].as_str(),
                            ) {
                                let arguments =
                                    serde_json::from_str(arguments).map_err(|_| err())?;
                                list.push((id, name.to_string(), arguments));
                            } else {
                                return Err(err());
                            }
                        }
                        tool_results = Some((content.to_text(), list, vec![]));
                    }
                    None => output.push(Message::new(role, content)),
                }
            }
            "tool" => match tool_results.take() {
                Some((text, tool_calls, mut tool_values)) => {
                    let tool_call_id = message["tool_call_id"].as_str().map(|v| v.to_string());
                    let content = content.to_text();
                    let value: Value = serde_json::from_str(&content)
                        .ok()
                        .unwrap_or_else(|| content.into());

                    tool_values.push((value, tool_call_id));

                    if tool_calls.len() == tool_values.len() {
                        let mut list = vec![];
                        for ((id, name, arguments), (value, tool_call_id)) in
                            tool_calls.into_iter().zip(tool_values)
                        {
                            if id != tool_call_id {
                                return Err(err());
                            }
                            list.push(ToolResult::new(ToolCall::new(name, arguments, id), value))
                        }
                        output.push(Message::new(
                            MessageRole::Assistant,
                            MessageContent::ToolCalls(MessageContentToolCalls::new(list, text)),
                        ));
                        tool_results = None;
                    } else {
                        tool_results = Some((text, tool_calls, tool_values));
                    }
                }
                None => return Err(err()),
            },
            _ => {
                return Err(err());
            }
        }
    }

    if tool_results.is_some() {
        bail!("Invalid messages");
    }

    Ok(output)
}

fn parse_tools(tools: Option<Vec<Value>>) -> Result<Option<Vec<FunctionDeclaration>>> {
    let tools = match tools {
        Some(v) => v,
        None => return Ok(None),
    };
    let mut functions = vec![];
    for (i, tool) in tools.into_iter().enumerate() {
        if let (Some("function"), Some(function)) = (
            tool["type"].as_str(),
            tool["function"]
                .as_object()
                .and_then(|v| serde_json::from_value(json!(v)).ok()),
        ) {
            functions.push(function);
        } else {
            bail!("Failed to parse '.tools[{i}]'")
        }
    }
    Ok(Some(functions))
}
