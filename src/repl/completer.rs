use super::{ReplCommand, REPL_COMMANDS};

use crate::{config::GlobalConfig, utils::fuzzy_filter};

use dirs::home_dir;
use reedline::{Completer, Span, Suggestion};
use std::{
    collections::HashMap,
    fs::DirEntry,
    path::{Component, PathBuf},
};

impl Completer for ReplCompleter {
    fn complete(&mut self, line: &str, pos: usize) -> Vec<Suggestion> {
        let mut suggestions = vec![];
        let line = &line[0..pos];
        let mut parts = split_line(line);
        if parts.is_empty() {
            return suggestions;
        }

        // ::: is a multiline indicator
        if parts[0].0 == r#":::"# {
            parts.remove(0);
        }

        let parts_len = parts.len();
        if parts_len == 0 {
            return suggestions;
        }
        let (cmd, cmd_start) = parts[0];

        if !cmd.starts_with('.') {
            return suggestions;
        }

        let state = self.config.read().state();

        let command_filter = parts
            .iter()
            .take(2)
            .map(|(v, _)| *v)
            .collect::<Vec<&str>>()
            .join(" ");
        let commands: Vec<_> = self
            .commands
            .iter()
            .filter(|cmd| {
                cmd.is_valid(state)
                    && (command_filter.len() == 1 || cmd.name.starts_with(&command_filter[..2]))
            })
            .collect();
        let commands = fuzzy_filter(commands, |v| v.name, &command_filter);

        if parts_len > 1 {
            let span = Span::new(parts[parts_len - 1].1, pos);

            let cur_token = parts[parts_len - 1].0;
            if let Some(path) = looks_like_path(cur_token) {
                return path_suggestions(path, span);
            }

            let args_line = &line[parts[1].1..];
            let args: Vec<&str> = parts.iter().skip(1).map(|(v, _)| *v).collect();
            suggestions.extend(
                self.config
                    .read()
                    .repl_complete(cmd, &args, args_line)
                    .iter()
                    .map(|(value, description)| {
                        let description = description.as_deref().unwrap_or_default();
                        create_suggestion(value, description, span)
                    }),
            )
        }

        // show default completion menu
        if suggestions.is_empty() {
            let span = Span::new(cmd_start, pos);
            suggestions.extend(commands.iter().map(|cmd| {
                let name = cmd.name;
                let description = cmd.description;
                let has_group = self.groups.get(name).map(|v| *v > 1).unwrap_or_default();
                let name = if has_group {
                    name.to_string()
                } else {
                    format!("{name} ")
                };
                create_suggestion(&name, description, span)
            }))
        }
        suggestions
    }
}

pub struct ReplCompleter {
    config: GlobalConfig,
    commands: Vec<ReplCommand>,
    groups: HashMap<&'static str, usize>,
}

impl ReplCompleter {
    pub fn new(config: &GlobalConfig) -> Self {
        let mut groups = HashMap::new();

        let commands: Vec<ReplCommand> = REPL_COMMANDS.to_vec();

        for cmd in REPL_COMMANDS.iter() {
            let name = cmd.name;
            if let Some(count) = groups.get(name) {
                groups.insert(name, count + 1);
            } else {
                groups.insert(name, 1);
            }
        }

        Self {
            config: config.clone(),
            commands,
            groups,
        }
    }
}

fn create_suggestion(value: &str, description: &str, span: Span) -> Suggestion {
    let description = if description.is_empty() {
        None
    } else {
        Some(description.to_string())
    };
    Suggestion {
        value: value.to_string(),
        description,
        style: None,
        extra: None,
        span,
        append_whitespace: false,
    }
}

fn path_suggestions(mut path: PathBuf, span: Span) -> Vec<Suggestion> {
    let mut results = vec![];

    if path.is_file() {
        if let Some(p) = path.to_str() {
            results.push(create_suggestion(p, "", span));
        }
        return results;
    };

    let mut remainder: Option<Component> = None;
    let path_copy = path.clone();

    if path_copy.components().count() > 1 && !path.exists() {
        remainder = path_copy.components().next_back();
        path.pop();
    }

    if path.is_dir() {
        if let Ok(entries) = path.read_dir() {
            results.extend(
                entries
                    .filter_map(|entry| entry.ok())
                    .filter(|entry| remainder.is_none() || is_last_comp_match(entry, &remainder))
                    .map(|entry| {
                        create_suggestion(entry.path().to_str().expect("invalid path"), "", span)
                    }),
            );
        }
    }
    results
}

// checks if last component of dir entry starts with a test component
fn is_last_comp_match(entry: &DirEntry, remainder: &Option<Component>) -> bool {
    if let Some(remainder_comp) = remainder {
        if let Some(entry_comp) = entry.path().components().next_back() {
            let remainder_str = remainder_comp.as_os_str().to_str().unwrap_or("");
            let entry_str = entry_comp.as_os_str().to_str().unwrap_or("");
            return entry_str.starts_with(remainder_str);
        }
    }
    true
}

fn looks_like_path(tok: &str) -> Option<PathBuf> {
    if tok.starts_with("../")
        || tok.starts_with("./")
        || tok.starts_with("/")
        || tok.starts_with("~/")
    {
        let homedir = home_dir()
            .expect("reading homedir")
            .to_str()
            .unwrap()
            .to_owned();
        Some(PathBuf::from(tok.replace("~", &homedir)))
    } else {
        None
    }
}

fn split_line(line: &str) -> Vec<(&str, usize)> {
    let mut parts = vec![];
    let mut part_start = None;
    for (i, ch) in line.char_indices() {
        if ch == ' ' {
            if let Some(s) = part_start {
                parts.push((&line[s..i], s));
                part_start = None;
            }
        } else if part_start.is_none() {
            part_start = Some(i)
        }
    }
    if let Some(s) = part_start {
        parts.push((&line[s..], s));
    } else {
        parts.push(("", line.len()))
    }
    parts
}

#[test]
fn test_split_line() {
    assert_eq!(split_line(".role coder"), vec![(".role", 0), ("coder", 6)],);
    assert_eq!(
        split_line(" .role   coder"),
        vec![(".role", 1), ("coder", 9)],
    );
    assert_eq!(
        split_line(".set highlight "),
        vec![(".set", 0), ("highlight", 5), ("", 15)],
    );
    assert_eq!(
        split_line(".set highlight t"),
        vec![(".set", 0), ("highlight", 5), ("t", 15)],
    );
}
