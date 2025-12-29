// ABOUTME: Built-in tools for common agent operations.
// ABOUTME: Includes file I/O, search, command execution, and web access.

mod bash;
mod list_files;
mod read_file;
mod search;
mod web_fetch;
mod web_search;
mod write_file;

pub use bash::BashTool;
pub use list_files::ListFilesTool;
pub use read_file::ReadFileTool;
pub use search::SearchTool;
pub use web_fetch::WebFetchTool;
pub use web_search::{SearchResult, WebSearchTool};
pub use write_file::WriteFileTool;
