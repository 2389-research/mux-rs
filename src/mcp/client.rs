// ABOUTME: MCP client for connecting to MCP servers via stdio, SSE, or HTTP.
// ABOUTME: Supports full MCP protocol: tools, resources, prompts, roots, logging.

use std::collections::HashMap;
use std::sync::Arc;

use super::transport::{HttpTransport, SseTransport, StdioTransport, Transport};
use super::{
    McpInitializeResult, McpLogLevel, McpNotification, McpPromptGetResult, McpPromptsListResult,
    McpRequest, McpResourceContent, McpResourceReadResult, McpResourceTemplatesListResult,
    McpResourcesListResult, McpRoot, McpRootsListResult, McpSamplingParams, McpSamplingResult,
    McpServerCapabilities, McpServerConfig, McpToolInfo, McpToolResult, McpTransport,
};
use crate::error::McpError;

/// Client for communicating with an MCP server.
pub struct McpClient {
    config: McpServerConfig,
    transport: Arc<dyn Transport>,
    capabilities: McpServerCapabilities,
}

impl McpClient {
    /// Connect to an MCP server.
    pub async fn connect(config: McpServerConfig) -> Result<Self, McpError> {
        let transport: Arc<dyn Transport> = match &config.transport {
            McpTransport::Stdio { command, args, env } => {
                Arc::new(StdioTransport::connect(command, args, env).await?)
            }
            McpTransport::Sse { url } => Arc::new(SseTransport::connect(url).await?),
            McpTransport::Http { url } => Arc::new(HttpTransport::connect(url).await?),
        };

        Ok(Self {
            config,
            transport,
            capabilities: McpServerCapabilities::default(),
        })
    }

    /// Get the server name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get the server capabilities (available after initialize).
    pub fn capabilities(&self) -> &McpServerCapabilities {
        &self.capabilities
    }

    /// Send a request and wait for response.
    async fn request(
        &self,
        method: &str,
        params: Option<serde_json::Value>,
    ) -> Result<serde_json::Value, McpError> {
        let request = McpRequest::new(method, params);
        let response = self.transport.send(request).await?;

        if let Some(error) = response.error {
            return Err(McpError::Rpc {
                code: error.code,
                message: error.message,
            });
        }

        response
            .result
            .ok_or_else(|| McpError::Protocol("No result in response".into()))
    }

    /// Send a notification (no response expected).
    async fn notify(&self, method: &str, params: Option<serde_json::Value>) -> Result<(), McpError> {
        let notification = McpNotification::new(method, params);
        self.transport.notify(notification).await
    }

    // ========================================================================
    // Lifecycle
    // ========================================================================

    /// Initialize the MCP connection.
    pub async fn initialize(&mut self) -> Result<McpInitializeResult, McpError> {
        let params = serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {
                    "listChanged": true
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": "mux-rs",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let result = self.request("initialize", Some(params)).await?;
        let init_result: McpInitializeResult = serde_json::from_value(result)?;

        // Store capabilities for later use
        self.capabilities = init_result.capabilities.clone();

        // Send initialized notification
        self.notify("notifications/initialized", None).await?;

        Ok(init_result)
    }

    /// Shutdown the server connection gracefully.
    pub async fn shutdown(&self) -> Result<(), McpError> {
        self.transport.shutdown().await
    }

    // ========================================================================
    // Tools
    // ========================================================================

    /// List available tools from the server.
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>, McpError> {
        let result = self.request("tools/list", None).await?;
        let tools: Vec<McpToolInfo> = serde_json::from_value(result["tools"].clone())?;
        Ok(tools)
    }

    /// Call a tool on the server.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: serde_json::Value,
    ) -> Result<McpToolResult, McpError> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments
        });

        let result = self.request("tools/call", Some(params)).await?;
        Ok(serde_json::from_value(result)?)
    }

    // ========================================================================
    // Resources
    // ========================================================================

    /// List available resources from the server.
    pub async fn list_resources(
        &self,
        cursor: Option<&str>,
    ) -> Result<McpResourcesListResult, McpError> {
        let params = cursor.map(|c| serde_json::json!({ "cursor": c }));
        let result = self.request("resources/list", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    /// Read a resource by URI.
    pub async fn read_resource(&self, uri: &str) -> Result<Vec<McpResourceContent>, McpError> {
        let params = serde_json::json!({ "uri": uri });
        let result = self.request("resources/read", Some(params)).await?;
        let read_result: McpResourceReadResult = serde_json::from_value(result)?;
        Ok(read_result.contents)
    }

    /// List resource templates.
    pub async fn list_resource_templates(
        &self,
        cursor: Option<&str>,
    ) -> Result<McpResourceTemplatesListResult, McpError> {
        let params = cursor.map(|c| serde_json::json!({ "cursor": c }));
        let result = self.request("resources/templates/list", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    /// Subscribe to resource updates.
    pub async fn subscribe_resource(&self, uri: &str) -> Result<(), McpError> {
        let params = serde_json::json!({ "uri": uri });
        self.request("resources/subscribe", Some(params)).await?;
        Ok(())
    }

    /// Unsubscribe from resource updates.
    pub async fn unsubscribe_resource(&self, uri: &str) -> Result<(), McpError> {
        let params = serde_json::json!({ "uri": uri });
        self.request("resources/unsubscribe", Some(params)).await?;
        Ok(())
    }

    // ========================================================================
    // Prompts
    // ========================================================================

    /// List available prompts from the server.
    pub async fn list_prompts(&self, cursor: Option<&str>) -> Result<McpPromptsListResult, McpError> {
        let params = cursor.map(|c| serde_json::json!({ "cursor": c }));
        let result = self.request("prompts/list", params).await?;
        Ok(serde_json::from_value(result)?)
    }

    /// Get a prompt by name with arguments.
    pub async fn get_prompt(
        &self,
        name: &str,
        arguments: Option<HashMap<String, String>>,
    ) -> Result<McpPromptGetResult, McpError> {
        let mut params = serde_json::json!({ "name": name });
        if let Some(args) = arguments {
            params["arguments"] = serde_json::to_value(args)?;
        }
        let result = self.request("prompts/get", Some(params)).await?;
        Ok(serde_json::from_value(result)?)
    }

    // ========================================================================
    // Roots
    // ========================================================================

    /// List roots available on the client.
    /// Note: This is typically called by the server, but provided for completeness.
    pub async fn list_roots(&self) -> Result<Vec<McpRoot>, McpError> {
        let result = self.request("roots/list", None).await?;
        let roots_result: McpRootsListResult = serde_json::from_value(result)?;
        Ok(roots_result.roots)
    }

    // ========================================================================
    // Logging
    // ========================================================================

    /// Set the logging level for the server.
    pub async fn set_log_level(&self, level: McpLogLevel) -> Result<(), McpError> {
        let params = serde_json::json!({ "level": level });
        self.request("logging/setLevel", Some(params)).await?;
        Ok(())
    }

    // ========================================================================
    // Sampling
    // ========================================================================

    /// Request the server to perform model sampling.
    /// Note: Not all servers support this capability.
    pub async fn create_message(
        &self,
        params: McpSamplingParams,
    ) -> Result<McpSamplingResult, McpError> {
        let params_json = serde_json::to_value(params)?;
        let result = self.request("sampling/createMessage", Some(params_json)).await?;
        Ok(serde_json::from_value(result)?)
    }

    // ========================================================================
    // Completion (for argument completion)
    // ========================================================================

    /// Get completions for a prompt argument or resource template.
    pub async fn complete(
        &self,
        ref_type: &str,
        ref_name: &str,
        argument_name: &str,
        argument_value: &str,
    ) -> Result<Vec<String>, McpError> {
        let params = serde_json::json!({
            "ref": {
                "type": ref_type,
                "name": ref_name
            },
            "argument": {
                "name": argument_name,
                "value": argument_value
            }
        });
        let result = self.request("completion/complete", Some(params)).await?;
        let values: Vec<String> = serde_json::from_value(result["completion"]["values"].clone())?;
        Ok(values)
    }

    // ========================================================================
    // Ping
    // ========================================================================

    /// Ping the server to check if it's alive.
    pub async fn ping(&self) -> Result<(), McpError> {
        self.request("ping", None).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connect_nonexistent_stdio() {
        let config = McpServerConfig {
            name: "test".into(),
            transport: McpTransport::Stdio {
                command: "/nonexistent/binary".into(),
                args: vec![],
                env: HashMap::new(),
            },
        };

        let result = McpClient::connect(config).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_connect_invalid_sse() {
        let config = McpServerConfig {
            name: "test".into(),
            transport: McpTransport::Sse {
                url: "http://localhost:99999/nonexistent".into(),
            },
        };

        let result = McpClient::connect(config).await;
        assert!(result.is_err());
    }
}
