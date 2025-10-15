from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator


class ModelConfig(BaseModel):
    # For model catalog references
    model_ref: Optional[str] = Field(None, description="Reference to model in catalog")
    
    # Legacy direct configuration (optional when using model_ref)
    provider: Optional[str] = Field(None, description="Model provider (bedrock, openai, etc.)")
    model_id: Optional[str] = Field(None, description="Model identifier")
    region: Optional[str] = Field(None, description="AWS region for bedrock")
    temperature: float = Field(0.7, description="Model temperature")
    max_tokens: int = Field(4096, description="Maximum tokens")
    
    # Additional configuration options
    client_args: Optional[Dict[str, Any]] = Field(None, description="Client configuration")
    params: Optional[Dict[str, Any]] = Field(None, description="Additional model parameters")
    
    @validator('provider', pre=True, always=True)
    def validate_config(cls, v, values):
        """Validate that either model_ref or provider/model_id is provided"""
        model_ref = values.get('model_ref')
        if not model_ref and not v:
            raise ValueError("Either model_ref or provider must be specified")
        return v


class ConversationManagerConfig(BaseModel):
    type: str = Field("sliding_window", description="Type of conversation manager")
    window_size: int = Field(40, description="Window size for sliding window")
    should_truncate_results: bool = Field(True, description="Whether to truncate results")


class ToolParameterSchema(BaseModel):
    """Schema definition for tool parameters"""
    type: str = Field(..., description="Parameter type (string, integer, boolean, array, object)")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(True, description="Whether parameter is required")
    default: Optional[Any] = Field(None, description="Default value")
    enum: Optional[List[Any]] = Field(None, description="Allowed values")
    minimum: Optional[Union[int, float]] = Field(None, description="Minimum value for numbers")
    maximum: Optional[Union[int, float]] = Field(None, description="Maximum value for numbers")
    pattern: Optional[str] = Field(None, description="Regex pattern for strings")


class CustomToolDefinition(BaseModel):
    """Definition for custom Python-based tools"""
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    type: str = Field("custom", description="Tool type (custom, built-in)")
    
    # Python function definition
    function_code: Optional[str] = Field(None, description="Python function code")
    function_file: Optional[str] = Field(None, description="Path to Python file containing function")
    function_name: str = Field(..., description="Name of the function to call")
    
    # Parameter schema
    parameters: Dict[str, ToolParameterSchema] = Field({}, description="Tool parameter schema")
    
    # Configuration
    async_execution: bool = Field(False, description="Whether function should be executed asynchronously")
    timeout: int = Field(30, description="Execution timeout in seconds")
    allowed_imports: List[str] = Field([], description="Allowed Python imports for security")
    
    # Tool metadata
    version: str = Field("1.0.0", description="Tool version")
    author: Optional[str] = Field(None, description="Tool author")
    tags: List[str] = Field([], description="Tool tags/categories")


class ToolConfig(BaseModel):
    name: str = Field(..., description="Tool name")
    enabled: bool = Field(True, description="Whether tool is enabled")
    type: str = Field("built-in", description="Tool type (built-in, custom)")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters")
    
    # For custom tools
    custom_definition: Optional[str] = Field(None, description="Reference to custom tool definition")


class HookConfig(BaseModel):
    type: str = Field(..., description="Hook type")
    config: Dict[str, Any] = Field({}, description="Hook configuration")


class StateConfig(BaseModel):
    persistent: bool = Field(False, description="Whether state is persistent")
    initial_values: Dict[str, Any] = Field({}, description="Initial state values")


class AgentConfig(BaseModel):
    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    agent_id: str = Field(..., description="Unique agent identifier")
    model: ModelConfig = Field(..., description="Model configuration")
    system_prompt: str = Field(..., description="System prompt for the agent")
    conversation_manager: ConversationManagerConfig = Field(
        ConversationManagerConfig(), description="Conversation manager config"
    )
    tools: List[ToolConfig] = Field([], description="Available tools")
    hooks: List[HookConfig] = Field([], description="Agent hooks")
    state: StateConfig = Field(StateConfig(), description="State configuration")


class ToolDefinition(BaseModel):
    description: str = Field(..., description="Tool description")
    parameters: Dict[str, Any] = Field({}, description="Tool parameters")


class ToolsConfig(BaseModel):
    tools: Dict[str, ToolDefinition] = Field({}, description="Tool definitions")
    custom_tools: Dict[str, CustomToolDefinition] = Field({}, description="Custom tool definitions")
