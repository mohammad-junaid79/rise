from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# Request Models
class AgentExecutionRequest(BaseModel):
    agent_config: str = Field(..., description="Agent config file name or inline config")
    prompt: str = Field(..., description="User prompt for the agent")
    session_id: Optional[str] = Field(None, description="Session ID for stateful conversations")
    max_iterations: int = Field(10, description="Maximum number of iterations")
    timeout: int = Field(300, description="Timeout in seconds")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class StreamingRequest(AgentExecutionRequest):
    stream_events: List[str] = Field(
        ["text_delta", "tool_call", "final_response"],
        description="Events to stream to client"
    )


# Response Models
class ToolCall(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    success: bool = True


class AgentExecutionResponse(BaseModel):
    session_id: str
    agent_name: str
    response: str
    execution_time: float
    token_usage: Optional[Dict[str, int]] = None
    tool_calls: List[ToolCall] = []
    stop_reason: str
    metrics: Dict[str, Any] = {}


class StreamingEvent(BaseModel):
    event_type: str
    timestamp: datetime
    session_id: str
    data: Dict[str, Any]


class ErrorResponse(BaseModel):
    error: str
    detail: str
    timestamp: datetime
    session_id: Optional[str] = None


class HealthCheckResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    agents_loaded: int


class AgentInfo(BaseModel):
    name: str
    description: str
    agent_id: str
    config_key: str  # The config filename key used for API calls
    available_tools: List[str]
    status: str = "active"


class SessionInfo(BaseModel):
    session_id: str
    agent_name: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    state: Dict[str, Any] = {}
