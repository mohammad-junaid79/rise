from fastapi import HTTPException


class AgentNotFoundError(HTTPException):
    def __init__(self, agent_name: str):
        super().__init__(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )


class ConfigurationError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=400,
            detail=f"Configuration error: {message}"
        )


class SessionNotFoundError(HTTPException):
    def __init__(self, session_id: str):
        super().__init__(
            status_code=404,
            detail=f"Session '{session_id}' not found"
        )


class AgentExecutionError(HTTPException):
    def __init__(self, message: str):
        super().__init__(
            status_code=500,
            detail=f"Agent execution error: {message}"
        )


class ToolExecutionError(HTTPException):
    def __init__(self, tool_name: str, message: str):
        super().__init__(
            status_code=500,
            detail=f"Tool '{tool_name}' execution error: {message}"
        )


class ContextWindowOverflowError(HTTPException):
    def __init__(self, message: str = "Context window overflow"):
        super().__init__(
            status_code=413,
            detail=message
        )


class TimeoutError(HTTPException):
    def __init__(self, timeout: int):
        super().__init__(
            status_code=408,
            detail=f"Agent execution timed out after {timeout} seconds"
        )
