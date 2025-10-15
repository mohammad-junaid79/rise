import os
from dotenv import load_dotenv
from mangum import Mangum
from pydantic import BaseModel


# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import agent_router, workflow_router, agent_config_router, tools_router
from app.utils.exceptions import (
    AgentNotFoundError, ConfigurationError, SessionNotFoundError,
    AgentExecutionError, ToolExecutionError, ContextWindowOverflowError, TimeoutError
)
from fastapi import Request
from fastapi.responses import JSONResponse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AWS Strands Agent Platform",
    description="FastAPI backend for agent orchestration using AWS Strands SDK",
    version="1.0.0"
)

# CORS middleware
"""app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)"""

@app.get("/")
async def root():
    return {
        "message": "AWS Strands Agent Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "agent-platform"}

# Global exception handlers
@app.exception_handler(AgentNotFoundError)
async def agent_not_found_handler(request: Request, exc: AgentNotFoundError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Agent not found", "detail": exc.detail}
    )

@app.exception_handler(ConfigurationError)
async def configuration_error_handler(request: Request, exc: ConfigurationError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Configuration error", "detail": exc.detail}
    )

@app.exception_handler(SessionNotFoundError)
async def session_not_found_handler(request: Request, exc: SessionNotFoundError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Session not found", "detail": exc.detail}
    )

@app.exception_handler(AgentExecutionError)
async def agent_execution_error_handler(request: Request, exc: AgentExecutionError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Agent execution error", "detail": exc.detail}
    )

@app.exception_handler(ToolExecutionError)
async def tool_execution_error_handler(request: Request, exc: ToolExecutionError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Tool execution error", "detail": exc.detail}
    )

@app.exception_handler(ContextWindowOverflowError)
async def context_overflow_handler(request: Request, exc: ContextWindowOverflowError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Context window overflow", "detail": exc.detail}
    )

@app.exception_handler(TimeoutError)
async def timeout_error_handler(request: Request, exc: TimeoutError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "Execution timeout", "detail": exc.detail}
    )

# Include routers
app.include_router(agent_router.router)
app.include_router(workflow_router.router)
app.include_router(agent_config_router.router)
app.include_router(tools_router.router)

"""if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)"""


handler = Mangum(app, lifespan="off")

