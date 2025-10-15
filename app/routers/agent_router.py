from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from app.models.agent_models import (
    AgentExecutionRequest, AgentExecutionResponse, 
    StreamingRequest, HealthCheckResponse, AgentInfo, SessionInfo
)
from app.services.agent_service import StrandsAgentService
from app.services.config_service import ConfigService
from app.utils.exceptions import AgentNotFoundError, SessionNotFoundError
import json


router = APIRouter(prefix="/agents", tags=["agents"])

# Dependency injection
config_service = ConfigService()
config_service.load_all_configs()
agent_service = StrandsAgentService(config_service)


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        agents_loaded=len(config_service.list_available_agents())
    )


@router.get("/", response_model=List[AgentInfo])
async def list_agents():
    """List all available agents"""
    agents = []
    for agent_name, agent_config in config_service.get_all_agent_configs().items():
        available_tools = [tool.name for tool in agent_config.tools if tool.enabled]
        agents.append(AgentInfo(
            name=agent_config.name,
            description=agent_config.description,
            agent_id=agent_config.agent_id,
            config_key=agent_name,  # The config filename key
            available_tools=available_tools,
            status="active"
        ))
    return agents


@router.get("/{agent_name}/config")
async def get_agent_config(agent_name: str):
    """Get agent configuration"""
    config = config_service.get_agent_config(agent_name)
    if not config:
        raise AgentNotFoundError(agent_name)
    return config


@router.post("/execute", response_model=AgentExecutionResponse)
async def execute_agent(request: AgentExecutionRequest):
    """Execute agent with the given prompt - creates a new session automatically"""
    # Remove any existing session_id to force creation of new session
    request.session_id = None
    return await agent_service.execute_agent(request)


@router.post("/continue", response_model=AgentExecutionResponse)
async def continue_conversation(request: AgentExecutionRequest):
    """Continue an existing conversation with the provided session_id, or create new session if not provided"""
    return await agent_service.execute_agent(request)


@router.post("/{agent_name}/execute", response_model=AgentExecutionResponse)
async def execute_specific_agent(agent_name: str, request: AgentExecutionRequest):
    """Execute specific agent - creates a new session automatically"""
    request.agent_config = agent_name
    # Remove any existing session_id to force creation of new session
    request.session_id = None
    return await agent_service.execute_agent(request)


@router.post("/{agent_name}/continue", response_model=AgentExecutionResponse)
async def continue_specific_agent(agent_name: str, request: AgentExecutionRequest):
    """Continue conversation with specific agent using existing session_id, or create new session if not provided"""
    request.agent_config = agent_name
    return await agent_service.execute_agent(request)


@router.post("/stream")
async def stream_agent_execution(request: StreamingRequest):
    """Stream agent execution events - creates a new session automatically"""
    # Remove any existing session_id to force creation of new session
    request.session_id = None
    
    async def event_generator():
        async for event in agent_service.stream_agent_execution(request):
            yield f"data: {json.dumps(event.dict(), default=str)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/stream/continue")
async def stream_continue_conversation(request: StreamingRequest):
    """Stream agent execution events for continuing conversation with existing session_id"""
    
    async def event_generator():
        async for event in agent_service.stream_agent_execution(request):
            yield f"data: {json.dumps(event.dict(), default=str)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/{agent_name}/stream")
async def stream_specific_agent(agent_name: str, request: StreamingRequest):
    """Stream specific agent execution - creates a new session automatically"""
    request.agent_config = agent_name
    # Remove any existing session_id to force creation of new session
    request.session_id = None
    
    async def event_generator():
        async for event in agent_service.stream_agent_execution(request):
            yield f"data: {json.dumps(event.dict(), default=str)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/{agent_name}/stream/continue")
async def stream_continue_specific_agent(agent_name: str, request: StreamingRequest):
    """Stream specific agent execution for continuing conversation with existing session_id"""
    request.agent_config = agent_name
    
    async def event_generator():
        async for event in agent_service.stream_agent_execution(request):
            yield f"data: {json.dumps(event.dict(), default=str)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# Session Management Endpoints
@router.get("/sessions", response_model=SessionInfo)
async def get_session(session_id: str = Query(..., description="Session ID to retrieve")):
    """Get session information by session ID provided as query parameter"""
    return agent_service.get_session(session_id)


@router.get("/sessions/list", response_model=List[SessionInfo])
async def list_all_sessions():
    """List all active sessions"""
    return agent_service.list_sessions()


@router.delete("/sessions")
async def clear_session(session_id: str = Query(..., description="Session ID to clear")):
    """Clear session by session ID provided as query parameter"""
    success = agent_service.clear_session(session_id)
    if not success:
        raise SessionNotFoundError(session_id)
    return {"message": "Session cleared successfully", "session_id": session_id}


@router.post("/sessions/reset")
async def reset_session(session_id: str = Query(..., description="Session ID to reset")):
    """Reset session state by session ID provided as query parameter"""
    success = agent_service.clear_session(session_id)
    if not success:
        raise SessionNotFoundError(session_id)
    return {"message": "Session reset successfully", "session_id": session_id}


@router.delete("/sessions/all")
async def clear_all_sessions():
    """Clear all active sessions"""
    count = agent_service.clear_all_sessions()
    return {"message": f"Cleared {count} sessions successfully"}


# Configuration Management
@router.post("/reload")
async def reload_configurations():
    """Reload all agent configurations"""
    try:
        config_service.reload_configs()
        return {
            "message": "Configurations reloaded successfully",
            "agents_loaded": len(config_service.list_available_agents())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload configurations: {str(e)}")


@router.post("/tools/reload")
async def reload_tool_configurations():
    """Reload tool configurations"""
    try:
        config_service.reload_configs()
        return {
            "message": "Tool configurations reloaded successfully",
            "tools_available": config_service.list_available_tools()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload tool configurations: {str(e)}")


# Custom Tools Management
@router.get("/tools/custom")
async def list_custom_tools():
    """List all available custom tools"""
    tools = agent_service.get_custom_tools()
    tool_details = []
    
    for tool_name in tools:
        definition = agent_service.get_custom_tool_definition(tool_name)
        if definition:
            tool_details.append({
                "name": tool_name,
                "description": definition.get("description", ""),
                "version": definition.get("version", ""),
                "author": definition.get("author", ""),
                "tags": definition.get("tags", []),
                "parameters": list(definition.get("parameters", {}).keys())
            })
    
    return {
        "custom_tools": tool_details,
        "total_count": len(tools)
    }


@router.get("/tools/custom/{tool_name}")
async def get_custom_tool_definition(tool_name: str):
    """Get detailed definition of a custom tool"""
    definition = agent_service.get_custom_tool_definition(tool_name)
    if not definition:
        raise HTTPException(status_code=404, detail=f"Custom tool '{tool_name}' not found")
    
    return {
        "tool_name": tool_name,
        "definition": definition
    }


@router.post("/tools/custom/register")
async def register_custom_tool(tool_definition: dict):
    """Register a new custom tool"""
    try:
        success = agent_service.register_custom_tool(tool_definition)
        if success:
            return {
                "message": f"Custom tool '{tool_definition.get('name')}' registered successfully",
                "tool_name": tool_definition.get("name")
            }
        else:
            raise HTTPException(status_code=400, detail="Failed to register custom tool - validation failed")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error registering custom tool: {str(e)}")


@router.post("/tools/custom/{tool_name}/test")
async def test_custom_tool(tool_name: str, test_parameters: dict):
    """Test a custom tool with provided parameters"""
    try:
        tool = agent_service.custom_tool_service.get_tool(tool_name)
        if not tool:
            raise HTTPException(status_code=404, detail=f"Custom tool '{tool_name}' not found")
        
        # Execute the tool with test parameters
        result = tool(**test_parameters)
        
        return {
            "tool_name": tool_name,
            "test_parameters": test_parameters,
            "result": result,
            "success": True
        }
    except Exception as e:
        return {
            "tool_name": tool_name,
            "test_parameters": test_parameters,
            "error": str(e),
            "success": False
        }
