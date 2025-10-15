from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.services.config_service import ConfigService
from app.models.config_models import ToolConfig, CustomToolDefinition, ToolDefinition, ToolParameterSchema
from app.utils.exceptions import AgentNotFoundError

router = APIRouter(prefix="/tools", tags=["tools"])

# Dependency injection
def get_config_service():
    config_service = ConfigService()
    config_service.load_all_configs()
    return config_service

# Response models
class ToolInfo(BaseModel):
    name: str
    description: str
    type: str
    enabled: bool
    parameters: Optional[Dict[str, Any]] = None
    version: Optional[str] = "1.0.0"
    author: Optional[str] = None
    tags: Optional[List[str]] = []

class ToolListResponse(BaseModel):
    tools: List[ToolInfo]
    total_count: int
    built_in_count: int
    custom_count: int

class CreateToolRequest(BaseModel):
    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    type: str = Field("custom", description="Tool type")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Tool parameters")
    function_code: Optional[str] = Field(None, description="Python function code for custom tools")
    function_name: Optional[str] = Field(None, description="Function name for custom tools")
    enabled: bool = Field(True, description="Whether tool is enabled")

class UpdateToolRequest(BaseModel):
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    function_code: Optional[str] = None
    function_name: Optional[str] = None
    enabled: Optional[bool] = None

@router.get("/", response_model=ToolListResponse)
async def list_tools(config_service: ConfigService = Depends(get_config_service)):
    """List all available tools (built-in and custom)"""
    tools_config = config_service.get_tools_config()
    tools = []
    
    # Get built-in tools
    if tools_config and tools_config.tools:
        for tool_name, tool_def in tools_config.tools.items():
            tools.append(ToolInfo(
                name=tool_name,
                description=tool_def.description,
                type="built-in",
                enabled=True,  # Built-in tools are always enabled
                parameters=tool_def.parameters
            ))
    
    # Get custom tools
    if tools_config and tools_config.custom_tools:
        for tool_name, custom_tool in tools_config.custom_tools.items():
            tools.append(ToolInfo(
                name=tool_name,
                description=custom_tool.description,
                type=custom_tool.type,
                enabled=True,  # Custom tools in config are considered enabled
                parameters={name: param.dict() for name, param in custom_tool.parameters.items()},
                version=custom_tool.version,
                author=custom_tool.author,
                tags=custom_tool.tags
            ))
    
    built_in_count = sum(1 for tool in tools if tool.type == "built-in")
    custom_count = len(tools) - built_in_count
    
    return ToolListResponse(
        tools=tools,
        total_count=len(tools),
        built_in_count=built_in_count,
        custom_count=custom_count
    )

@router.get("/{tool_name}")
async def get_tool(tool_name: str, config_service: ConfigService = Depends(get_config_service)):
    """Get specific tool configuration"""
    tools_config = config_service.get_tools_config()
    
    if not tools_config:
        raise HTTPException(status_code=404, detail="No tools configuration found")
    
    # Check built-in tools first
    if tools_config.tools and tool_name in tools_config.tools:
        tool_def = tools_config.tools[tool_name]
        return {
            "name": tool_name,
            "description": tool_def.description,
            "type": "built-in",
            "enabled": True,
            "parameters": tool_def.parameters
        }
    
    # Check custom tools
    if tools_config.custom_tools and tool_name in tools_config.custom_tools:
        custom_tool = tools_config.custom_tools[tool_name]
        return {
            "name": tool_name,
            "description": custom_tool.description,
            "type": custom_tool.type,
            "enabled": True,
            "parameters": {name: param.dict() for name, param in custom_tool.parameters.items()},
            "function_code": custom_tool.function_code,
            "function_name": custom_tool.function_name,
            "version": custom_tool.version,
            "author": custom_tool.author,
            "tags": custom_tool.tags
        }
    
    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

@router.post("/")
async def create_tool(request: CreateToolRequest, config_service: ConfigService = Depends(get_config_service)):
    """Create a new custom tool"""
    tools_config = config_service.get_tools_config()
    
    if not tools_config:
        # Initialize tools config if it doesn't exist
        tools_config = config_service.create_empty_tools_config()
    
    # Check if tool already exists
    if (tools_config.tools and request.name in tools_config.tools) or \
       (tools_config.custom_tools and request.name in tools_config.custom_tools):
        raise HTTPException(status_code=400, detail=f"Tool '{request.name}' already exists")
    
    if request.type == "custom":
        # Create custom tool
        custom_tool = CustomToolDefinition(
            name=request.name,
            description=request.description,
            type=request.type,
            function_code=request.function_code,
            function_name=request.function_name or request.name,
            parameters={}
        )
        
        if not tools_config.custom_tools:
            tools_config.custom_tools = {}
        tools_config.custom_tools[request.name] = custom_tool
    else:
        # Create built-in tool
        tool_def = ToolDefinition(
            description=request.description,
            parameters=request.parameters or {}
        )
        
        if not tools_config.tools:
            tools_config.tools = {}
        tools_config.tools[request.name] = tool_def
    
    # Save updated configuration
    config_service.save_tools_config(tools_config)
    
    return {"message": f"Tool '{request.name}' created successfully", "tool_name": request.name}

@router.put("/{tool_name}")
async def update_tool(tool_name: str, request: UpdateToolRequest, config_service: ConfigService = Depends(get_config_service)):
    """Update an existing tool"""
    tools_config = config_service.get_tools_config()
    
    if not tools_config:
        raise HTTPException(status_code=404, detail="No tools configuration found")
    
    # Check if tool exists and update accordingly
    updated = False
    
    # Update built-in tool
    if tools_config.tools and tool_name in tools_config.tools:
        tool_def = tools_config.tools[tool_name]
        if request.description:
            tool_def.description = request.description
        if request.parameters is not None:
            tool_def.parameters = request.parameters
        updated = True
    
    # Update custom tool
    elif tools_config.custom_tools and tool_name in tools_config.custom_tools:
        custom_tool = tools_config.custom_tools[tool_name]
        if request.description:
            custom_tool.description = request.description
        if request.function_code:
            custom_tool.function_code = request.function_code
        if request.function_name:
            custom_tool.function_name = request.function_name
        updated = True
    
    if not updated:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    # Save updated configuration
    config_service.save_tools_config(tools_config)
    
    return {"message": f"Tool '{tool_name}' updated successfully"}

@router.delete("/{tool_name}")
async def delete_tool(tool_name: str, config_service: ConfigService = Depends(get_config_service)):
    """Delete a tool (only custom tools can be deleted)"""
    tools_config = config_service.get_tools_config()
    
    if not tools_config:
        raise HTTPException(status_code=404, detail="No tools configuration found")
    
    # Only allow deletion of custom tools
    if tools_config.custom_tools and tool_name in tools_config.custom_tools:
        del tools_config.custom_tools[tool_name]
        config_service.save_tools_config(tools_config)
        return {"message": f"Custom tool '{tool_name}' deleted successfully"}
    
    if tools_config.tools and tool_name in tools_config.tools:
        raise HTTPException(status_code=400, detail="Built-in tools cannot be deleted")
    
    raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

@router.get("/categories/list")
async def list_tool_categories(config_service: ConfigService = Depends(get_config_service)):
    """Get all tool categories/types"""
    tools_config = config_service.get_tools_config()
    categories = set()
    
    if tools_config:
        # Built-in tools
        if tools_config.tools:
            categories.add("built-in")
        
        # Custom tool types
        if tools_config.custom_tools:
            for custom_tool in tools_config.custom_tools.values():
                categories.add(custom_tool.type)
    
    return {"categories": list(categories)}
