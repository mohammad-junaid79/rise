from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Path, status
from pydantic import BaseModel, Field
from app.services.config_service import ConfigService
from app.models.config_models import AgentConfig, ToolDefinition
from app.utils.exceptions import AgentNotFoundError

router = APIRouter(prefix="/agent-config", tags=["agent-config"])

# Response models
class ModelCatalogInfo(BaseModel):
    model_ref: str
    display_name: str
    description: str
    provider: str
    model_id: str
    capabilities: List[str]
    pricing: Optional[Dict[str, Any]] = None
    context_window: Optional[int] = None

class ModelCatalogResponse(BaseModel):
    models: List[ModelCatalogInfo]
    categories: Dict[str, List[str]]
    use_cases: Dict[str, Dict[str, str]]

# Response models
class AgentConfigInfo(BaseModel):
    agent_name: str
    model: str
    description: str
    tags: List[str]
    tools: List[str]
    model_settings: Dict[str, Any]
    system_prompt: Optional[str] = None
    workflow_integration: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

class AgentConfigListResponse(BaseModel):
    agents: List[AgentConfigInfo]
    total_count: int

class CreateAgentRequest(BaseModel):
    agent_name: str = Field(..., description="Agent name")
    model: str = Field(..., description="Model to use")
    description: str = Field(..., description="Agent description")
    tags: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    model_settings: Dict[str, Any] = Field(default_factory=dict)
    workflow_integration: Optional[Dict[str, Any]] = Field(default_factory=dict)

class UpdateAgentRequest(BaseModel):
    model_ref: Optional[str] = None  # Model catalog reference
    model: Optional[str] = None      # For backward compatibility
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    tools: Optional[List[str]] = None
    model_settings: Optional[Dict[str, Any]] = None
    workflow_integration: Optional[Dict[str, Any]] = None

class ToolAssignmentRequest(BaseModel):
    tool_names: List[str]

# Dependency injection
def get_config_service():
    config_service = ConfigService()
    config_service.load_all_configs()
    return config_service

@router.get("/", response_model=AgentConfigListResponse)
async def list_agent_configs(
    tag: Optional[str] = Query(None, description="Filter by tag"),
    limit: Optional[int] = Query(100, description="Maximum number of agents to return"),
    offset: Optional[int] = Query(0, description="Number of agents to skip"),
    config_service: ConfigService = Depends(get_config_service)
):
    """Get list of all agent configurations with optional filtering"""
    agent_configs = config_service.get_all_agent_configs()
    agents = []
    
    for agent_name, agent_config in agent_configs.items():
        # Apply tag filter if specified
        if tag and hasattr(agent_config, 'tags') and tag not in agent_config.tags:
            continue
            
        # Create model settings from ModelConfig
        model_settings = {
            'provider': agent_config.model.provider,
            'model_id': agent_config.model.model_id,
            'temperature': agent_config.model.temperature,
            'max_tokens': agent_config.model.max_tokens
        }
        if agent_config.model.region:
            model_settings['region'] = agent_config.model.region
            
        agent_info = AgentConfigInfo(
            agent_name=agent_name,
            model=agent_config.model.model_id,
            description=agent_config.description,
            tags=getattr(agent_config, 'tags', []),
            tools=[tool.name for tool in agent_config.tools if hasattr(tool, 'name')],
            model_settings=model_settings,
            system_prompt=getattr(agent_config, 'system_prompt', None),
            workflow_integration=getattr(agent_config, 'workflow_integration', {})
        )
        agents.append(agent_info)
    
    # Apply pagination
    total_count = len(agents)
    agents = agents[offset:offset + limit]
    
    return AgentConfigListResponse(agents=agents, total_count=total_count)

@router.get("/{agent_name}")
async def get_agent_config(agent_name: str, config_service: ConfigService = Depends(get_config_service)):
    """Get specific agent configuration"""
    try:
        agent_config = config_service.get_agent_config(agent_name)
        if not agent_config:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Create model settings from ModelConfig
        model_settings = {
            'provider': agent_config.model.provider,
            'model_id': agent_config.model.model_id,
            'temperature': agent_config.model.temperature,
            'max_tokens': agent_config.model.max_tokens
        }
        if agent_config.model.region:
            model_settings['region'] = agent_config.model.region
        
        return AgentConfigInfo(
            agent_name=agent_name,
            model=agent_config.model.model_id,
            description=agent_config.description,
            tags=getattr(agent_config, 'tags', []),
            tools=[tool.name for tool in agent_config.tools if hasattr(tool, 'name')],
            model_settings=model_settings,
            system_prompt=getattr(agent_config, 'system_prompt', None),
            workflow_integration=getattr(agent_config, 'workflow_integration', {})
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found: {str(e)}")

@router.post("/")
async def create_agent_config(request: CreateAgentRequest, config_service: ConfigService = Depends(get_config_service)):
    """Create a new agent configuration"""
    # Check if agent already exists
    existing_configs = config_service.get_all_agent_configs()
    if request.agent_name in existing_configs:
        raise HTTPException(status_code=400, detail=f"Agent '{request.agent_name}' already exists")
    
    # Create new agent config
    tools_config = []
    if request.tools:
        # Convert tool names to tool definitions
        available_tools = config_service.get_tools_config()
        for tool_name in request.tools:
            if available_tools and available_tools.tools and tool_name in available_tools.tools:
                tool_def = available_tools.tools[tool_name]
                tools_config.append(ToolDefinition(
                    name=tool_name,
                    description=tool_def.description,
                    enabled=True,
                    parameters=tool_def.parameters
                ))
    
    new_agent_config = AgentConfig(
        name=request.agent_name,
        agent_id=request.agent_name.lower().replace(" ", "_"),
        description=request.description,
        model=request.model,
        tags=request.tags,
        tools=tools_config,
        model_config=request.model_settings
    )
    
    # Save the new configuration
    success = config_service.save_agent_config(request.agent_name, new_agent_config)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save agent configuration")
    
    return {"message": f"Agent '{request.agent_name}' created successfully", "agent_name": request.agent_name}

@router.put("/{agent_name}")
async def update_agent_config(
    agent_name: str, 
    request: UpdateAgentRequest, 
    config_service: ConfigService = Depends(get_config_service)
):
    """Update an existing agent configuration"""
    try:
        # Get existing configuration
        agent_config = config_service.get_agent_config(agent_name)
        if not agent_config:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Handle model reference change
        model_ref_changed = False
        new_model_ref = None
        model_params = None
        
        if request.model_ref:
            # Validate model reference exists in catalog
            if not config_service.model_catalog.validate_model_ref(request.model_ref):
                raise HTTPException(status_code=400, detail=f"Invalid model reference: {request.model_ref}")
            
            # Get new model from catalog and resolve it
            catalog_model = config_service.model_catalog.get_model(request.model_ref)
            if not catalog_model:
                raise HTTPException(status_code=400, detail=f"Model reference not found: {request.model_ref}")
            
            # Preserve existing parameter overrides or use new ones
            current_params = agent_config.model.params or {}
            new_params = request.model_settings or {}
            model_params = {**current_params, **new_params}
            
            # Create new resolved model config
            from app.models.config_models import ModelConfig
            agent_config.model = ModelConfig(
                provider=catalog_model.provider,
                model_id=catalog_model.model_id,
                region=catalog_model.region,
                temperature=model_params.get('temperature', catalog_model.temperature),
                max_tokens=model_params.get('max_tokens', catalog_model.max_tokens),
                client_args=catalog_model.client_args,
                params=model_params
            )
            
            model_ref_changed = True
            new_model_ref = request.model_ref
        
        elif request.model:
            # Handle backward compatibility - direct model ID
            agent_config.model.model_id = request.model
            
        # Update other fields if provided
        if request.description:
            agent_config.description = request.description
        if request.tags is not None:
            # Note: AgentConfig might not have tags field, handle gracefully
            if hasattr(agent_config, 'tags'):
                agent_config.tags = request.tags
        if request.model_settings is not None and not model_ref_changed:
            # Update model settings if not changing model reference
            agent_config.model.params = {**(agent_config.model.params or {}), **request.model_settings}
            # Update direct fields
            if 'temperature' in request.model_settings:
                agent_config.model.temperature = request.model_settings['temperature']
            if 'max_tokens' in request.model_settings:
                agent_config.model.max_tokens = request.model_settings['max_tokens']
        
        # Update tools if provided
        if request.tools is not None:
            from app.models.config_models import ToolConfig
            tools_config = []
            for tool_name in request.tools:
                tools_config.append(ToolConfig(
                    name=tool_name,
                    enabled=True,
                    type="built-in"
                ))
            agent_config.tools = tools_config
        
        # Save updated configuration
        if model_ref_changed:
            success = config_service.save_agent_config_with_model_ref(agent_name, agent_config, new_model_ref, model_params)
        else:
            success = config_service.save_agent_config(agent_name, agent_config)
            
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update agent configuration")
        
        return {"message": f"Agent '{agent_name}' updated successfully"}
        
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")

@router.delete("/{agent_name}")
async def delete_agent_config(agent_name: str, config_service: ConfigService = Depends(get_config_service)):
    """Delete an agent configuration"""
    try:
        # Check if agent exists
        agent_config = config_service.get_agent_config(agent_name)
        if not agent_config:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Delete the configuration
        success = config_service.delete_agent_config(agent_name)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete agent configuration")
        
        return {"message": f"Agent '{agent_name}' deleted successfully"}
        
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")

@router.post("/{agent_name}/tools")
async def assign_tools_to_agent(
    agent_name: str, 
    request: ToolAssignmentRequest,
    config_service: ConfigService = Depends(get_config_service)
):
    """Assign tools to an agent"""
    try:
        # Get existing configuration
        agent_config = config_service.get_agent_config(agent_name)
        if not agent_config:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Validate tools exist
        available_tools = config_service.get_tools_config()
        tools_config = []
        
        for tool_name in request.tool_names:
            if available_tools and available_tools.tools and tool_name in available_tools.tools:
                tool_def = available_tools.tools[tool_name]
                tools_config.append(ToolDefinition(
                    name=tool_name,
                    description=tool_def.description,
                    enabled=True,
                    parameters=tool_def.parameters
                ))
            else:
                raise HTTPException(status_code=400, detail=f"Tool '{tool_name}' not found")
        
        # Update agent tools
        agent_config.tools = tools_config
        
        # Save updated configuration
        success = config_service.save_agent_config(agent_name, agent_config)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update agent tools")
        
        return {
            "message": f"Tools assigned to agent '{agent_name}' successfully",
            "tools_assigned": request.tool_names
        }
        
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error assigning tools: {str(e)}")

@router.get("/{agent_name}/tools")
async def get_agent_tools(agent_name: str, config_service: ConfigService = Depends(get_config_service)):
    """Get tools assigned to an agent"""
    try:
        agent_config = config_service.get_agent_config(agent_name)
        if not agent_config:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        tools = []
        for tool in agent_config.tools:
            if hasattr(tool, 'name'):
                tools.append({
                    "name": tool.name,
                    "description": getattr(tool, 'description', ''),
                    "enabled": getattr(tool, 'enabled', True),
                    "parameters": getattr(tool, 'parameters', {})
                })
        
        return {"agent_name": agent_name, "tools": tools, "total_count": len(tools)}
        
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting agent tools: {str(e)}")

@router.delete("/{agent_name}/tools/{tool_name}")
async def remove_tool_from_agent(
    agent_name: str, 
    tool_name: str, 
    config_service: ConfigService = Depends(get_config_service)
):
    """Remove a specific tool from an agent"""
    try:
        # Get existing configuration
        agent_config = config_service.get_agent_config(agent_name)
        if not agent_config:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
        
        # Remove the tool
        updated_tools = [tool for tool in agent_config.tools if getattr(tool, 'name', '') != tool_name]
        
        if len(updated_tools) == len(agent_config.tools):
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found on agent '{agent_name}'")
        
        agent_config.tools = updated_tools
        
        # Save updated configuration
        success = config_service.save_agent_config(agent_name, agent_config)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to remove tool from agent")
        
        return {"message": f"Tool '{tool_name}' removed from agent '{agent_name}' successfully"}
        
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error removing tool: {str(e)}")

@router.get("/models/catalog", response_model=ModelCatalogResponse)
async def get_model_catalog(config_service: ConfigService = Depends(get_config_service)):
    """Get available models from the model catalog"""
    try:
        catalog = config_service.model_catalog
        models_info = []
        
        # Load catalog data
        import yaml
        with open(catalog.catalog_file, 'r') as f:
            catalog_data = yaml.safe_load(f)
        
        # Convert models to response format
        for model_ref, model_data in catalog_data.get('models', {}).items():
            model_info = ModelCatalogInfo(
                model_ref=model_ref,
                display_name=model_data.get('display_name', model_ref),
                description=model_data.get('description', ''),
                provider=model_data.get('provider', ''),
                model_id=model_data.get('model_id', ''),
                capabilities=model_data.get('capabilities', []),
                pricing=model_data.get('pricing'),
                context_window=model_data.get('context_window')
            )
            models_info.append(model_info)
        
        return ModelCatalogResponse(
            models=models_info,
            categories=catalog_data.get('categories', {}),
            use_cases=catalog_data.get('use_cases', {})
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model catalog: {str(e)}")
