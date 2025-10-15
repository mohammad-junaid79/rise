from datetime import datetime
from typing import List
import uuid
import asyncio
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from app.models.workflow_models import (
    WorkflowConfig, WorkflowExecutionRequest, WorkflowExecutionResult,
    WorkflowStatusResponse, WorkflowInfo, ExecutionStatus,
    CreateWorkflowRequest, UpdateWorkflowRequest
)
from app.services.workflow_service import WorkflowOrchestrationService

router = APIRouter(prefix="/workflows", tags=["workflows"])
workflow_service = WorkflowOrchestrationService()


@router.get("/health")
async def health_check():
    """Workflow service health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "workflows_loaded": len(workflow_service.workflows),
        "active_executions": len(workflow_service.active_executions)
    }


@router.get("/", response_model=List[WorkflowInfo])
async def list_workflows():
    """List all available workflows"""
    workflows = workflow_service.list_workflows()
    
    workflow_infos = []
    for workflow in workflows:
        workflow_info = WorkflowInfo(
            workflow_id=workflow.workflow_id,
            name=workflow.name,
            description=workflow.description,
            topology=workflow.topology,
            version=workflow.version,
            node_count=len(workflow.nodes),
            tags=workflow.tags,
            created_at=workflow.created_at,
            updated_at=workflow.updated_at
        )
        workflow_infos.append(workflow_info)
    
    return workflow_infos


@router.get("/{workflow_id}", response_model=WorkflowConfig)
async def get_workflow(workflow_id: str):
    """Get specific workflow configuration"""
    workflow = workflow_service.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
    return workflow


@router.post("/execute-stream")
async def execute_workflow_stream(request: WorkflowExecutionRequest):
    """Execute a workflow with streaming output"""
    
    async def workflow_stream():
        """Stream workflow execution progress"""
        try:
            # Send initial status
            event_data = {
                "workflow_id": request.workflow_id,
                "status": "started",
                "timestamp": datetime.now().isoformat()
            }
            yield f"event: workflow_started\ndata: {json.dumps(event_data)}\n\n"
            
            # Execute workflow with streaming
            async for event in workflow_service.execute_workflow_stream(request):
                event_type = event.get("event", "unknown")
                event_data = event.get("data", {})
                yield f"event: {event_type}\ndata: {json.dumps(event_data, default=str)}\n\n"
                
        except Exception as e:
            error_data = {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
    
    return EventSourceResponse(workflow_stream())


@router.post("/execute", response_model=WorkflowExecutionResult)
async def execute_workflow(request: WorkflowExecutionRequest):
    """Execute a workflow"""
    try:
        result = await workflow_service.execute_workflow(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {str(e)}")


@router.post("/execute-async")
async def execute_workflow_async(request: WorkflowExecutionRequest, background_tasks: BackgroundTasks):
    """Execute a workflow asynchronously"""
    try:
        # Start workflow execution in background
        execution_id = str(uuid.uuid4())
        
        async def run_workflow():
            try:
                result = await workflow_service.execute_workflow(request)
                # Store result for later retrieval
                workflow_service.execution_results[execution_id] = result
            except Exception as e:
                workflow_service.execution_results[execution_id] = WorkflowExecutionResult(
                    workflow_id=request.workflow_id,
                    execution_id=execution_id,
                    status=ExecutionStatus.FAILED,
                    errors=[str(e)]
                )
        
        background_tasks.add_task(run_workflow)
        
        return {
            "execution_id": execution_id,
            "status": "started",
            "message": "Workflow execution started in background"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start workflow: {str(e)}")


@router.get("/execution/{execution_id}/status", response_model=WorkflowStatusResponse)
async def get_execution_status(execution_id: str):
    """Get workflow execution status"""
    if execution_id in workflow_service.active_executions:
        execution = workflow_service.active_executions[execution_id]
        
        # Calculate progress (simplified)
        total_nodes = len(workflow_service.get_workflow(execution["workflow_id"]).nodes)
        completed_nodes = len(execution.get("completed_nodes", []))
        progress = (completed_nodes / total_nodes) * 100 if total_nodes > 0 else 0
        
        return WorkflowStatusResponse(
            workflow_id=execution["workflow_id"],
            execution_id=execution_id,
            status=ExecutionStatus.RUNNING,
            progress_percentage=progress,
            current_nodes=execution.get("current_nodes", []),
            completed_nodes=execution.get("completed_nodes", []),
            failed_nodes=execution.get("failed_nodes", []),
            execution_time=(datetime.now() - execution["start_time"]).total_seconds()
        )
    
    # Check if execution is completed
    if hasattr(workflow_service, 'execution_results') and execution_id in workflow_service.execution_results:
        result = workflow_service.execution_results[execution_id]
        return WorkflowStatusResponse(
            workflow_id=result.workflow_id,
            execution_id=execution_id,
            status=result.status,
            progress_percentage=100.0,
            completed_nodes=[r.node_id for r in result.node_results if r.status == ExecutionStatus.COMPLETED],
            failed_nodes=[r.node_id for r in result.node_results if r.status == ExecutionStatus.FAILED],
            execution_time=result.total_execution_time
        )
    
    raise HTTPException(status_code=404, detail=f"Execution not found: {execution_id}")


@router.get("/execution/{execution_id}/result", response_model=WorkflowExecutionResult)
async def get_execution_result(execution_id: str):
    """Get workflow execution result"""
    if hasattr(workflow_service, 'execution_results') and execution_id in workflow_service.execution_results:
        return workflow_service.execution_results[execution_id]
    
    raise HTTPException(status_code=404, detail=f"Execution result not found: {execution_id}")


@router.delete("/execution/{execution_id}")
async def cancel_execution(execution_id: str):
    """Cancel a running workflow execution"""
    if execution_id in workflow_service.active_executions:
        # Mark execution for cancellation
        workflow_service.active_executions[execution_id]["cancelled"] = True
        return {"message": f"Execution {execution_id} marked for cancellation"}
    
    raise HTTPException(status_code=404, detail=f"Execution not found: {execution_id}")


@router.post("/reload")
async def reload_workflows():
    """Reload workflow configurations from files"""
    try:
        workflow_service.load_workflows()
        return {
            "message": "Workflows reloaded successfully",
            "workflows_loaded": len(workflow_service.workflows)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload workflows: {str(e)}")


@router.get("/{workflow_id}/validate")
async def validate_workflow(workflow_id: str):
    """Validate workflow configuration"""
    workflow = workflow_service.get_workflow(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {workflow_id}")
    
    validation_results = {
        "valid": True,
        "errors": [],
        "warnings": []
    }
    
    # Basic validation
    if not workflow.nodes:
        validation_results["valid"] = False
        validation_results["errors"].append("Workflow has no nodes")
    
    # Check for duplicate node IDs
    node_ids = [node.node_id for node in workflow.nodes]
    if len(node_ids) != len(set(node_ids)):
        validation_results["valid"] = False
        validation_results["errors"].append("Duplicate node IDs found")
    
    # Validate edges for graph topology
    if workflow.topology == "graph" and workflow.edges:
        node_id_set = set(node_ids)
        for edge in workflow.edges:
            if edge.from_node not in node_id_set:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Edge references non-existent node: {edge.from_node}")
            if edge.to_node not in node_id_set:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Edge references non-existent node: {edge.to_node}")
    
    # Check agent configurations
    for node in workflow.nodes:
        if node.type == "agent" and node.agent_config:
            agent_config = workflow_service.config_service.get_agent_config(node.agent_config)
            if not agent_config:
                validation_results["valid"] = False
                validation_results["errors"].append(f"Agent config not found: {node.agent_config}")
    
    return validation_results


@router.post("/create", response_model=dict)
async def create_workflow(request: CreateWorkflowRequest):
    """Create a new workflow with optional dynamic agent creation"""
    try:
        # First, create any new agents that were defined in the workflow
        created_agents = []
        if request.new_agents:
            from app.services.config_service import ConfigService
            from app.models.config_models import AgentConfig, ModelConfig, ConversationManagerConfig, ToolConfig
            
            config_service = ConfigService()
            
            for agent_data in request.new_agents:
                agent_name = agent_data.get('agent_name')
                if not agent_name:
                    continue
                    
                # Check if agent already exists
                if config_service.get_agent_config(agent_name):
                    continue  # Skip if agent already exists
                
                # Create model configuration
                model_ref = agent_data.get('model_ref')
                model_settings = agent_data.get('model_settings', {})
                
                if model_ref:
                    # Validate model reference
                    if not config_service.model_catalog.validate_model_ref(model_ref):
                        raise HTTPException(status_code=400, detail=f"Invalid model reference: {model_ref}")
                    
                    # Get catalog model and create resolved config
                    catalog_model = config_service.model_catalog.get_model(model_ref)
                    model_config = ModelConfig(
                        provider=catalog_model.provider,
                        model_id=catalog_model.model_id,
                        region=catalog_model.region,
                        temperature=model_settings.get('temperature', catalog_model.temperature),
                        max_tokens=model_settings.get('max_tokens', catalog_model.max_tokens),
                        client_args=catalog_model.client_args,
                        params={**(catalog_model.default_params or {}), **model_settings}
                    )
                else:
                    # Create model config from direct settings
                    model_config = ModelConfig(
                        provider=agent_data.get('provider', 'litellm'),
                        model_id=agent_data.get('model', 'gpt-4'),
                        temperature=model_settings.get('temperature', 0.7),
                        max_tokens=model_settings.get('max_tokens', 4096)
                    )
                
                # Create agent configuration
                agent_config = AgentConfig(
                    name=agent_data.get('name', agent_name),
                    description=agent_data.get('description', f'Agent created for workflow {request.name}'),
                    agent_id=f"{agent_name}-{uuid.uuid4().hex[:8]}",
                    model=model_config,
                    system_prompt=agent_data.get('system_prompt', 'You are a helpful AI assistant.'),
                    conversation_manager=ConversationManagerConfig(),
                    tools=[
                        ToolConfig(name=tool_name, enabled=True, type="built-in")
                        for tool_name in agent_data.get('tools', [])
                    ],
                    hooks=[],
                    state={}
                )
                
                # Save agent with model reference if applicable
                if model_ref:
                    success = config_service.save_agent_config_with_model_ref(
                        agent_name, agent_config, model_ref, model_settings
                    )
                else:
                    success = config_service.save_agent_config(agent_name, agent_config)
                
                if success:
                    created_agents.append(agent_name)
        
        # Convert nodes and edges to proper workflow format
        workflow_nodes = []
        
        # Find nodes that need to be linked to created agents
        new_agent_nodes = [
            node for node in request.nodes 
            if node.get('type') == 'agent' and not node.get('agent_config')
        ]
        
        for node_data in request.nodes:
            from app.models.workflow_models import WorkflowNode, NodeType
            
            # Determine agent_config: use existing or link to newly created agent
            agent_config = node_data.get('agent_config')
            node_id = node_data.get('id', node_data.get('node_id'))
            
            if not agent_config and node_data.get('type') == 'agent' and created_agents:
                # For new agent nodes, try to match with created agents
                # If there's only one created agent and one new node, link them
                if len(created_agents) == 1 and len(new_agent_nodes) == 1:
                    agent_config = created_agents[0]
                else:
                    # Try to match by name similarity
                    node_name = node_data.get('name', '').lower().replace(' ', '_')
                    for agent_name in created_agents:
                        if node_name in agent_name or agent_name in node_name:
                            agent_config = agent_name
                            break
                    
                    # If no match found, use the first created agent
                    if not agent_config and created_agents:
                        agent_config = created_agents[0]
            
            node = WorkflowNode(
                node_id=node_id,
                name=node_data.get('name', 'Unnamed Node'),
                type=NodeType.AGENT if node_data.get('type') == 'agent' else NodeType.CONDITION,
                agent_config=agent_config,
                system_prompt=node_data.get('system_prompt'),
                condition_expression=node_data.get('condition_expression'),
                timeout=node_data.get('timeout', 300),
                retry_count=node_data.get('retry_count', 0),
                priority=node_data.get('priority', 0),
                context_keys=node_data.get('context_keys')
            )
            workflow_nodes.append(node)
        
        # Convert edges if provided and create communication patterns
        workflow_edges = []
        communication_patterns = []
        if request.edges:
            from app.models.workflow_models import WorkflowEdge, CommunicationPattern
            for edge_data in request.edges:
                source = edge_data.get('from_node', edge_data.get('source'))
                target = edge_data.get('to_node', edge_data.get('target'))
                
                edge = WorkflowEdge(
                    from_node=source,
                    to_node=target,
                    condition=edge_data.get('condition')
                )
                workflow_edges.append(edge)
                
                # Create communication pattern for this connection
                if source and target:
                    pattern = CommunicationPattern(
                        pattern="direct",
                        source_nodes=[source],
                        target_nodes=[target],
                        message_template=edge_data.get('condition') or f"Output from {source}"
                    )
                    communication_patterns.append(pattern)
        
        # Create context configuration
        from app.models.workflow_models import ContextConfig
        context_config = ContextConfig(
            shared_keys=["workflow_input", "workflow_context"],
            node_specific_keys={node.node_id: [f"{node.node_id}_output", f"{node.node_id}_context"] for node in workflow_nodes},
            persistence=True,
            isolation_level="workflow"
        )
        
        # Create workflow configuration
        workflow_config = WorkflowConfig(
            workflow_id=request.workflow_id,
            name=request.name,
            description=request.description,
            version=request.version,
            topology=request.topology,
            nodes=workflow_nodes,
            edges=workflow_edges,
            communication=communication_patterns,
            context=context_config,
            tags=request.tags,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Save workflow configuration
        workflow_service.save_workflow_config(workflow_config)
        
        return {
            "message": f"Workflow '{request.name}' created successfully",
            "workflow_id": request.workflow_id,
            "created_agents": created_agents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating workflow: {str(e)}")


@router.put("/{workflow_id}", response_model=dict)
async def update_workflow(workflow_id: str, request: UpdateWorkflowRequest):
    """Update an existing workflow"""
    try:
        # Get existing workflow
        workflow_config = workflow_service.get_workflow_config(workflow_id)
        if not workflow_config:
            raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
        
        # Create any new agents if specified
        created_agents = []
        if request.new_agents:
            from app.services.config_service import ConfigService
            from app.models.config_models import AgentConfig, ModelConfig, ConversationManagerConfig, ToolConfig
            
            config_service = ConfigService()
            
            for agent_data in request.new_agents:
                agent_name = agent_data.get('agent_name')
                if not agent_name or config_service.get_agent_config(agent_name):
                    continue  # Skip if no name or already exists
                
                # Same agent creation logic as in create_workflow
                model_ref = agent_data.get('model_ref')
                model_settings = agent_data.get('model_settings', {})
                
                if model_ref:
                    catalog_model = config_service.model_catalog.get_model(model_ref)
                    model_config = ModelConfig(
                        provider=catalog_model.provider,
                        model_id=catalog_model.model_id,
                        region=catalog_model.region,
                        temperature=model_settings.get('temperature', catalog_model.temperature),
                        max_tokens=model_settings.get('max_tokens', catalog_model.max_tokens),
                        client_args=catalog_model.client_args,
                        params={**(catalog_model.default_params or {}), **model_settings}
                    )
                else:
                    model_config = ModelConfig(
                        provider=agent_data.get('provider', 'litellm'),
                        model_id=agent_data.get('model', 'gpt-4'),
                        temperature=model_settings.get('temperature', 0.7),
                        max_tokens=model_settings.get('max_tokens', 4096)
                    )
                
                agent_config = AgentConfig(
                    name=agent_data.get('name', agent_name),
                    description=agent_data.get('description', f'Agent created for workflow {workflow_config.name}'),
                    agent_id=f"{agent_name}-{uuid.uuid4().hex[:8]}",
                    model=model_config,
                    system_prompt=agent_data.get('system_prompt', 'You are a helpful AI assistant.'),
                    conversation_manager=ConversationManagerConfig(),
                    tools=[
                        ToolConfig(name=tool_name, enabled=True, type="built-in")
                        for tool_name in agent_data.get('tools', [])
                    ],
                    hooks=[],
                    state={}
                )
                
                if model_ref:
                    success = config_service.save_agent_config_with_model_ref(
                        agent_name, agent_config, model_ref, model_settings
                    )
                else:
                    success = config_service.save_agent_config(agent_name, agent_config)
                
                if success:
                    created_agents.append(agent_name)
        
        # Update workflow fields if provided
        if request.name:
            workflow_config.name = request.name
        if request.description is not None:
            workflow_config.description = request.description
        if request.topology:
            workflow_config.topology = request.topology
        if request.tags is not None:
            workflow_config.tags = request.tags
        
        # Update nodes if provided
        if request.nodes:
            from app.models.workflow_models import WorkflowNode, NodeType
            workflow_nodes = []
            for node_data in request.nodes:
                node = WorkflowNode(
                    node_id=node_data.get('id', node_data.get('node_id')),
                    name=node_data.get('name', 'Unnamed Node'),
                    type=NodeType.AGENT if node_data.get('type') == 'agent' else NodeType.CONDITION,
                    agent_config=node_data.get('agent_config'),
                    system_prompt=node_data.get('system_prompt'),
                    condition_expression=node_data.get('condition_expression'),
                    timeout=node_data.get('timeout', 300),
                    retry_count=node_data.get('retry_count', 0)
                )
                workflow_nodes.append(node)
            workflow_config.nodes = workflow_nodes
        
        # Update edges if provided
        if request.edges is not None:
            from app.models.workflow_models import WorkflowEdge
            workflow_edges = []
            for edge_data in request.edges:
                edge = WorkflowEdge(
                    from_node=edge_data.get('from_node', edge_data.get('source')),
                    to_node=edge_data.get('to_node', edge_data.get('target')),
                    condition=edge_data.get('condition')
                )
                workflow_edges.append(edge)
            workflow_config.edges = workflow_edges
        
        workflow_config.updated_at = datetime.now()
        
        # Save updated workflow
        workflow_service.save_workflow_config(workflow_config)
        
        return {
            "message": f"Workflow '{workflow_id}' updated successfully",
            "created_agents": created_agents
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating workflow: {str(e)}")


# Add import for uuid at the top of the file
import uuid
