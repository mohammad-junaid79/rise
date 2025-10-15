import asyncio
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from strands import Agent
from strands.multiagent import GraphBuilder, Swarm
from strands.multiagent.base import MultiAgentBase

from app.models.workflow_models import (
    WorkflowConfig, WorkflowNode, WorkflowEdge, NodeResult, 
    WorkflowExecutionResult, WorkflowExecutionRequest, ExecutionStatus,
    WorkflowTopology, NodeType, CommunicationPattern
)
from app.services.config_service import ConfigService
from app.services.custom_tool_service import CustomToolService
from app.services.agent_service import StrandsAgentService

logger = logging.getLogger(__name__)


class WorkflowOrchestrationService:
    def __init__(self):
        self.config_service = ConfigService()
        self.custom_tool_service = CustomToolService()
        # Load custom tools to make them available for function nodes
        self.custom_tool_service.load_custom_tools()
        self.agent_service = StrandsAgentService(self.config_service)
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.active_executions: Dict[str, Dict[str, Any]] = {}
        self.execution_results: Dict[str, WorkflowExecutionResult] = {}
        self.load_workflows()
    
    def load_workflows(self):
        """Load workflow configurations from YAML files"""
        workflows_dir = Path("configs/workflows")
        if not workflows_dir.exists():
            workflows_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created workflows directory: {workflows_dir}")
            return
        
        for yaml_file in workflows_dir.glob("*.yaml"):
            try:
                with open(yaml_file, 'r') as f:
                    workflow_data = yaml.safe_load(f)
                    workflow_config = WorkflowConfig(**workflow_data)
                    self.workflows[workflow_config.workflow_id] = workflow_config
                    logger.info(f"Loaded workflow: {workflow_config.workflow_id}")
            except Exception as e:
                logger.error(f"Error loading workflow from {yaml_file}: {e}")
    
    def get_workflow(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """Get workflow configuration by ID"""
        return self.workflows.get(workflow_id)
    
    def get_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """Get workflow configuration by ID (alias for get_workflow)"""
        return self.get_workflow(workflow_id)
    
    def list_workflows(self) -> List[WorkflowConfig]:
        """List all available workflows"""
        return list(self.workflows.values())
    
    def save_workflow_config(self, workflow_config: WorkflowConfig) -> bool:
        """Save workflow configuration to YAML file"""
        try:
            workflows_dir = Path("configs/workflows")
            workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Use workflow name (converted to safe filename) instead of workflow_id
            workflow_name = workflow_config.name or "untitled_workflow"
            safe_name = workflow_name.lower().replace(' ', '_').replace('-', '_')
            # Remove any characters that aren't alphanumeric or underscore
            safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')
            config_file = workflows_dir / f"{safe_name}.yaml"
            
            # Convert workflow config to dict with proper YAML serialization
            config_dict = {
                "workflow_id": workflow_config.workflow_id,
                "name": workflow_config.name,
                "description": workflow_config.description,
                "version": workflow_config.version,
                "topology": workflow_config.topology.value if hasattr(workflow_config.topology, 'value') else str(workflow_config.topology),
                "nodes": [],
                "edges": workflow_config.edges if workflow_config.edges else [],
                "entry_points": workflow_config.entry_points,
                "communication": workflow_config.communication if workflow_config.communication else [],
                "context": {
                    "shared_keys": workflow_config.context.shared_keys if workflow_config.context else [],
                    "node_specific_keys": workflow_config.context.node_specific_keys if workflow_config.context else {},
                    "persistence": workflow_config.context.persistence if workflow_config.context else True,
                    "isolation_level": workflow_config.context.isolation_level.value if workflow_config.context and hasattr(workflow_config.context.isolation_level, 'value') else "workflow"
                },
                "parallel_config": workflow_config.parallel_config,
                "max_execution_time": workflow_config.max_execution_time,
                "error_handling": workflow_config.error_handling.value if hasattr(workflow_config.error_handling, 'value') else str(workflow_config.error_handling),
                "tags": workflow_config.tags if workflow_config.tags else [],
                "created_at": workflow_config.created_at,
                "updated_at": workflow_config.updated_at
            }
            
            # Convert nodes with proper enum handling
            for node in workflow_config.nodes:
                node_dict = {
                    "node_id": node.node_id,
                    "name": node.name,
                    "type": node.type.value if hasattr(node.type, 'value') else str(node.type),
                    "agent_config": node.agent_config,
                    "system_prompt": node.system_prompt,
                    "function_name": node.function_name,
                    "function_params": node.function_params,
                    "condition_expression": node.condition_expression,
                    "timeout": node.timeout,
                    "retry_count": node.retry_count,
                    "priority": node.priority,
                    "input_mapping": node.input_mapping,
                    "output_mapping": node.output_mapping,
                    "context_keys": node.context_keys
                }
                config_dict["nodes"].append(node_dict)
            
            # Save to YAML file
            with open(config_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            # Update in-memory cache
            self.workflows[workflow_config.workflow_id] = workflow_config
            
            logger.info(f"Saved workflow configuration: {workflow_config.workflow_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save workflow config {workflow_config.workflow_id}: {str(e)}")
            return False
    
    async def execute_workflow_stream(self, request: WorkflowExecutionRequest):
        """Execute a workflow with streaming output"""
        workflow = self.get_workflow(request.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {request.workflow_id}")
        
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Initialize execution context
        execution_context = {
            "workflow_id": request.workflow_id,
            "execution_id": execution_id,
            "input_data": request.input_data,
            "context": request.context or {},
            "start_time": start_time,
            "shared_context": workflow.context.shared_keys,
            "node_contexts": {}
        }
        
        self.active_executions[execution_id] = execution_context
        
        try:
            # Stream execution based on topology
            if workflow.topology == WorkflowTopology.SEQUENTIAL:
                async for event in self._execute_sequential_stream(workflow, execution_context):
                    yield event
            elif workflow.topology == WorkflowTopology.PARALLEL:
                async for event in self._execute_parallel_stream(workflow, execution_context):
                    yield event
            else:
                # For other topologies, fall back to regular execution and stream the result
                result = await self.execute_workflow(request)
                event_data = {
                    "execution_id": execution_id,
                    "workflow_id": request.workflow_id,
                    "status": result.status.value,
                    "result": result.dict(),
                    "timestamp": datetime.now().isoformat()
                }
                yield {"event": "workflow_completed", "data": event_data}
        except Exception as e:
            logger.error(f"Workflow streaming failed: {e}")
            yield {
                "event": "workflow_failed",
                "data": {
                    "execution_id": execution_id,
                    "workflow_id": request.workflow_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            }
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]

    async def execute_workflow(self, request: WorkflowExecutionRequest) -> WorkflowExecutionResult:
        """Execute a workflow based on its topology"""
        workflow = self.get_workflow(request.workflow_id)
        if not workflow:
            raise ValueError(f"Workflow not found: {request.workflow_id}")
        
        execution_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Initialize execution context
        execution_context = {
            "workflow_id": request.workflow_id,
            "execution_id": execution_id,
            "input_data": request.input_data,
            "context": request.context or {},
            "start_time": start_time,
            "shared_context": workflow.context.shared_keys,
            "node_contexts": {}
        }
        
        self.active_executions[execution_id] = execution_context
        
        try:
            # Execute based on topology
            if workflow.topology == WorkflowTopology.SEQUENTIAL:
                result = await self._execute_sequential(workflow, execution_context)
            elif workflow.topology == WorkflowTopology.PARALLEL:
                result = await self._execute_parallel(workflow, execution_context)
            elif workflow.topology == WorkflowTopology.GRAPH:
                result = await self._execute_graph(workflow, execution_context)
            elif workflow.topology == WorkflowTopology.SWARM:
                result = await self._execute_swarm(workflow, execution_context)
            elif workflow.topology == WorkflowTopology.MESH:
                result = await self._execute_mesh(workflow, execution_context)
            else:
                raise ValueError(f"Unsupported topology: {workflow.topology}")
            
            result.execution_id = execution_id
            result.workflow_id = request.workflow_id
            result.start_time = start_time
            result.end_time = datetime.now()
            result.total_execution_time = (result.end_time - result.start_time).total_seconds()
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return WorkflowExecutionResult(
                workflow_id=request.workflow_id,
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                errors=[str(e)]
            )
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    def _extract_response_text(self, result) -> str:
        """Extract response text from agent result safely"""
        if hasattr(result, 'message') and hasattr(result.message, 'content'):
            if isinstance(result.message.content, list) and len(result.message.content) > 0:
                return result.message.content[0].text
            elif isinstance(result.message.content, str):
                return result.message.content
        return str(result) if result else "No response generated"
    
    async def _execute_sequential_stream(self, workflow: WorkflowConfig, execution_context: Dict[str, Any]):
        """Execute workflow in sequential order with streaming"""
        logger.info(f"Executing sequential workflow with streaming: {workflow.workflow_id}")
        
        node_results = []
        execution_order = []
        shared_context = execution_context["context"].copy()
        
        # Sort nodes by priority if specified, otherwise use order in config
        sorted_nodes = sorted(workflow.nodes, key=lambda n: n.priority or 0, reverse=True)
        
        # Stream workflow metadata
        metadata = {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "topology": workflow.topology.value,
            "total_nodes": len(sorted_nodes),
            "node_names": [node.name for node in sorted_nodes],
            "timestamp": datetime.now().isoformat()
        }
        yield {"event": "workflow_metadata", "data": metadata}
        
        for i, node in enumerate(sorted_nodes):
            try:
                logger.info(f"Executing node: {node.node_id}")
                
                # Stream node started event
                yield {
                    "event": "node_started",
                    "data": {
                        "node_id": node.node_id,
                        "node_name": node.name,
                        "node_type": node.type.value,
                        "progress": f"{i}/{len(sorted_nodes)}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Prepare node input
                node_input = self._prepare_node_input(node, execution_context, shared_context, node_results)
                
                # Execute node with streaming and collect the result
                node_result = None
                async for node_event in self._execute_node_stream(node, node_input, shared_context):
                    yield node_event
                    # Capture the final result from the stream
                    if node_event.get("event") == "node_result":
                        node_result_data = node_event.get("data", {}).get("result", {})
                        execution_time = node_result_data.get("execution_time", 0.0)
                        node_result = NodeResult(
                            node_id=node.node_id,
                            name=node.name,
                            status=ExecutionStatus.COMPLETED if not node_result_data.get("error") else ExecutionStatus.FAILED,
                            execution_time=execution_time,
                            result=node_result_data,
                            error=node_result_data.get("message") if node_result_data.get("error") else None,
                            retry_count=0
                        )
                
                # If we didn't get a result from streaming, create a failed result
                if node_result is None:
                    node_result = NodeResult(
                        node_id=node.node_id,
                        name=node.name,
                        status=ExecutionStatus.FAILED,
                        execution_time=0.0,
                        result={"message": "Node execution failed - no result received"},
                        error="Node execution failed - no result received",
                        retry_count=0
                    )
                
                node_results.append(node_result)
                execution_order.append(node.node_id)
                
                # Stream node completed event
                yield {
                    "event": "node_completed",
                    "data": {
                        "node_id": node.node_id,
                        "node_name": node.name,
                        "status": node_result.status.value,
                        "execution_time": node_result.execution_time,
                        "progress": f"{i+1}/{len(sorted_nodes)}",
                        "result_preview": self._get_result_preview(node_result),
                        "response": node_result.result.get("message", "") if node_result.result and "message" in node_result.result else "",
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Update shared context
                if node_result.context_updates:
                    shared_context.update(node_result.context_updates)
                
                # Handle failure
                if node_result.status == ExecutionStatus.FAILED:
                    error_message = node_result.error or "Unknown error"
                    error_type = self._classify_error(error_message)
                    
                    yield {
                        "event": "node_failed",
                        "data": {
                            "node_id": node.node_id,
                            "node_name": node.name,
                            "error": error_message,
                            "error_type": error_type,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    # Stop execution for critical errors regardless of error_handling setting
                    if error_type in ['billing', 'auth']:
                        logger.error(f"Critical error encountered ({error_type}), stopping workflow execution")
                        yield {
                            "event": "workflow_stopped",
                            "data": {
                                "reason": f"Critical {error_type} error",
                                "error": error_message,
                                "timestamp": datetime.now().isoformat()
                            }
                        }
                        break
                    elif workflow.error_handling == "stop":
                        break
                
            except Exception as e:
                logger.error(f"Node execution failed: {node.node_id}, error: {e}")
                
                error_message = str(e)
                error_type = self._classify_error(error_message)
                
                # Stream error event
                yield {
                    "event": "node_error",
                    "data": {
                        "node_id": node.node_id,
                        "node_name": node.name,
                        "error": error_message,
                        "error_type": error_type,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                node_results.append(NodeResult(
                    node_id=node.node_id,
                    name=node.name,
                    status=ExecutionStatus.FAILED,
                    error=error_message
                ))
                
                # Stop execution for critical errors regardless of error_handling setting
                if error_type in ['billing', 'auth']:
                    logger.error(f"Critical error encountered ({error_type}), stopping workflow execution")
                    yield {
                        "event": "workflow_stopped",
                        "data": {
                            "reason": f"Critical {error_type} error",
                            "error": error_message,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    break
                elif workflow.error_handling == "stop":
                    break
        
        # Calculate final metrics
        completed_nodes = [r for r in node_results if r.status == ExecutionStatus.COMPLETED]
        failed_nodes = [r for r in node_results if r.status == ExecutionStatus.FAILED]
        
        # Stream final result
        final_result = WorkflowExecutionResult(
            workflow_id=workflow.workflow_id,
            execution_id=execution_context["execution_id"],
            status=ExecutionStatus.COMPLETED if not failed_nodes else ExecutionStatus.FAILED,
            node_results=node_results,
            execution_order=execution_order,
            final_context=shared_context,
            nodes_executed=len(completed_nodes),
            nodes_failed=len(failed_nodes)
        )
        
        yield {
            "event": "workflow_completed",
            "data": {
                "workflow_id": workflow.workflow_id,
                "execution_id": execution_context["execution_id"],
                "status": final_result.status.value,
                "nodes_executed": len(completed_nodes),
                "nodes_failed": len(failed_nodes),
                "execution_order": execution_order,
                "timestamp": datetime.now().isoformat()
            }
        }

    async def _execute_sequential(self, workflow: WorkflowConfig, execution_context: Dict[str, Any]) -> WorkflowExecutionResult:
        """Execute workflow in sequential order"""
        logger.info(f"Executing sequential workflow: {workflow.workflow_id}")
        
        node_results = []
        execution_order = []
        shared_context = execution_context["context"].copy()
        
        # Sort nodes by priority if specified, otherwise use order in config
        sorted_nodes = sorted(workflow.nodes, key=lambda n: n.priority or 0, reverse=True)
        
        for node in sorted_nodes:
            try:
                logger.info(f"Executing node: {node.node_id}")
                
                # Prepare node input
                node_input = self._prepare_node_input(node, execution_context, shared_context, node_results)
                
                # Execute node
                node_result = await self._execute_node(node, node_input, shared_context)
                node_results.append(node_result)
                execution_order.append(node.node_id)
                
                # Update shared context
                if node_result.context_updates:
                    shared_context.update(node_result.context_updates)
                
                # Handle failure
                if node_result.status == ExecutionStatus.FAILED:
                    if workflow.error_handling == "stop":
                        break
                    elif workflow.error_handling == "retry" and node_result.retry_count < (node.retry_count or 0):
                        # Retry logic would go here
                        pass
                
            except Exception as e:
                logger.error(f"Node execution failed: {node.node_id}, error: {e}")
                node_results.append(NodeResult(
                    node_id=node.node_id,
                    name=node.name,
                    status=ExecutionStatus.FAILED,
                    error=str(e)
                ))
                if workflow.error_handling == "stop":
                    break
        
        # Calculate metrics
        completed_nodes = [r for r in node_results if r.status == ExecutionStatus.COMPLETED]
        failed_nodes = [r for r in node_results if r.status == ExecutionStatus.FAILED]
        
        return WorkflowExecutionResult(
            workflow_id=workflow.workflow_id,
            execution_id=execution_context["execution_id"],
            status=ExecutionStatus.COMPLETED if not failed_nodes else ExecutionStatus.FAILED,
            node_results=node_results,
            execution_order=execution_order,
            final_context=shared_context,
            nodes_executed=len(completed_nodes),
            nodes_failed=len(failed_nodes)
        )
    
    async def _execute_parallel(self, workflow: WorkflowConfig, execution_context: Dict[str, Any]) -> WorkflowExecutionResult:
        """Execute workflow with parallel execution where possible"""
        logger.info(f"Executing parallel workflow: {workflow.workflow_id}")
        
        node_results = []
        execution_order = []
        shared_context = execution_context["context"].copy()
        
        # Group nodes by dependencies for parallel execution
        dependency_groups = self._analyze_dependencies(workflow)
        
        for group_nodes in dependency_groups:
            # Execute all nodes in this group in parallel
            tasks = []
            for node in group_nodes:
                node_input = self._prepare_node_input(node, execution_context, shared_context, node_results)
                task = asyncio.create_task(self._execute_node(node, node_input, shared_context))
                tasks.append((node, task))
            
            # Wait for all nodes in this group to complete
            for node, task in tasks:
                try:
                    node_result = await task
                    node_results.append(node_result)
                    execution_order.append(node.node_id)
                    
                    # Update shared context
                    if node_result.context_updates:
                        shared_context.update(node_result.context_updates)
                        
                except Exception as e:
                    logger.error(f"Parallel node execution failed: {node.node_id}, error: {e}")
                    node_results.append(NodeResult(
                        node_id=node.node_id,
                        name=node.name,
                        status=ExecutionStatus.FAILED,
                        error=str(e)
                    ))
        
        completed_nodes = [r for r in node_results if r.status == ExecutionStatus.COMPLETED]
        failed_nodes = [r for r in node_results if r.status == ExecutionStatus.FAILED]
        
        return WorkflowExecutionResult(
            workflow_id=workflow.workflow_id,
            execution_id=execution_context["execution_id"],
            status=ExecutionStatus.COMPLETED if not failed_nodes else ExecutionStatus.FAILED,
            node_results=node_results,
            execution_order=execution_order,
            final_context=shared_context,
            nodes_executed=len(completed_nodes),
            nodes_failed=len(failed_nodes)
        )
    
    async def _execute_graph(self, workflow: WorkflowConfig, execution_context: Dict[str, Any]) -> WorkflowExecutionResult:
        """Execute workflow using enhanced Graph pattern with support for hybrid topologies"""
        logger.info(f"Executing hybrid graph workflow: {workflow.workflow_id}")
        
        # For complex hybrid workflows, we'll implement custom execution logic
        # that supports conditional routing, parallel groups, and sequential chains
        
        node_results = []
        execution_order = []
        shared_context = execution_context["context"].copy()
        
        # Track node status and dependencies
        node_status = {node.node_id: "pending" for node in workflow.nodes}
        node_dependencies = self._analyze_node_dependencies(workflow)
        conditional_nodes = self._identify_conditional_nodes(workflow)
        
        # Initialize execution tracking
        executed_nodes = set()
        failed_nodes = set()
        skipped_nodes = set()
        
        # Main execution loop
        while True:
            # Find nodes that are ready to execute
            ready_nodes = self._find_ready_nodes(
                workflow.nodes, 
                node_status, 
                node_dependencies, 
                shared_context,
                conditional_nodes
            )
            
            if not ready_nodes:
                # Check if we're done or stuck
                pending_nodes = [n for n, status in node_status.items() if status == "pending"]
                if not pending_nodes:
                    break  # All nodes completed
                else:
                    # Handle stuck nodes (dependencies not met)
                    logger.warning(f"Workflow stuck with pending nodes: {pending_nodes}")
                    for node_id in pending_nodes:
                        node_status[node_id] = "skipped"
                        skipped_nodes.add(node_id)
                    break
            
            # Group ready nodes by execution pattern
            execution_groups = self._group_nodes_by_execution_pattern(ready_nodes, workflow)
            
            # Execute each group according to its pattern
            for group_type, group_nodes in execution_groups.items():
                if group_type == "parallel":
                    await self._execute_parallel_group(
                        group_nodes, execution_context, shared_context, 
                        node_results, execution_order, node_status
                    )
                elif group_type == "sequential":
                    await self._execute_sequential_group(
                        group_nodes, execution_context, shared_context,
                        node_results, execution_order, node_status
                    )
                else:  # individual nodes
                    for node in group_nodes:
                        await self._execute_single_node_in_graph(
                            node, execution_context, shared_context,
                            node_results, execution_order, node_status
                        )
                
                # Update executed nodes
                for node in group_nodes:
                    executed_nodes.add(node.node_id)
        
        # Calculate final metrics
        completed_nodes = [r for r in node_results if r.status == ExecutionStatus.COMPLETED]
        failed_node_results = [r for r in node_results if r.status == ExecutionStatus.FAILED]
        
        final_status = ExecutionStatus.COMPLETED
        if failed_node_results:
            final_status = ExecutionStatus.FAILED if workflow.error_handling == "stop" else ExecutionStatus.COMPLETED
        
        return WorkflowExecutionResult(
            workflow_id=workflow.workflow_id,
            execution_id=execution_context["execution_id"],
            status=final_status,
            node_results=node_results,
            execution_order=execution_order,
            final_context=shared_context,
            nodes_executed=len(completed_nodes),
            nodes_failed=len(failed_node_results),
            nodes_skipped=len(skipped_nodes)
        )
    
    async def _execute_swarm(self, workflow: WorkflowConfig, execution_context: Dict[str, Any]) -> WorkflowExecutionResult:
        """Execute workflow using Strands Swarm pattern"""
        logger.info(f"Executing swarm workflow: {workflow.workflow_id}")
        
        # Create agents for swarm
        agents = []
        for node in workflow.nodes:
            if node.type == NodeType.AGENT:
                agent = self._create_agent_for_node(node)
                agents.append(agent)
        
        # Create swarm
        swarm = Swarm(
            agents,
            max_handoffs=20,
            max_iterations=20,
            execution_timeout=workflow.max_execution_time or 900.0,
            node_timeout=300.0
        )
        
        # Execute swarm
        task_input = execution_context["input_data"].get("task", "Execute workflow")
        swarm_result = await swarm.invoke_async(task_input)
        
        # Convert swarm result to workflow result
        node_results = []
        execution_order = []
        
        for node in swarm_result.node_history:
            node_result = NodeResult(
                node_id=node.node_id,
                name=node.node_id,
                status=ExecutionStatus.COMPLETED,
                result={"message": self._extract_response_text(node.result) if node.result else None},
                execution_time=getattr(node, 'execution_time', None)
            )
            node_results.append(node_result)
            execution_order.append(node.node_id)
        
        return WorkflowExecutionResult(
            workflow_id=workflow.workflow_id,
            execution_id=execution_context["execution_id"],
            status=ExecutionStatus.COMPLETED if swarm_result.status.name == "COMPLETED" else ExecutionStatus.FAILED,
            node_results=node_results,
            execution_order=execution_order,
            final_context=execution_context["context"],
            nodes_executed=len([r for r in node_results if r.status == ExecutionStatus.COMPLETED]),
            nodes_failed=len([r for r in node_results if r.status == ExecutionStatus.FAILED])
        )
    
    async def _execute_mesh(self, workflow: WorkflowConfig, execution_context: Dict[str, Any]) -> WorkflowExecutionResult:
        """Execute workflow with mesh topology (full interconnectivity)"""
        logger.info(f"Executing mesh workflow: {workflow.workflow_id}")
        
        # For mesh topology, all agents can communicate with each other
        # This is a simplified implementation - in practice, you'd implement
        # more sophisticated mesh communication patterns
        
        agents = {}
        for node in workflow.nodes:
            if node.type == NodeType.AGENT:
                agents[node.node_id] = self._create_agent_for_node(node)
        
        # Execute with mesh communication pattern
        node_results = []
        execution_order = []
        shared_context = execution_context["context"].copy()
        
        # Simple round-robin execution with context sharing
        for node in workflow.nodes:
            if node.type == NodeType.AGENT:
                node_input = self._prepare_node_input(node, execution_context, shared_context, node_results)
                node_result = await self._execute_node(node, node_input, shared_context)
                node_results.append(node_result)
                execution_order.append(node.node_id)
                
                if node_result.context_updates:
                    shared_context.update(node_result.context_updates)
        
        completed_nodes = [r for r in node_results if r.status == ExecutionStatus.COMPLETED]
        failed_nodes = [r for r in node_results if r.status == ExecutionStatus.FAILED]
        
        return WorkflowExecutionResult(
            workflow_id=workflow.workflow_id,
            execution_id=execution_context["execution_id"],
            status=ExecutionStatus.COMPLETED if not failed_nodes else ExecutionStatus.FAILED,
            node_results=node_results,
            execution_order=execution_order,
            final_context=shared_context,
            nodes_executed=len(completed_nodes),
            nodes_failed=len(failed_nodes)
        )
    
    def _create_agent_for_node(self, node: WorkflowNode) -> Agent:
        """Create an agent instance for a workflow node using AgentService"""
        if node.agent_config:
            # Use existing agent configuration through AgentService
            agent_config = self.config_service.get_agent_config(node.agent_config)
            if not agent_config:
                raise ValueError(f"Agent config not found: {node.agent_config}")
            
            # Create agent using AgentService methods
            model = self.agent_service._create_model(agent_config)
            conversation_manager = self.agent_service._create_conversation_manager(agent_config)
            tools = self.agent_service._create_tools(agent_config)
            hooks = self.agent_service._create_hooks(agent_config.hooks)
            
            # Override system prompt if specified in the node
            system_prompt = node.system_prompt or agent_config.system_prompt
            
            # Create the agent
            return Agent(
                model=model,
                system_prompt=system_prompt,
                conversation_manager=conversation_manager,
                tools=tools,
                hooks=hooks
            )
        else:
            # Create a basic agent configuration for nodes without explicit agent_config
            from app.models.config_models import AgentConfig, ModelConfig, ConversationManagerConfig
            
            # Create a basic model config using Gemini
            model_config = ModelConfig(
                provider="litellm",
                model_id="gemini/gemini-1.5-flash",
                temperature=0.7,
                max_tokens=4000
            )
            
            # Create a basic agent config
            basic_config = AgentConfig(
                name=node.name,
                description=f"{node.name} - A specialized workflow agent",
                agent_id=f"{node.node_id}_agent",
                system_prompt=node.system_prompt or f"You are {node.name}, a specialized agent.",
                model=model_config,
                tools=[],
                hooks=[],
                conversation_manager=ConversationManagerConfig(
                    type="sliding_window",
                    window_size=10,
                    should_truncate_results=True
                )
            )
            
            # Create agent using AgentService methods
            model = self.agent_service._create_model(basic_config)
            conversation_manager = self.agent_service._create_conversation_manager(basic_config)
            tools = self.agent_service._create_tools(basic_config)
            hooks = self.agent_service._create_hooks(basic_config.hooks)
            
            return Agent(
                model=model,
                system_prompt=basic_config.system_prompt,
                conversation_manager=conversation_manager,
                tools=tools,
                hooks=hooks
            )
    
    async def _execute_node_stream(self, node: WorkflowNode, node_input: Dict[str, Any], shared_context: Dict[str, Any]):
        """Stream node execution events"""
        if node.type == NodeType.AGENT:
            async for event in self._execute_agent_node_stream(node, node_input, shared_context):
                yield event
        elif node.type == NodeType.FUNCTION:
            # Handle function nodes with streaming
            try:
                yield {
                    "event": "function_executing",
                    "data": {
                        "node_id": node.node_id,
                        "function_name": node.function_name,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                # Execute function node (non-streaming)
                result = await self._execute_function_node(node, node_input, shared_context)
                
                yield {
                    "event": "function_completed",
                    "data": {
                        "node_id": node.node_id,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                yield {
                    "event": "node_result",
                    "data": {
                        "node_id": node.node_id,
                        "result": result,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
            except Exception as e:
                logger.error(f"Function node execution failed: {node.node_id}, error: {e}")
                yield {
                    "event": "function_error",
                    "data": {
                        "node_id": node.node_id,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                }
                
                yield {
                    "event": "node_result",
                    "data": {
                        "node_id": node.node_id,
                        "result": {
                            "message": f"❌ Function execution failed: {str(e)}",
                            "error": True
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                }
        else:
            # Handle unknown node types gracefully
            logger.warning(f"Unknown node type: {node.type} for node {node.node_id}")
            yield {
                "event": "node_error",
                "data": {
                    "node_id": node.node_id,
                    "error": f"Unsupported node type: {node.type}",
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            yield {
                "event": "node_result",
                "data": {
                    "node_id": node.node_id,
                    "result": {
                        "message": f"❌ Unsupported node type: {node.type}",
                        "error": True
                    },
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    async def _execute_agent_node_stream(self, node: WorkflowNode, node_input: Dict[str, Any], shared_context: Dict[str, Any]):
        """Stream agent node execution events"""
        # Stream agent preparation
        yield {
            "event": "agent_preparing",
            "data": {
                "node_id": node.node_id,
                "agent_config": node.agent_config,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Prepare prompt with context
        # Handle plain text input_data or structured data
        user_message = None
        
        if "input_data" in node_input:
            input_data = node_input["input_data"]
            
            # If input_data is a string (plain text), use it directly
            if isinstance(input_data, str):
                user_message = input_data.strip()
            # If it's a dict, try to extract meaningful content
            elif isinstance(input_data, dict):
                # Try different field names for user input
                user_message = (
                    input_data.get("message") or 
                    input_data.get("topic") or 
                    input_data.get("query") or 
                    input_data.get("task")
                )
                
                # If no direct message field, construct a detailed prompt from all input data
                if not user_message and input_data:
                    parts = []
                    if "topic" in input_data:
                        parts.append(f"Topic: {input_data['topic']}")
                    if "initial_context" in input_data:
                        context = input_data["initial_context"]
                        if isinstance(context, dict):
                            context_parts = [f"{k}: {v}" for k, v in context.items()]
                            parts.append(f"Context: {', '.join(context_parts)}")
                        else:
                            parts.append(f"Context: {context}")
                    
                    # Add any other fields from input_data
                    for key, value in input_data.items():
                        if key not in ["topic", "initial_context", "message", "query", "task"]:
                            parts.append(f"{key.replace('_', ' ').title()}: {value}")
                    
                    if parts:
                        user_message = "Please process the following request:\n\n" + "\n".join(parts)

        if user_message:
            prompt = user_message
        else:
            prompt = node_input.get("prompt", "Please process the given task.")
            
        if "previous_results" in node_input:
            prompt += f"\n\nPrevious results:\n{node_input['previous_results']}"
        
        # Stream prompt
        yield {
            "event": "agent_prompt",
            "data": {
                "node_id": node.node_id,
                "prompt": prompt[:200] + "..." if len(prompt) > 200 else prompt,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Stream agent execution start
        yield {
            "event": "agent_executing",
            "data": {
                "node_id": node.node_id,
                "status": "executing",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            # Create and execute agent
            agent = self._create_agent_for_node(node)
            
            logger.info(f"Executing agent for node {node.node_id} with prompt: {prompt[:100]}...")
            
            # Track execution time
            execution_start = datetime.now()
            
            # Execute agent
            result = await agent.invoke_async(prompt)
            
            # Calculate execution time
            execution_end = datetime.now()
            execution_time = (execution_end - execution_start).total_seconds()
            
            # Extract response text properly
            response_text = self._extract_response_text(result)
            
            logger.info(f"Agent execution completed for node {node.node_id}. Response: {response_text[:100]}...")
            
            # Stream agent completion
            yield {
                "event": "agent_completed",
                "data": {
                    "node_id": node.node_id,
                    "status": "completed",
                    "response": response_text,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"Agent execution completed for node {node.node_id}. Response: {response_text[:100]}... (took {execution_time:.2f}s)")
            
            # Stream final result
            yield {
                "event": "node_result",
                "data": {
                    "node_id": node.node_id,
                    "result": {
                        "message": response_text,
                        "context_updates": node_input.get("context_updates", {}),
                        "execution_time": execution_time
                    },
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Agent execution failed for node {node.node_id}: {str(e)}")
            
            # Parse error message to provide user-friendly messages
            error_message = str(e)
            user_friendly_message = self._get_user_friendly_error_message(error_message)
            
            # Stream error
            yield {
                "event": "agent_error",
                "data": {
                    "node_id": node.node_id,
                    "status": "error",
                    "error": user_friendly_message,
                    "original_error": error_message,
                    "error_type": self._classify_error(error_message),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Stream fallback result
            yield {
                "event": "node_result",
                "data": {
                    "node_id": node.node_id,
                    "result": {
                        "message": f"❌ Agent execution failed: {user_friendly_message}",
                        "context_updates": {}
                    },
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def _get_user_friendly_error_message(self, error_message: str) -> str:
        """Convert technical error messages to user-friendly ones"""
        error_lower = error_message.lower()
        
        # Credit/billing issues
        if "credit balance is too low" in error_lower or "insufficient credits" in error_lower:
            return "API credits have been exhausted. Please check your billing settings and add more credits to continue."
        
        # Rate limiting
        if "rate limit" in error_lower or "too many requests" in error_lower:
            return "API rate limit exceeded. Please wait a moment before trying again."
        
        # Authentication issues
        if "authentication" in error_lower or "api key" in error_lower or "unauthorized" in error_lower:
            return "API authentication failed. Please check your API key configuration."
        
        # Model availability
        if "model not found" in error_lower or "model unavailable" in error_lower:
            return "The requested AI model is currently unavailable. Please try a different model."
        
        # Network/connection issues
        if "connection" in error_lower or "network" in error_lower or "timeout" in error_lower:
            return "Network connection issue. Please check your internet connection and try again."
        
        # Generic API errors
        if "badrequest" in error_lower or "400" in error_lower:
            return "Invalid request sent to AI service. Please check your configuration."
        
        if "500" in error_lower or "internal server error" in error_lower:
            return "AI service is experiencing internal issues. Please try again later."
        
        # Default fallback
        return f"AI service error: {error_message[:200]}{'...' if len(error_message) > 200 else ''}"
    
    def _classify_error(self, error_message: str) -> str:
        """Classify error type for better handling"""
        error_lower = error_message.lower()
        
        if "credit balance is too low" in error_lower or "insufficient credits" in error_lower:
            return "billing"
        elif "rate limit" in error_lower or "too many requests" in error_lower:
            return "rate_limit"
        elif "authentication" in error_lower or "api key" in error_lower:
            return "auth"
        elif "model not found" in error_lower or "model unavailable" in error_lower:
            return "model"
        elif "connection" in error_lower or "network" in error_lower or "timeout" in error_lower:
            return "network"
        else:
            return "unknown"
    
    def _get_result_preview(self, node_result: NodeResult) -> str:
        """Get a preview of the node result for streaming"""
        if node_result.result and "message" in node_result.result:
            message = node_result.result["message"]
            if len(message) > 150:
                return message[:150] + "..."
            return message
        return "No response generated"
    
    async def _execute_parallel_stream(self, workflow: WorkflowConfig, execution_context: Dict[str, Any]):
        """Execute workflow with parallel execution and streaming"""
        logger.info(f"Executing parallel workflow with streaming: {workflow.workflow_id}")
        
        # Stream workflow metadata
        yield {
            "event": "workflow_metadata",
            "data": {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "topology": workflow.topology.value,
                "total_nodes": len(workflow.nodes),
                "execution_type": "parallel",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # For parallel execution, we'll stream updates as nodes complete
        # This is a simplified version - in production, you might want more sophisticated parallel streaming
        node_results = []
        execution_order = []
        shared_context = execution_context["context"].copy()
        
        # Group nodes by dependencies for parallel execution
        dependency_groups = self._analyze_dependencies(workflow)
        
        for group_index, group_nodes in enumerate(dependency_groups):
            yield {
                "event": "parallel_group_started",
                "data": {
                    "group_index": group_index,
                    "group_size": len(group_nodes),
                    "node_ids": [node.node_id for node in group_nodes],
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Execute all nodes in this group in parallel
            tasks = []
            for node in group_nodes:
                node_input = self._prepare_node_input(node, execution_context, shared_context, node_results)
                task = asyncio.create_task(self._execute_node(node, node_input, shared_context))
                tasks.append((node, task))
            
            # Wait for all nodes in this group to complete and stream results
            for node, task in tasks:
                try:
                    node_result = await task
                    node_results.append(node_result)
                    execution_order.append(node.node_id)
                    
                    yield {
                        "event": "parallel_node_completed",
                        "data": {
                            "node_id": node.node_id,
                            "node_name": node.name,
                            "status": node_result.status.value,
                            "execution_time": node_result.execution_time,
                            "group_index": group_index,
                            "result_preview": self._get_result_preview(node_result),
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    
                    # Update shared context
                    if node_result.context_updates:
                        shared_context.update(node_result.context_updates)
                        
                except Exception as e:
                    logger.error(f"Parallel node execution failed: {node.node_id}, error: {e}")
                    yield {
                        "event": "parallel_node_failed",
                        "data": {
                            "node_id": node.node_id,
                            "node_name": node.name,
                            "error": str(e),
                            "group_index": group_index,
                            "timestamp": datetime.now().isoformat()
                        }
                    }
                    node_results.append(NodeResult(
                        node_id=node.node_id,
                        name=node.name,
                        status=ExecutionStatus.FAILED,
                        error=str(e)
                    ))
        
        # Calculate final metrics and stream completion
        completed_nodes = [r for r in node_results if r.status == ExecutionStatus.COMPLETED]
        failed_nodes = [r for r in node_results if r.status == ExecutionStatus.FAILED]
        
        yield {
            "event": "workflow_completed",
            "data": {
                "workflow_id": workflow.workflow_id,
                "execution_id": execution_context["execution_id"],
                "status": "completed" if not failed_nodes else "failed",
                "nodes_executed": len(completed_nodes),
                "nodes_failed": len(failed_nodes),
                "execution_order": execution_order,
                "timestamp": datetime.now().isoformat()
            }
        }

    async def _execute_node(self, node: WorkflowNode, node_input: Dict[str, Any], shared_context: Dict[str, Any]) -> NodeResult:
        """Execute a single workflow node"""
        start_time = datetime.now()
        
        try:
            if node.type == NodeType.AGENT:
                result = await self._execute_agent_node(node, node_input, shared_context)
            elif node.type == NodeType.FUNCTION:
                result = await self._execute_function_node(node, node_input, shared_context)
            elif node.type == NodeType.CONDITION:
                result = await self._execute_condition_node(node, node_input, shared_context)
            else:
                raise ValueError(f"Unsupported node type: {node.type}")
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return NodeResult(
                node_id=node.node_id,
                name=node.name,
                status=ExecutionStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time
            )
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            return NodeResult(
                node_id=node.node_id,
                name=node.name,
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time
            )
    
    async def _execute_agent_node(self, node: WorkflowNode, node_input: Dict[str, Any], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an agent node"""
        agent = self._create_agent_for_node(node)
        
        # Prepare prompt with context
        # Handle plain text input_data or structured data
        user_message = None
        
        if "input_data" in node_input:
            input_data = node_input["input_data"]
            
            # If input_data is a string (plain text), use it directly
            if isinstance(input_data, str):
                user_message = input_data.strip()
            # If it's a dict, try to extract meaningful content
            elif isinstance(input_data, dict):
                # Try different field names for user input
                user_message = (
                    input_data.get("message") or 
                    input_data.get("topic") or 
                    input_data.get("query") or 
                    input_data.get("task")
                )
                
                # If no direct message field, construct a detailed prompt from all input data
                if not user_message and input_data:
                    parts = []
                    if "topic" in input_data:
                        parts.append(f"Topic: {input_data['topic']}")
                    if "initial_context" in input_data:
                        context = input_data["initial_context"]
                        if isinstance(context, dict):
                            context_parts = [f"{k}: {v}" for k, v in context.items()]
                            parts.append(f"Context: {', '.join(context_parts)}")
                        else:
                            parts.append(f"Context: {context}")
                    
                    # Add any other fields from input_data
                    for key, value in input_data.items():
                        if key not in ["topic", "initial_context", "message", "query", "task"]:
                            parts.append(f"{key.replace('_', ' ').title()}: {value}")
                    
                    if parts:
                        user_message = "Please process the following request:\n\n" + "\n".join(parts)

        if user_message:
            prompt = user_message
        else:
            prompt = node_input.get("prompt", "Please process the given task.")
            
        if "previous_results" in node_input:
            prompt += f"\n\nPrevious results:\n{node_input['previous_results']}"
        
        try:
            # Execute agent
            result = await agent.invoke_async(prompt)
            
            # Extract response text properly
            response_text = self._extract_response_text(result)
            
            return {
                "message": response_text,
                "context_updates": node_input.get("context_updates", {})
            }
        except Exception as e:
            logger.error(f"Agent execution failed for node {node.node_id}: {str(e)}")
            user_friendly_message = self._get_user_friendly_error_message(str(e))
            
            return {
                "message": f"❌ Agent execution failed: {user_friendly_message}",
                "context_updates": {},
                "error": True,
                "error_type": self._classify_error(str(e))
            }
    
    async def _execute_function_node(self, node: WorkflowNode, node_input: Dict[str, Any], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function node"""
        if not node.function_name:
            raise ValueError(f"Function name not specified for node: {node.node_id}")
        
        # Get custom tool function
        tool_func = self.custom_tool_service.get_tool(node.function_name)
        if not tool_func:
            raise ValueError(f"Function not found: {node.function_name}")
        
        # Prepare function parameters
        params = node.function_params or {}
        params.update(node_input)
        
        # For route_workflow function, add the shared context
        if node.function_name == "route_workflow":
            params["context"] = shared_context
        
        try:
            # Execute function - call tool directly as a function
            result = tool_func(**params)
            return {"result": result}
        except Exception as e:
            logger.error(f"Function execution failed for {node.function_name}: {str(e)}")
            raise ValueError(f"Function execution failed: {str(e)}")
    
    async def _execute_condition_node(self, node: WorkflowNode, node_input: Dict[str, Any], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a condition node"""
        if not node.condition_expression:
            raise ValueError(f"Condition expression not specified for node: {node.node_id}")
        
        # Evaluate condition (simplified - in practice, use safe evaluation)
        context = {**shared_context, **node_input}
        try:
            result = eval(node.condition_expression, {"__builtins__": {}}, context)
            return {"condition_result": bool(result)}
        except Exception as e:
            raise ValueError(f"Condition evaluation failed: {e}")
    
    def _prepare_node_input(self, node: WorkflowNode, execution_context: Dict[str, Any], 
                          shared_context: Dict[str, Any], previous_results: List[NodeResult]) -> Dict[str, Any]:
        """Prepare input for node execution"""
        node_input = {
            "execution_id": execution_context["execution_id"],
            "workflow_id": execution_context["workflow_id"],
            "shared_context": shared_context,
            "input_data": execution_context["input_data"]
        }
        
        # Add previous results if available
        if previous_results:
            node_input["previous_results"] = "\n".join([
                f"{r.node_id}: {r.result}" for r in previous_results 
                if r.status == ExecutionStatus.COMPLETED and r.result
            ])
        
        # Apply input mapping
        if node.input_mapping:
            for target_key, source_key in node.input_mapping.items():
                if source_key in shared_context:
                    node_input[target_key] = shared_context[source_key]
        
        return node_input
    
    def _analyze_dependencies(self, workflow: WorkflowConfig) -> List[List[WorkflowNode]]:
        """Analyze node dependencies for parallel execution"""
        # Simple implementation - group nodes by dependency level
        # In practice, you'd implement more sophisticated dependency analysis
        
        if not workflow.edges:
            # No dependencies - all nodes can run in parallel
            return [workflow.nodes]
        
        # Build dependency graph
        dependencies = {node.node_id: set() for node in workflow.nodes}
        for edge in workflow.edges:
            dependencies[edge.to_node].add(edge.from_node)
        
        # Group nodes by dependency level
        groups = []
        remaining_nodes = {node.node_id: node for node in workflow.nodes}
        
        while remaining_nodes:
            # Find nodes with no dependencies
            ready_nodes = []
            for node_id, node in remaining_nodes.items():
                if not dependencies[node_id]:
                    ready_nodes.append(node)
            
            if not ready_nodes:
                # Circular dependency or other issue
                ready_nodes = list(remaining_nodes.values())
            
            groups.append(ready_nodes)
            
            # Remove ready nodes and update dependencies
            for node in ready_nodes:
                del remaining_nodes[node.node_id]
                for deps in dependencies.values():
                    deps.discard(node.node_id)
        
        return groups
    
    def _analyze_node_dependencies(self, workflow: WorkflowConfig) -> Dict[str, Set[str]]:
        """Analyze node dependencies from edges and dependency specifications"""
        dependencies = {node.node_id: set() for node in workflow.nodes}
        
        # Add dependencies from edges
        if workflow.edges:
            for edge in workflow.edges:
                dependencies[edge.to_node].add(edge.from_node)
        
        # Add explicit dependencies from node configuration
        for node in workflow.nodes:
            if hasattr(node, 'dependencies') and node.dependencies:
                for dep in node.dependencies:
                    dependencies[node.node_id].add(dep)
        
        return dependencies
    
    def _identify_conditional_nodes(self, workflow: WorkflowConfig) -> Dict[str, str]:
        """Identify nodes that have conditional execution requirements"""
        conditional_nodes = {}
        
        for node in workflow.nodes:
            if hasattr(node, 'conditional_execution') and node.conditional_execution:
                conditional_nodes[node.node_id] = node.conditional_execution
            
            # Check for conditional edges
            if workflow.edges:
                for edge in workflow.edges:
                    if edge.to_node == node.node_id and hasattr(edge, 'condition') and edge.condition:
                        conditional_nodes[node.node_id] = edge.condition
        
        return conditional_nodes
    
    def _find_ready_nodes(self, nodes: List[WorkflowNode], node_status: Dict[str, str], 
                         dependencies: Dict[str, Set[str]], shared_context: Dict[str, Any],
                         conditional_nodes: Dict[str, str]) -> List[WorkflowNode]:
        """Find nodes that are ready to execute based on dependencies and conditions"""
        ready_nodes = []
        
        for node in nodes:
            if node_status[node.node_id] != "pending":
                continue
            
            # Check if all dependencies are satisfied
            deps_satisfied = all(
                node_status.get(dep, "pending") == "completed" 
                for dep in dependencies[node.node_id]
            )
            
            if not deps_satisfied:
                continue
            
            # Check conditional execution requirements
            if node.node_id in conditional_nodes:
                condition = conditional_nodes[node.node_id]
                if not self._evaluate_node_condition(condition, shared_context):
                    node_status[node.node_id] = "skipped"
                    continue
            
            ready_nodes.append(node)
        
        return ready_nodes
    
    def _evaluate_node_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a node execution condition"""
        try:
            # Handle different condition formats
            if condition == "comprehensive":
                return context.get("analysis_route") == "comprehensive"
            elif condition == "alternative":
                return context.get("analysis_route") == "alternative"
            elif "context.get" in condition:
                # Safe evaluation of context-based conditions
                return eval(condition, {"__builtins__": {}}, {"context": context})
            else:
                # Simple string matching
                return context.get("analysis_route") == condition
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {condition}, error: {e}")
            return False
    
    def _group_nodes_by_execution_pattern(self, nodes: List[WorkflowNode], 
                                        workflow: WorkflowConfig) -> Dict[str, List[WorkflowNode]]:
        """Group nodes by their execution pattern (parallel, sequential, individual)"""
        groups = {"parallel": [], "sequential": [], "individual": []}
        
        # Group nodes by execution_group if specified
        execution_groups = {}
        for node in nodes:
            if hasattr(node, 'execution_group') and node.execution_group:
                if node.execution_group not in execution_groups:
                    execution_groups[node.execution_group] = []
                execution_groups[node.execution_group].append(node)
            else:
                groups["individual"].append(node)
        
        # Determine execution pattern for each group
        for group_name, group_nodes in execution_groups.items():
            # Check if group should execute in parallel (default for groups)
            if len(group_nodes) > 1:
                groups["parallel"].extend(group_nodes)
            else:
                groups["individual"].extend(group_nodes)
        
        return groups
    
    async def _execute_parallel_group(self, nodes: List[WorkflowNode], execution_context: Dict[str, Any],
                                    shared_context: Dict[str, Any], node_results: List[NodeResult],
                                    execution_order: List[str], node_status: Dict[str, str]):
        """Execute a group of nodes in parallel"""
        logger.info(f"Executing parallel group: {[n.node_id for n in nodes]}")
        
        # Create tasks for parallel execution
        tasks = []
        for node in nodes:
            node_input = self._prepare_node_input(node, execution_context, shared_context, node_results)
            task = asyncio.create_task(self._execute_node(node, node_input, shared_context))
            tasks.append((node, task))
        
        # Wait for all tasks to complete
        for node, task in tasks:
            try:
                node_result = await task
                node_results.append(node_result)
                execution_order.append(node.node_id)
                node_status[node.node_id] = "completed"
                
                # Update shared context
                if node_result.context_updates:
                    shared_context.update(node_result.context_updates)
                    
            except Exception as e:
                logger.error(f"Parallel node execution failed: {node.node_id}, error: {e}")
                node_result = NodeResult(
                    node_id=node.node_id,
                    name=node.name,
                    status=ExecutionStatus.FAILED,
                    error=str(e)
                )
                node_results.append(node_result)
                node_status[node.node_id] = "failed"
    
    async def _execute_sequential_group(self, nodes: List[WorkflowNode], execution_context: Dict[str, Any],
                                       shared_context: Dict[str, Any], node_results: List[NodeResult],
                                       execution_order: List[str], node_status: Dict[str, str]):
        """Execute a group of nodes sequentially"""
        logger.info(f"Executing sequential group: {[n.node_id for n in nodes]}")
        
        for node in nodes:
            try:
                node_input = self._prepare_node_input(node, execution_context, shared_context, node_results)
                node_result = await self._execute_node(node, node_input, shared_context)
                node_results.append(node_result)
                execution_order.append(node.node_id)
                node_status[node.node_id] = "completed"
                
                # Update shared context
                if node_result.context_updates:
                    shared_context.update(node_result.context_updates)
                    
                # Check if we should stop on failure
                if node_result.status == ExecutionStatus.FAILED:
                    node_status[node.node_id] = "failed"
                    # For sequential execution, failure might stop the group
                    break
                    
            except Exception as e:
                logger.error(f"Sequential node execution failed: {node.node_id}, error: {e}")
                node_result = NodeResult(
                    node_id=node.node_id,
                    name=node.name,
                    status=ExecutionStatus.FAILED,
                    error=str(e)
                )
                node_results.append(node_result)
                node_status[node.node_id] = "failed"
                break  # Stop sequential execution on failure
    
    async def _execute_single_node_in_graph(self, node: WorkflowNode, execution_context: Dict[str, Any],
                                          shared_context: Dict[str, Any], node_results: List[NodeResult],
                                          execution_order: List[str], node_status: Dict[str, str]):
        """Execute a single node in the graph context"""
        try:
            node_input = self._prepare_node_input(node, execution_context, shared_context, node_results)
            node_result = await self._execute_node(node, node_input, shared_context)
            node_results.append(node_result)
            execution_order.append(node.node_id)
            node_status[node.node_id] = "completed"
            
            # Update shared context
            if node_result.context_updates:
                shared_context.update(node_result.context_updates)
                
            # Special handling for routing nodes
            if node.type == NodeType.FUNCTION and hasattr(node, 'function_name'):
                if node.function_name == "route_workflow":
                    self._handle_workflow_routing(node_result, shared_context)
                    
        except Exception as e:
            logger.error(f"Single node execution failed: {node.node_id}, error: {e}")
            node_result = NodeResult(
                node_id=node.node_id,
                name=node.name,
                status=ExecutionStatus.FAILED,
                error=str(e)
            )
            node_results.append(node_result)
            node_status[node.node_id] = "failed"
    
    def _handle_workflow_routing(self, node_result: NodeResult, shared_context: Dict[str, Any]):
        """Handle routing decisions from workflow router nodes"""
        if node_result.result and "route_decision" in node_result.result:
            route_decision = node_result.result["route_decision"]
            shared_context["analysis_route"] = route_decision
            logger.info(f"Workflow routed to: {route_decision}")
    
    def _create_condition_function(self, condition: str):
        """Create a condition function for graph edges"""
        def condition_func(state):
            # Simplified condition evaluation
            # In practice, implement safe evaluation with proper context
            try:
                return eval(condition, {"__builtins__": {}}, {"state": state})
            except:
                return False
        return condition_func
