from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum


# Workflow Topology Types
class WorkflowTopology(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    MESH = "mesh"
    GRAPH = "graph"
    SWARM = "swarm"


# Communication Patterns
class CommunicationPattern(str, Enum):
    DIRECT = "direct"  # Point-to-point communication
    BROADCAST = "broadcast"  # One-to-many communication
    CONDITIONAL = "conditional"  # Conditional routing based on conditions
    AGGREGATION = "aggregation"  # Many-to-one aggregation


# Workflow Node Types
class NodeType(str, Enum):
    AGENT = "agent"
    WORKFLOW = "workflow"  # Nested workflow
    FUNCTION = "function"  # Custom function node
    CONDITION = "condition"  # Conditional node


# Execution Status
class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


# Node Definition
class WorkflowNode(BaseModel):
    node_id: str = Field(..., description="Unique identifier for the node")
    name: str = Field(..., description="Display name for the node")
    type: NodeType = Field(..., description="Type of the node")
    
    # Agent Configuration
    agent_config: Optional[str] = Field(None, description="Agent configuration name for agent nodes")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt override")
    
    # Function Configuration
    function_name: Optional[str] = Field(None, description="Function name for function nodes")
    function_params: Optional[Dict[str, Any]] = Field(None, description="Function parameters")
    
    # Condition Configuration
    condition_expression: Optional[str] = Field(None, description="Condition expression for conditional nodes")
    
    # Execution Configuration
    timeout: Optional[int] = Field(300, description="Node execution timeout in seconds")
    retry_count: Optional[int] = Field(0, description="Number of retries on failure")
    priority: Optional[int] = Field(0, description="Execution priority (higher = more important)")
    
    # Context and Data Flow
    input_mapping: Optional[Dict[str, str]] = Field(None, description="Input parameter mapping")
    output_mapping: Optional[Dict[str, str]] = Field(None, description="Output parameter mapping")
    context_keys: Optional[List[str]] = Field(None, description="Context keys this node contributes")


# Edge Definition for Graph Topologies
class WorkflowEdge(BaseModel):
    from_node: str = Field(..., description="Source node ID")
    to_node: str = Field(..., description="Target node ID")
    condition: Optional[str] = Field(None, description="Condition for edge traversal")
    communication_pattern: CommunicationPattern = Field(CommunicationPattern.DIRECT, description="Communication pattern")
    weight: Optional[float] = Field(1.0, description="Edge weight for prioritization")


# Communication Configuration
class CommunicationConfig(BaseModel):
    pattern: CommunicationPattern = Field(..., description="Communication pattern type")
    source_nodes: List[str] = Field(..., description="Source node IDs")
    target_nodes: List[str] = Field(..., description="Target node IDs")
    condition: Optional[str] = Field(None, description="Condition for communication")
    message_template: Optional[str] = Field(None, description="Message template for communication")


# Context Sharing Configuration
class ContextConfig(BaseModel):
    shared_keys: List[str] = Field(default_factory=list, description="Keys shared across all nodes")
    node_specific_keys: Dict[str, List[str]] = Field(default_factory=dict, description="Node-specific context keys")
    persistence: bool = Field(True, description="Whether to persist context across workflow runs")
    isolation_level: Literal["none", "node", "workflow"] = Field("workflow", description="Context isolation level")


# Parallel Execution Configuration
class ParallelConfig(BaseModel):
    max_concurrent_nodes: Optional[int] = Field(None, description="Maximum concurrent node executions")
    batch_size: Optional[int] = Field(None, description="Batch size for parallel execution")
    aggregation_strategy: Literal["wait_all", "wait_any", "timeout"] = Field("wait_all", description="Aggregation strategy")
    aggregation_timeout: Optional[int] = Field(None, description="Timeout for aggregation in seconds")


# Workflow Configuration
class WorkflowConfig(BaseModel):
    workflow_id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Workflow display name")
    description: Optional[str] = Field(None, description="Workflow description")
    version: str = Field("1.0.0", description="Workflow version")
    
    # Topology Configuration
    topology: WorkflowTopology = Field(..., description="Workflow topology type")
    
    # Nodes and Edges
    nodes: List[WorkflowNode] = Field(..., description="Workflow nodes")
    edges: Optional[List[WorkflowEdge]] = Field(None, description="Workflow edges (for graph topology)")
    entry_points: Optional[List[str]] = Field(None, description="Entry point node IDs")
    
    # Communication and Context
    communication: List[CommunicationConfig] = Field(default_factory=list, description="Communication configurations")
    context: ContextConfig = Field(default_factory=ContextConfig, description="Context sharing configuration")
    
    # Execution Configuration
    parallel_config: Optional[ParallelConfig] = Field(None, description="Parallel execution configuration")
    max_execution_time: Optional[int] = Field(1800, description="Maximum workflow execution time in seconds")
    error_handling: Literal["stop", "continue", "retry"] = Field("stop", description="Error handling strategy")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")


# Workflow Creation Request
class CreateWorkflowRequest(BaseModel):
    workflow_id: str = Field(..., description="Unique workflow identifier")
    name: str = Field(..., description="Workflow display name")
    description: Optional[str] = Field(None, description="Workflow description")
    version: str = Field("1.0.0", description="Workflow version")
    topology: WorkflowTopology = Field(..., description="Workflow topology type")
    nodes: List[Dict[str, Any]] = Field(..., description="Workflow nodes (raw format)")
    edges: Optional[List[Dict[str, Any]]] = Field(None, description="Workflow edges (raw format)")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    # New agents to be created
    new_agents: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="New agents to create")

# Update Workflow Request
class UpdateWorkflowRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    topology: Optional[WorkflowTopology] = None
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None
    tags: Optional[List[str]] = None
    new_agents: Optional[List[Dict[str, Any]]] = None

# Workflow Execution Request
class WorkflowExecutionRequest(BaseModel):
    workflow_id: str = Field(..., description="Workflow ID to execute")
    input_data: Union[str, Dict[str, Any]] = Field(default_factory=dict, description="Input data for the workflow (can be plain text or JSON object)")
    context: Optional[Dict[str, Any]] = Field(None, description="Initial context")
    session_id: Optional[str] = Field(None, description="Session ID for stateful execution")
    execution_options: Optional[Dict[str, Any]] = Field(None, description="Execution options override")


# Node Execution Result
class NodeResult(BaseModel):
    node_id: str
    name: str
    status: ExecutionStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    context_updates: Optional[Dict[str, Any]] = None


# Workflow Execution Result
class WorkflowExecutionResult(BaseModel):
    workflow_id: str
    execution_id: str
    status: ExecutionStatus
    
    # Execution Details
    node_results: List[NodeResult] = Field(default_factory=list)
    execution_order: List[str] = Field(default_factory=list)
    
    # Context and Data
    final_context: Dict[str, Any] = Field(default_factory=dict)
    output_data: Dict[str, Any] = Field(default_factory=dict)
    
    # Metrics
    total_execution_time: Optional[float] = None
    nodes_executed: int = 0
    nodes_failed: int = 0
    nodes_skipped: int = 0
    
    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Error Information
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# Workflow Status Response
class WorkflowStatusResponse(BaseModel):
    workflow_id: str
    execution_id: str
    status: ExecutionStatus
    progress_percentage: float
    current_nodes: List[str] = Field(default_factory=list)
    completed_nodes: List[str] = Field(default_factory=list)
    failed_nodes: List[str] = Field(default_factory=list)
    remaining_nodes: List[str] = Field(default_factory=list)
    execution_time: Optional[float] = None
    estimated_remaining_time: Optional[float] = None


# Workflow List Response
class WorkflowInfo(BaseModel):
    workflow_id: str
    name: str
    description: Optional[str] = None
    topology: WorkflowTopology
    version: str
    node_count: int
    tags: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    status: Literal["active", "inactive", "draft"] = "active"
