import asyncio
import time
import uuid
import os
import logging
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.models.litellm import LiteLLMModel
from strands.agent.conversation_manager import SlidingWindowConversationManager, SummarizingConversationManager, NullConversationManager
from strands.hooks import HookProvider, HookRegistry
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException

# Configure logging
logger = logging.getLogger(__name__)

# Import tools from strands_tools
try:
    from strands_tools.calculator import calculator
    from strands_tools.tavily import tavily_search  # Using tavily_search for web search
    from strands_tools.browser import browser
    from strands_tools.python_repl import python_repl
    from strands_tools.current_time import current_time
except ImportError as e:
    # Fallback if specific tools not available
    logger.warning(f"Some tools not available: {e}")
    calculator = None
    tavily_search = None
    browser = None
    python_repl = None
    current_time = None

from app.models.agent_models import (
    AgentExecutionRequest, AgentExecutionResponse, 
    StreamingEvent, ToolCall, SessionInfo
)
from app.models.config_models import AgentConfig, ToolConfig
from app.services.config_service import ConfigService
from app.services.custom_tool_service import CustomToolService
from app.utils.exceptions import (
    AgentNotFoundError, AgentExecutionError, SessionNotFoundError,
    TimeoutError, ToolExecutionError, ContextWindowOverflowError as CustomContextOverflow
)


class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SessionInfo] = {}
        
    def create_session(self, agent_name: str) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        session = SessionInfo(
            session_id=session_id,
            agent_name=agent_name,
            created_at=datetime.now(),
            last_activity=datetime.now(),
            message_count=0
        )
        self._sessions[session_id] = session
        return session_id
        
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID"""
        return self._sessions.get(session_id)
        
    def update_session_activity(self, session_id: str):
        """Update session last activity"""
        if session_id in self._sessions:
            self._sessions[session_id].last_activity = datetime.now()
            self._sessions[session_id].message_count += 1
            
    def clear_session(self, session_id: str) -> bool:
        """Clear session"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


class StrandsAgentService:
    def __init__(self, config_service: ConfigService):
        self.config_service = config_service
        self.session_manager = SessionManager()
        logger.info("Initializing CustomToolService...")
        self.custom_tool_service = CustomToolService()
        self.custom_tool_service.load_custom_tools()
        logger.info("CustomToolService initialized successfully")
        self._agent_instances: Dict[str, Agent] = {}
        
    def _create_model(self, agent_config: AgentConfig) -> LiteLLMModel:
        """Create a LiteLLM model instance from configuration"""
        model_config = agent_config.model
        
        # Prepare client_args with environment variable substitution
        client_args = {}
        if hasattr(model_config, 'client_args') and model_config.client_args:
            for key, value in model_config.client_args.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    # Extract environment variable name
                    env_var = value[2:-1]
                    env_value = os.getenv(env_var)
                    if env_value is None:
                        logger.warning(f"Environment variable {env_var} not found")
                        client_args[key] = None
                    else:
                        logger.debug(f"Loaded environment variable {env_var}")
                        client_args[key] = env_value
                else:
                    client_args[key] = value
        
        logger.debug(f"Model client args configured for {model_config.provider}")
        
        # Prepare model parameters
        params = {}
        if hasattr(model_config, 'params') and model_config.params:
            params = model_config.params
        else:
            # Use legacy fields for backward compatibility
            params = {
                "temperature": model_config.temperature,
                "max_tokens": model_config.max_tokens
            }
        
        return LiteLLMModel(
            model_id=model_config.model_id,
            client_args=client_args,
            params=params
        )
        
    def _create_conversation_manager(self, config: AgentConfig):
        """Create conversation manager based on config"""
        cm_config = config.conversation_manager
        
        if cm_config.type == "sliding_window":
            return SlidingWindowConversationManager(
                window_size=cm_config.window_size,
                should_truncate_results=cm_config.should_truncate_results
            )
        elif cm_config.type == "summarizing":
            return SummarizingConversationManager(
                summary_ratio=getattr(cm_config, 'summary_ratio', 0.3),
                preserve_recent_messages=getattr(cm_config, 'preserve_recent_messages', 10)
            )
        elif cm_config.type == "null":
            return NullConversationManager()
        else:  # default to null
            return None
            
    def _create_tools(self, agent_config: AgentConfig) -> List[Any]:
        """Create tools for the agent"""
        tools = []
        
        # Load built-in tools and custom tools
        for tool_config in agent_config.tools:
            if not tool_config.enabled:
                continue
                
            tool_name = tool_config.name
            
            # Try built-in tools first
            if tool_name == "calculator" and calculator:
                tools.append(calculator)
            elif tool_name == "current_time" and current_time:
                tools.append(current_time)
            elif tool_name == "web_search" and tavily_search:
                tools.append(tavily_search)
            elif tool_name == "browser" and browser:
                tools.append(browser)
            elif tool_name == "python_repl" and python_repl:
                tools.append(python_repl)
            else:
                # Try custom tools
                logger.debug(f"Looking for custom tool: {tool_name}")
                custom_tool = self.custom_tool_service.get_tool(tool_name)
                if custom_tool:
                    logger.debug(f"Adding custom tool {tool_name} to agent")
                    tools.append(custom_tool)
                else:
                    logger.warning(f"Unknown tool {tool_name}")
        
        return tools
        
    def _create_hooks(self, hook_configs: List) -> List:
        """Create hook instances from configuration"""
        hooks = []
        
        # For now, we'll skip hook implementation as LoggingHook is not available
        # in the current Strands SDK version. Hooks can be added later when
        # proper hook implementations are available.
        
        return hooks
        
    def _get_or_create_agent(self, agent_name: str) -> Agent:
        """Get or create an agent instance"""
        if agent_name in self._agent_instances:
            return self._agent_instances[agent_name]
            
        # Get agent configuration
        agent_config = self.config_service.get_agent_config(agent_name)
        if not agent_config:
            raise AgentNotFoundError(agent_name)
            
        try:
            # Create LiteLLM model
            model = self._create_model(agent_config)
            
            # Create conversation manager
            conversation_manager = self._create_conversation_manager(agent_config)
            
            # Create tools
            tools = self._create_tools(agent_config)
            
            # Create hooks
            hooks = self._create_hooks(agent_config.hooks)
            
            # Create the agent
            agent = Agent(
                model=model,
                system_prompt=agent_config.system_prompt,
                conversation_manager=conversation_manager,
                tools=tools,
                hooks=hooks
            )
            
            # Note: State initialization is handled by the Agent internally
            # The AgentState object doesn't have an update method in the current SDK version
            
            self._agent_instances[agent_name] = agent
            return agent
            
        except Exception as e:
            raise AgentExecutionError(f"Failed to create agent {agent_name}: {str(e)}")
            
    async def execute_agent(self, request: AgentExecutionRequest) -> AgentExecutionResponse:
        """Execute agent with the given request"""
        start_time = time.time()
        
        # Get or create agent
        agent = self._get_or_create_agent(request.agent_config)
        agent_config = self.config_service.get_agent_config(request.agent_config)
        
        # Create or get session
        session_id = request.session_id or self.session_manager.create_session(agent_config.name)
        self.session_manager.update_session_activity(session_id)
        
        try:
            # Execute agent with timeout using invoke_async
            result: AgentResult = await asyncio.wait_for(
                agent.invoke_async(
                    prompt=request.prompt,
                    **request.context or {}
                ),
                timeout=request.timeout
            )
            
            execution_time = time.time() - start_time
            
            logger.debug(f"Agent execution completed for {request.agent_config}")
            
            # Extract response text from the result
            response_text = ""
            if hasattr(result, 'message') and hasattr(result.message, 'content'):
                if isinstance(result.message.content, list) and len(result.message.content) > 0:
                    response_text = result.message.content[0].text
                elif isinstance(result.message.content, str):
                    response_text = result.message.content
            else:
                response_text = str(result) if result else "No response generated"
            
            # Extract token usage from metrics
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'accumulated_usage'):
                usage = result.metrics.accumulated_usage
                token_usage = {
                    "prompt_tokens": usage.get('inputTokens', 0),
                    "completion_tokens": usage.get('outputTokens', 0), 
                    "total_tokens": usage.get('totalTokens', 0)
                }
            
            # Extract tool calls and metrics
            tool_calls = []
            tools_used = 0
            
            # Extract tool call information from metrics
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'tool_metrics'):
                logger.info(f"Tool metrics found: {result.metrics.tool_metrics}")
                for tool_name, tool_metric in result.metrics.tool_metrics.items():
                    tools_used += tool_metric.call_count
                    
                    # Log detailed tool execution info
                    logger.info(f"Tool '{tool_name}' executed:")
                    logger.info(f"  - Call count: {tool_metric.call_count}")
                    logger.info(f"  - Success count: {tool_metric.success_count}")
                    logger.info(f"  - Error count: {tool_metric.error_count}")
                    logger.info(f"  - Total time: {tool_metric.total_time}")
                    if hasattr(tool_metric, 'input'):
                        logger.info(f"  - Input: {tool_metric.input}")
                    if hasattr(tool_metric, 'output'):
                        logger.info(f"  - Output: {tool_metric.output}")
                    
                    # Extract tool call details
                    tool_call = ToolCall(
                        tool_name=tool_name,
                        parameters=tool_metric.input if hasattr(tool_metric, 'input') else {},
                        result={"success": tool_metric.success_count > 0, "errors": tool_metric.error_count},
                        execution_time=tool_metric.total_time,
                        success=tool_metric.success_count > 0
                    )
                    tool_calls.append(tool_call)
            else:
                logger.info("No tool metrics found in result")
                
            # Debug: Print the complete agent result structure
            logger.info("=== AGENT RESULT DEBUG ===")
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Result content: {result}")
            if hasattr(result, 'message'):
                logger.info(f"Message: {result.message}")
            if hasattr(result, 'metrics'):
                logger.info(f"Metrics: {result.metrics}")
                if hasattr(result.metrics, 'tool_metrics'):
                    logger.info(f"Tool metrics: {result.metrics.tool_metrics}")
            logger.info("=== END AGENT RESULT DEBUG ===")
            
            # Extract stop reason
            stop_reason = getattr(result, 'stop_reason', 'completed')
            
            # Extract cycle count for iterations
            iterations = 1
            if hasattr(result, 'metrics') and hasattr(result.metrics, 'cycle_count'):
                iterations = result.metrics.cycle_count
            
            return AgentExecutionResponse(
                session_id=session_id,
                agent_name=agent_config.name,
                response=response_text,
                execution_time=execution_time,
                token_usage=token_usage,
                tool_calls=tool_calls,
                stop_reason=stop_reason,
                metrics={"iterations": iterations, "tools_used": tools_used, "success": True}
            )
            
        except asyncio.TimeoutError:
            raise TimeoutError(request.timeout)
        except ContextWindowOverflowException as e:
            raise CustomContextOverflow(str(e))
        except Exception as e:
            raise AgentExecutionError(f"Unexpected error: {str(e)}")
            
    async def stream_agent_execution(self, request: AgentExecutionRequest) -> AsyncIterator[StreamingEvent]:
        """Stream agent execution events"""
        session_id = request.session_id or self.session_manager.create_session(request.agent_config)
        
        # Get or create agent
        agent = self._get_or_create_agent(request.agent_config)
        agent_config = self.config_service.get_agent_config(request.agent_config)
        
        # Start streaming events
        yield StreamingEvent(
            event_type="execution_start",
            timestamp=datetime.now(),
            session_id=session_id,
            data={"agent_name": agent_config.name, "prompt": request.prompt}
        )
        
        try:
            # Stream execution using Strands streaming API
            async for event in agent.stream_async(
                prompt=request.prompt,
                **request.context or {}
            ):
                # Convert Strands events to our format
                yield StreamingEvent(
                    event_type="stream_event",
                    timestamp=datetime.now(),
                    session_id=session_id,
                    data={"event": str(event)}  # Simplified conversion
                )
                
        except ContextWindowOverflowException as e:
            yield StreamingEvent(
                event_type="error",
                timestamp=datetime.now(),
                session_id=session_id,
                data={"error": "context_overflow", "message": str(e)}
            )
        except Exception as e:
            yield StreamingEvent(
                event_type="error",
                timestamp=datetime.now(),
                session_id=session_id,
                data={"error": "unexpected_error", "message": str(e)}
            )
            
    def get_session(self, session_id: str) -> SessionInfo:
        """Get session information"""
        session = self.session_manager.get_session(session_id)
        if not session:
            raise SessionNotFoundError(session_id)
        return session
        
    def list_sessions(self) -> List[SessionInfo]:
        """List all active sessions"""
        return list(self.session_manager._sessions.values())
        
    def clear_session(self, session_id: str) -> bool:
        """Clear a session"""
        return self.session_manager.clear_session(session_id)
        
    def clear_all_sessions(self) -> int:
        """Clear all sessions and return count of cleared sessions"""
        count = len(self.session_manager._sessions)
        self.session_manager._sessions.clear()
        return count
        
    def list_available_agents(self) -> List[str]:
        """List all available agents"""
        return self.config_service.list_available_agents()
        
    def reload_agents(self):
        """Reload all agent configurations and clear cached instances"""
        self._agent_instances.clear()
        self.config_service.reload_configs()
