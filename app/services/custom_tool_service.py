import asyncio
import inspect
import os
import sys
import logging
import json
from typing import Any, Dict, List, Optional, Callable
import yaml
import importlib.util

from app.models.config_models import CustomToolDefinition, ToolParameterSchema
from app.utils.exceptions import ToolExecutionError

# Configure logging
logger = logging.getLogger(__name__)

# Import Strands tool decorator
try:
    from strands import tool
except ImportError:
    logger.warning("Strands SDK not available, using fallback tool implementation")
    def tool(name: Optional[str] = None, description: Optional[str] = None):
        def decorator(func):
            return func
        return decorator


class CustomToolService:
    """Service for managing custom tools with YAML-based definitions and dynamic Python function loading."""
    
    # Constants
    DEFAULT_TOOLS_DIR = "configs/tools"
    ALLOWED_FILE_EXTENSIONS = {'.yaml', '.yml'}
    
    # Security: Allowed built-in functions for custom tool execution
    SAFE_BUILTINS = {
        'print', 'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set',
        'abs', 'min', 'max', 'sum', 'round', 'sorted', 'range', 'enumerate', 'zip',
        'Exception', 'ValueError', 'TypeError', 'ImportError', 'AttributeError',
        'hasattr', 'getattr', 'setattr', '__import__'
    }
    
    def __init__(self, tools_directory: Optional[str] = None):
        """Initialize the custom tool service.
        
        Args:
            tools_directory: Directory containing tool YAML files. Defaults to DEFAULT_TOOLS_DIR.
        """
        self.tools_directory = tools_directory or self.DEFAULT_TOOLS_DIR
        self.tool_definitions: Dict[str, CustomToolDefinition] = {}
        self.tool_instances: Dict[str, Any] = {}
        
        # Ensure tools directory exists
        os.makedirs(self.tools_directory, exist_ok=True)
        
        logger.info(f"CustomToolService initialized with directory: {self.tools_directory}")
        
    def load_custom_tools(self) -> None:
        """Load all custom tools from YAML configuration files."""
        if not os.path.exists(self.tools_directory):
            logger.warning(f"Tools directory does not exist: {self.tools_directory}")
            return
            
        loaded_count = 0
        for filename in os.listdir(self.tools_directory):
            if any(filename.endswith(ext) for ext in self.ALLOWED_FILE_EXTENSIONS):
                file_path = os.path.join(self.tools_directory, filename)
                if self._load_tools_from_file(file_path):
                    loaded_count += 1
                    
        logger.info(f"Loaded custom tools from {loaded_count} files")
        
    def _load_tools_from_file(self, file_path: str) -> bool:
        """Load custom tools from a specific YAML file.
        
        Args:
            file_path: Path to the YAML file containing tool definitions.
            
        Returns:
            True if tools were successfully loaded, False otherwise.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                
            if not config or 'custom_tools' not in config:
                logger.warning(f"No custom_tools section found in {file_path}")
                return False
                
            tools_loaded = 0
            for tool_name, tool_config in config['custom_tools'].items():
                try:
                    tool_config['name'] = tool_name
                    definition = CustomToolDefinition(**tool_config)
                    self.tool_definitions[tool_name] = definition
                    self._create_tool_instance(definition)
                    tools_loaded += 1
                except Exception as e:
                    logger.error(f"Failed to load tool '{tool_name}' from {file_path}: {e}")
                    
            logger.info(f"Loaded {tools_loaded} tools from {file_path}")
            return tools_loaded > 0
            
        except Exception as e:
            logger.error(f"Error loading custom tools from {file_path}: {e}")
            return False
            
    def _create_tool_instance(self, definition: CustomToolDefinition) -> None:
        """Create a tool instance from definition using Strands @tool decorator.
        
        Args:
            definition: The custom tool definition containing function code and metadata.
        """
        try:
            base_function = self._load_function(definition)
            param_names = list(definition.parameters.keys())
            
            # Create optimized tool wrapper
            tool_wrapper = self._create_tool_wrapper(definition, base_function, param_names)
            
            # Set function metadata
            tool_wrapper.__doc__ = self._build_docstring(definition)
            tool_wrapper.__name__ = definition.name
            
            # Apply @tool decorator
            strands_tool = tool(name=definition.name, description=definition.description)(tool_wrapper)
            self.tool_instances[definition.name] = strands_tool
            
            logger.info(f"Created tool instance: {definition.name}")
            
        except Exception as e:
            logger.error(f"Error creating tool instance for {definition.name}: {e}")
            raise ToolExecutionError(f"Failed to create tool {definition.name}: {e}")
            
    def _create_tool_wrapper(self, definition: CustomToolDefinition, base_function: Callable, param_names: List[str]) -> Callable:
        """Create the appropriate tool wrapper function based on execution type.
        
        Args:
            definition: Tool definition
            base_function: The actual function to execute
            param_names: List of parameter names
            
        Returns:
            Configured tool wrapper function
        """
        if definition.async_execution:
            return self._create_async_wrapper(definition, base_function, param_names)
        else:
            return self._create_sync_wrapper(definition, base_function, param_names)
            
    def _create_async_wrapper(self, definition: CustomToolDefinition, base_function: Callable, param_names: List[str]) -> Callable:
        """Create async tool wrapper."""
        async def tool_wrapper(**kwargs):
            try:
                parsed_params = self._parse_tool_parameters(kwargs, param_names)
                validated_kwargs = self._validate_parameters(parsed_params, definition)
                
                if inspect.iscoroutinefunction(base_function):
                    result = await base_function(**validated_kwargs)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: base_function(**validated_kwargs))
                
                return result
            except Exception as e:
                logger.error(f"Error executing async tool {definition.name}: {e}")
                raise ToolExecutionError(f"Tool execution failed: {e}")
        
        return tool_wrapper
        
    def _create_sync_wrapper(self, definition: CustomToolDefinition, base_function: Callable, param_names: List[str]) -> Callable:
        """Create sync tool wrapper."""
        def tool_wrapper(**kwargs):
            try:
                parsed_params = self._parse_tool_parameters(kwargs, param_names)
                validated_kwargs = self._validate_parameters(parsed_params, definition)
                result = base_function(**validated_kwargs)
                return result
            except Exception as e:
                logger.error(f"Error executing sync tool {definition.name}: {e}")
                raise ToolExecutionError(f"Tool execution failed: {e}")
        
        return tool_wrapper
        
    def _parse_tool_parameters(self, kwargs: Dict[str, Any], param_names: List[str]) -> Dict[str, Any]:
        """Parse tool parameters from Strands input format.
        
        Handles the format: {'kwargs': 'value'} or {'kwargs': '{"param": "value"}'}
        
        Args:
            kwargs: Raw parameters from Strands
            param_names: Expected parameter names
            
        Returns:
            Parsed parameters dictionary
        """
        if 'kwargs' in kwargs and isinstance(kwargs['kwargs'], str):
            try:
                # Try to parse as JSON first
                return json.loads(kwargs['kwargs'])
            except json.JSONDecodeError:
                # If single parameter and plain string, map it directly
                if len(param_names) == 1:
                    return {param_names[0]: kwargs['kwargs']}
                else:
                    logger.warning(f"Failed to parse parameters, using raw kwargs: {kwargs}")
                    return kwargs
        else:
            return kwargs
            
    def _build_docstring(self, definition: CustomToolDefinition) -> str:
        """Build docstring for the tool function."""
        docstring = f"{definition.description}\n\n"
        if definition.parameters:
            docstring += "Args:\n"
            for param_name, param_schema in definition.parameters.items():
                docstring += f"    {param_name}: {param_schema.description}\n"
        return docstring
            
    def _validate_parameters(self, params: Dict[str, Any], definition: CustomToolDefinition) -> Dict[str, Any]:
        """Validate and convert parameters according to schema.
        
        Args:
            params: Input parameters
            definition: Tool definition with parameter schemas
            
        Returns:
            Validated and type-converted parameters
        """
        validated = {}
        
        for param_name, param_schema in definition.parameters.items():
            value = params.get(param_name)
            
            # Handle required parameters
            if param_schema.required and value is None:
                if param_schema.default is not None:
                    value = param_schema.default
                else:
                    raise ValueError(f"Required parameter '{param_name}' is missing")
            
            # Type validation and conversion
            if value is not None:
                validated[param_name] = self._convert_parameter(value, param_schema)
        
        return validated
    
    def _convert_parameter(self, value: Any, schema: ToolParameterSchema) -> Any:
        """Convert parameter to correct type based on schema."""
        try:
            if schema.type == "string":
                return str(value)
            elif schema.type == "integer":
                return int(value)
            elif schema.type == "boolean":
                return bool(value)
            elif schema.type == "number":
                return float(value)
            elif schema.type == "array":
                return list(value) if not isinstance(value, list) else value
            elif schema.type == "object":
                return dict(value) if not isinstance(value, dict) else value
            else:
                return value
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert parameter value '{value}' to type '{schema.type}': {e}")
    
    def _load_function(self, definition: CustomToolDefinition) -> Callable:
        """Load function from definition."""
        if definition.function_code:
            return self._load_function_from_code(definition)
        elif definition.function_file:
            return self._load_function_from_file(definition)
        else:
            raise ValueError(f"No function source specified for tool {definition.name}")
    
    def _load_function_from_code(self, definition: CustomToolDefinition) -> Callable:
        """Load function from inline code with security restrictions."""
        # Create secure execution environment with proper builtins
        builtins_dict = {}
        for name in self.SAFE_BUILTINS:
            if hasattr(__builtins__, name):
                builtins_dict[name] = getattr(__builtins__, name)
            elif name in dir(__builtins__):
                builtins_dict[name] = getattr(__builtins__, name)
        
        # Explicitly add essential functions and exceptions
        builtins_dict.update({
            'Exception': Exception,
            'ValueError': ValueError,
            'TypeError': TypeError,
            'ImportError': ImportError,
            'AttributeError': AttributeError,
            'str': str,
            'int': int,
            'float': float,
            'bool': bool,
            'len': len,
            'hasattr': hasattr,
            'getattr': getattr,
            '__import__': __import__
        })
        
        safe_globals = {'__builtins__': builtins_dict}
        
        # Add allowed imports
        for import_name in definition.allowed_imports:
            try:
                module = __import__(import_name)
                safe_globals[import_name] = module
            except ImportError:
                logger.warning(f"Could not import {import_name} for tool {definition.name}")
        
        # Execute code in controlled environment
        try:
            exec(definition.function_code, safe_globals)
        except Exception as e:
            raise ValueError(f"Error executing function code for {definition.name}: {e}")
        
        # Get the function
        if definition.function_name not in safe_globals:
            raise ValueError(f"Function {definition.function_name} not found in code")
            
        return safe_globals[definition.function_name]
    
    def _load_function_from_file(self, definition: CustomToolDefinition) -> Callable:
        """Load function from external file."""
        file_path = definition.function_file
        
        # Make path relative to tools directory if not absolute
        if not os.path.isabs(file_path):
            file_path = os.path.join(self.tools_directory, file_path)
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Function file not found: {file_path}")
            
        # Load module from file
        try:
            spec = importlib.util.spec_from_file_location("custom_tool_module", file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            raise ValueError(f"Error loading module from {file_path}: {e}")
        
        # Get the function
        if not hasattr(module, definition.function_name):
            raise ValueError(f"Function {definition.function_name} not found in {file_path}")
            
        return getattr(module, definition.function_name)
    
    # Public API methods
    def get_tool(self, tool_name: str) -> Optional[Any]:
        """Get a custom tool instance by name."""
        return self.tool_instances.get(tool_name)
    
    def list_custom_tools(self) -> List[str]:
        """List all available custom tools."""
        return list(self.tool_definitions.keys())
    
    def get_tool_definition(self, tool_name: str) -> Optional[CustomToolDefinition]:
        """Get tool definition by name."""
        return self.tool_definitions.get(tool_name)
    
    def register_tool(self, definition: CustomToolDefinition) -> None:
        """Register a new custom tool."""
        self.tool_definitions[definition.name] = definition
        self._create_tool_instance(definition)
        logger.info(f"Registered new tool: {definition.name}")
    
    def reload_tools(self) -> None:
        """Reload all custom tools from configuration files."""
        logger.info("Reloading all custom tools")
        self.tool_definitions.clear()
        self.tool_instances.clear()
        self.load_custom_tools()
        
    def validate_tool_definition(self, definition: CustomToolDefinition) -> bool:
        """Validate a tool definition.
        
        Args:
            definition: Tool definition to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Try to load the function
            function = self._load_function(definition)
            
            # Check function signature matches parameters
            sig = inspect.signature(function)
            
            # Validate parameter names match
            defined_params = set(definition.parameters.keys())
            function_params = set(sig.parameters.keys())
            
            if defined_params != function_params:
                logger.warning(f"Parameter mismatch for {definition.name}: "
                             f"defined={defined_params}, function={function_params}")
                
            return True
            
        except Exception as e:
            logger.error(f"Validation failed for {definition.name}: {e}")
            return False
            
    def get_tool_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded tools."""
        return {
            "total_tools": len(self.tool_definitions),
            "tools_by_type": {
                "sync": sum(1 for d in self.tool_definitions.values() if not d.async_execution),
                "async": sum(1 for d in self.tool_definitions.values() if d.async_execution)
            },
            "tools_directory": self.tools_directory,
            "tool_names": list(self.tool_definitions.keys())
        }
