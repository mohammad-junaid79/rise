import os
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from app.models.config_models import AgentConfig, ToolsConfig, ModelConfig
from app.models.workflow_models import WorkflowConfig
from app.services.model_catalog import ModelCatalog
from app.utils.exceptions import ConfigurationError


class ConfigService:
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.agents_dir = self.config_dir / "agents"
        self.tools_dir = self.config_dir / "tools"
        self.workflows_dir = self.config_dir / "workflows"
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._tool_configs: Dict[str, ToolsConfig] = {}
        self._workflow_configs: Dict[str, WorkflowConfig] = {}
        
        # Initialize model catalog
        self.model_catalog = ModelCatalog(config_dir)
        
        # Load all configurations on initialization
        self.load_all_configs()
        
    def load_all_configs(self):
        """Load all agent, tool, and workflow configurations from YAML files"""
        self._load_agent_configs()
        self._load_tool_configs()
        self._load_workflow_configs()
        
    def _resolve_model_config(self, model_config: ModelConfig) -> ModelConfig:
        """Resolve model configuration from catalog reference if needed"""
        if not model_config.model_ref:
            # No model reference, use direct configuration
            return model_config
            
        # Get model from catalog
        catalog_model = self.model_catalog.get_model(model_config.model_ref)
        if not catalog_model:
            raise ConfigurationError(f"Model reference '{model_config.model_ref}' not found in catalog")
            
        # Get defaults from catalog
        catalog_defaults = catalog_model.default_params or {}
        catalog_temp = catalog_defaults.get('temperature', catalog_model.temperature)
        catalog_max_tokens = catalog_defaults.get('max_tokens', catalog_model.max_tokens)
        
        # Get overrides from agent config params
        agent_params = model_config.params or {}
        override_temp = agent_params.get('temperature', catalog_temp)
        override_max_tokens = agent_params.get('max_tokens', catalog_max_tokens)
        
        # Merge all parameters
        final_params = {**catalog_defaults, **agent_params}
        
        # Create resolved config
        resolved_config = ModelConfig(
            provider=catalog_model.provider,
            model_id=catalog_model.model_id,
            region=catalog_model.region,
            temperature=override_temp,
            max_tokens=override_max_tokens,
            client_args=catalog_model.client_args,
            params=final_params
        )
        
        return resolved_config
        
    def _load_agent_configs(self):
        """Load all agent configurations"""
        if not self.agents_dir.exists():
            raise ConfigurationError(f"Agents config directory not found: {self.agents_dir}")
            
        for config_file in self.agents_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                # Create agent config and resolve model references
                agent_config = AgentConfig(**config_data['agent'])
                agent_config.model = self._resolve_model_config(agent_config.model)
                
                agent_name = config_file.stem
                self._agent_configs[agent_name] = agent_config
                
            except Exception as e:
                raise ConfigurationError(f"Failed to load agent config {config_file}: {str(e)}")
                
    def _load_tool_configs(self):
        """Load all tool configurations"""
        if not self.tools_dir.exists():
            raise ConfigurationError(f"Tools config directory not found: {self.tools_dir}")
            
        for config_file in self.tools_dir.glob("*.yaml"):
            # Skip custom_tools.yaml as it's handled by CustomToolService
            if config_file.name == "custom_tools.yaml":
                continue
                
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                tools_config = ToolsConfig(**config_data)
                config_name = config_file.stem
                self._tool_configs[config_name] = tools_config
                
            except Exception as e:
                raise ConfigurationError(f"Failed to load tool config {config_file}: {str(e)}")
                
    def _load_workflow_configs(self):
        """Load all workflow configurations"""
        if not self.workflows_dir.exists():
            # Workflows directory is optional
            return
            
        for config_file in self.workflows_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                    
                if not config_data:
                    logger.warning(f"Empty workflow config file: {config_file}")
                    continue
                    
                workflow_config = WorkflowConfig(**config_data)
                workflow_id = config_data.get('workflow_id', config_file.stem)
                self._workflow_configs[workflow_id] = workflow_config
                
            except yaml.YAMLError as ye:
                logger.error(f"YAML parsing error in {config_file}: {str(ye)}")
                logger.warning(f"Skipping corrupted workflow config: {config_file}")
                continue
            except Exception as e:
                logger.error(f"Failed to load workflow config {config_file}: {str(e)}")
                logger.warning(f"Skipping invalid workflow config: {config_file}")
                continue
                
    def get_workflow_config(self, workflow_id: str) -> Optional[WorkflowConfig]:
        """Get workflow configuration by ID"""
        return self._workflow_configs.get(workflow_id)
        
    def get_all_workflow_configs(self) -> Dict[str, WorkflowConfig]:
        """Get all workflow configurations"""
        return self._workflow_configs.copy()
        
    def list_available_workflows(self) -> List[str]:
        """List all available workflow IDs"""
        return list(self._workflow_configs.keys())
                
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get agent configuration by name"""
        return self._agent_configs.get(agent_name)
        
    def get_all_agent_configs(self) -> Dict[str, AgentConfig]:
        """Get all agent configurations"""
        return self._agent_configs.copy()
        
    def get_tool_config(self, tool_name: str) -> Optional[ToolsConfig]:
        """Get tool configuration by name"""
        return self._tool_configs.get(tool_name)
        
    def get_all_tool_configs(self) -> Dict[str, ToolsConfig]:
        """Get all tool configurations"""
        return self._tool_configs.copy()
        
    def reload_configs(self):
        """Reload all configurations"""
        self._agent_configs.clear()
        self._tool_configs.clear()
        self._workflow_configs.clear()
        
        # Reload model catalog first
        self.model_catalog.reload_catalog()
        
        self.load_all_configs()
        
    def validate_agent_config(self, config_data: dict) -> AgentConfig:
        """Validate agent configuration data"""
        try:
            return AgentConfig(**config_data['agent'])
        except Exception as e:
            raise ConfigurationError(f"Invalid agent configuration: {str(e)}")
            
    def list_available_agents(self) -> List[str]:
        """List all available agent names"""
        return list(self._agent_configs.keys())
        
    def list_available_tools(self) -> List[str]:
        """List all available tool names"""
        all_tools = set()
        for tools_config in self._tool_configs.values():
            all_tools.update(tools_config.tools.keys())
        return list(all_tools)
    
    def save_agent_config_with_model_ref(self, agent_name: str, agent_config: AgentConfig, model_ref: Optional[str] = None, model_params: Optional[Dict[str, Any]] = None) -> bool:
        """Save agent configuration to YAML file with model catalog reference"""
        try:
            # Create agents directory if it doesn't exist
            self.agents_dir.mkdir(parents=True, exist_ok=True)
            
            config_file = self.agents_dir / f"{agent_name}.yaml"
            
            # Convert agent config to dict
            config_dict = agent_config.dict()
            
            # Replace model configuration with model reference if provided
            if model_ref:
                config_dict['model'] = {
                    'model_ref': model_ref
                }
                # Add parameter overrides if provided
                if model_params:
                    config_dict['model']['params'] = model_params
            
            config_data = {
                "agent": config_dict
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            # Update in-memory cache with resolved config
            self._agent_configs[agent_name] = agent_config
            return True
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save agent config {agent_name}: {str(e)}")
    
    def save_agent_config(self, agent_name: str, agent_config: AgentConfig) -> bool:
        """Save agent configuration to YAML file"""
        try:
            # Create agents directory if it doesn't exist
            self.agents_dir.mkdir(parents=True, exist_ok=True)
            
            config_file = self.agents_dir / f"{agent_name}.yaml"
            config_data = {
                "agent": agent_config.dict()
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            # Update in-memory cache
            self._agent_configs[agent_name] = agent_config
            return True
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save agent config {agent_name}: {str(e)}")
    
    def delete_agent_config(self, agent_name: str) -> bool:
        """Delete agent configuration file"""
        try:
            config_file = self.agents_dir / f"{agent_name}.yaml"
            
            if config_file.exists():
                config_file.unlink()
                
            # Remove from in-memory cache
            if agent_name in self._agent_configs:
                del self._agent_configs[agent_name]
                
            return True
            
        except Exception as e:
            raise ConfigurationError(f"Failed to delete agent config {agent_name}: {str(e)}")
    
    def get_tools_config(self) -> Optional[ToolsConfig]:
        """Get the main tools configuration"""
        # Return the first tools config, or try to find a main one
        if not self._tool_configs:
            return None
            
        # Look for a main tools config file
        if "main" in self._tool_configs:
            return self._tool_configs["main"]
        elif "tools" in self._tool_configs:
            return self._tool_configs["tools"]
        else:
            # Return the first available tools config
            return next(iter(self._tool_configs.values()))
    
    def save_tools_config(self, tools_config: ToolsConfig, config_name: str = "main") -> bool:
        """Save tools configuration to YAML file"""
        try:
            # Create tools directory if it doesn't exist
            self.tools_dir.mkdir(parents=True, exist_ok=True)
            
            config_file = self.tools_dir / f"{config_name}.yaml"
            config_data = tools_config.dict()
            
            with open(config_file, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            
            # Update in-memory cache
            self._tool_configs[config_name] = tools_config
            return True
            
        except Exception as e:
            raise ConfigurationError(f"Failed to save tools config {config_name}: {str(e)}")
    
    def create_empty_tools_config(self) -> ToolsConfig:
        """Create an empty tools configuration"""
        from app.models.config_models import ToolsConfig
        return ToolsConfig(tools={}, custom_tools={})
