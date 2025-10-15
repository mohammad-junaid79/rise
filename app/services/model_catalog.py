"""
Model Catalog Service for managing centralized model configurations
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Optional, Any
from pydantic import BaseModel, Field
from app.utils.exceptions import ConfigurationError


class CatalogModelConfig(BaseModel):
    """Model configuration from the catalog"""
    provider: str = Field(..., description="Model provider (bedrock, openai, etc.)")
    model_id: str = Field(..., description="Model identifier")
    region: Optional[str] = Field(None, description="AWS region for bedrock")
    temperature: float = Field(0.7, description="Model temperature")
    max_tokens: int = Field(4096, description="Maximum tokens")
    client_args: Optional[Dict[str, Any]] = Field(None, description="Client configuration")
    params: Optional[Dict[str, Any]] = Field(None, description="Additional model parameters")
    default_params: Optional[Dict[str, Any]] = Field(None, description="Default model parameters")
    
    # Metadata
    display_name: Optional[str] = Field(None, description="Human-readable model name")
    description: Optional[str] = Field(None, description="Model description")
    context_window: Optional[int] = Field(None, description="Context window size")
    performance_tier: Optional[str] = Field(None, description="Performance tier (high, medium, low)")
    cost_tier: Optional[str] = Field(None, description="Cost tier (high, medium, low)")
    capabilities: Optional[list] = Field(None, description="Model capabilities")
    pricing: Optional[Dict[str, Any]] = Field(None, description="Pricing information")


class ModelCatalog:
    """Service for managing model catalog and resolving model references"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.catalog_file = self.config_dir / "models" / "model_catalog.yaml"
        self._models: Dict[str, CatalogModelConfig] = {}
        self.load_catalog()
        
    def load_catalog(self):
        """Load model catalog from YAML file"""
        if not self.catalog_file.exists():
            raise ConfigurationError(f"Model catalog file not found: {self.catalog_file}")
            
        try:
            with open(self.catalog_file, 'r') as f:
                catalog_data = yaml.safe_load(f)
                
            if 'models' not in catalog_data:
                raise ConfigurationError("Model catalog must have a 'models' section")
                
            self._models.clear()
            for model_key, model_data in catalog_data['models'].items():
                try:
                    self._models[model_key] = CatalogModelConfig(**model_data)
                except Exception as e:
                    raise ConfigurationError(f"Invalid model config for {model_key}: {str(e)}")
                    
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse model catalog YAML: {str(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load model catalog: {str(e)}")
            
    def get_model(self, model_ref: str) -> Optional[CatalogModelConfig]:
        """Get model configuration by reference"""
        return self._models.get(model_ref)
        
    def list_models(self) -> Dict[str, CatalogModelConfig]:
        """Get all available models"""
        return self._models.copy()
        
    def list_model_refs(self) -> list:
        """Get list of all model reference keys"""
        return list(self._models.keys())
        
    def reload_catalog(self):
        """Reload the model catalog"""
        self.load_catalog()
        
    def validate_model_ref(self, model_ref: str) -> bool:
        """Check if a model reference exists"""
        return model_ref in self._models
