import re
import pandas as pd
import logging
import json
import requests
import os
from datetime import datetime
import openai
import anthropic
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

# Get logger
logger = logging.getLogger('timesheet_analyzer')

# Load environment variables
load_dotenv()

# Configuration for AI service
AI_API_KEY = os.getenv('AI_API_KEY')  # Store your API key in environment variable
AI_SERVICE_URL = os.getenv('AI_SERVICE_URL', 'https://api.openai.com/v1/chat/completions')  # Default to OpenAI
AI_MODEL = os.getenv('AI_MODEL', 'gpt-4')  # Default to GPT-4

# Enhanced LLM Client
class EnhancedLLMClient:
    """Enhanced LLM client with dynamic model loading and unified interface."""
    
    def __init__(self):
        """Initialize the LLM client."""
        # Load API keys from environment
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        
        # Print masked API keys for verification
        if self.openai_api_key:
            masked_openai = f"{self.openai_api_key[:4]}...{self.openai_api_key[-4:]}" if len(self.openai_api_key) > 8 else "Too short"
            logger.info(f"OpenAI API key loaded: {masked_openai}")
        else:
            logger.info("OpenAI API key not found")
            
        if self.anthropic_api_key:
            masked_anthropic = f"{self.anthropic_api_key[:4]}...{self.anthropic_api_key[-4:]}" if len(self.anthropic_api_key) > 8 else "Too short"
            logger.info(f"Anthropic API key loaded: {masked_anthropic}")
        else:
            logger.info("Anthropic API key not found")
        
        # Initialize clients if API keys are available
        self.openai_client = None
        self.anthropic_client = None
        
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        if self.anthropic_api_key:
            try:
                # Fix: The anthropic module imported is possibly a different version than the one designed for.
                # We'll use a try-except approach to handle initialization
                try:
                    # Try the standard initialization (works for newer versions)
                    self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                except TypeError as e:
                    if "unexpected keyword argument 'proxies'" in str(e):
                        # If we get the proxies error, try importing Client directly
                        from anthropic import Client
                        self.anthropic_client = Client(api_key=self.anthropic_api_key)
                    else:
                        raise
                
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.error(f"Error initializing Anthropic client: {str(e)}")
                logger.error(f"Exception type: {type(e)}")
                logger.error("IMPORTANT: There appears to be an issue with the Anthropic Python library version")
                logger.error("Try reinstalling it with: pip install --upgrade anthropic")
        
        # Cache for available models
        self.available_models = {
            "openai": [],
            "anthropic": [],
            "ollama": []
        }
    
    def get_available_models(self, refresh: bool = False) -> Dict[str, List[str]]:
        """
        Get available models from all providers.
        
        Args:
            refresh: Whether to refresh the cache
            
        Returns:
            Dictionary with provider names as keys and lists of models as values
        """
        if not refresh and all(self.available_models.values()):
            return self.available_models
        
        # Get OpenAI models
        if self.openai_client:
            try:
                logger.info("Fetching OpenAI models...")
                response = self.openai_client.models.list()
                self.available_models["openai"] = [model.id for model in response.data]
                logger.info(f"Found {len(self.available_models['openai'])} OpenAI models")
            except Exception as e:
                logger.error(f"Error fetching OpenAI models: {e}")
        
        # Get Anthropic models
        if self.anthropic_client:
            try:
                logger.info("Fetching Anthropic models...")
                # Handle different API versions
                try:
                    # Try the .models.list() method (newer Anthropic client versions)
                    response = self.anthropic_client.models.list()
                    self.available_models["anthropic"] = [model.id for model in response.data]
                except (AttributeError, TypeError):
                    # For older versions, try alternate methods or fall back to hardcoded list
                    logger.info("Using fallback list of Anthropic models")
                    self.available_models["anthropic"] = [
                        "claude-3-haiku-20240307", 
                        "claude-3-sonnet-20240229", 
                        "claude-3-opus-20240229",
                        "claude-3-5-sonnet-20240620"
                    ]
                
                logger.info(f"Found {len(self.available_models['anthropic'])} Anthropic models")
                logger.info(f"Available Anthropic models: {', '.join(self.available_models['anthropic'])}")
            except Exception as e:
                logger.error(f"Error fetching Anthropic models: {e}")
                # Fallback to known models if API fails
                self.available_models["anthropic"] = [
                    "claude-3-haiku-20240307", 
                    "claude-3-sonnet-20240229", 
                    "claude-3-opus-20240229",
                    "claude-3-5-sonnet-20240620"
                ]
        
        # Get Ollama models
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                models_data = response.json()
                self.available_models["ollama"] = [model["name"] for model in models_data.get("models", [])]
        except Exception as e:
            logger.error(f"Error fetching Ollama models: {e}")
        
        return self.available_models
    
    def get_flat_model_list(self) -> List[Dict[str, str]]:
        """
        Get a flat list of all available models with provider info.
        
        Returns:
            List of dicts with 'id', 'provider', and 'display_name' keys
        """
        models = []
        providers = self.get_available_models()
        
        for provider, provider_models in providers.items():
            for model_id in provider_models:
                models.append({
                    "id": model_id,
                    "provider": provider,
                    "display_name": f"{provider.capitalize()}: {model_id}"
                })
        
        return models
        
    def update_api_keys(self, openai_key: Optional[str] = None, anthropic_key: Optional[str] = None) -> None:
        """
        Update the API keys and reinitialize the clients.
        
        Args:
            openai_key: New OpenAI API key
            anthropic_key: New Anthropic API key
        """
        # Update OpenAI client if a key is provided
        if openai_key:
            self.openai_api_key = openai_key
            os.environ['OPENAI_API_KEY'] = openai_key
            self.openai_client = openai.OpenAI(api_key=openai_key)
            logger.info("OpenAI client updated with new API key")
        
        # Update Anthropic client if a key is provided
        if anthropic_key:
            self.anthropic_api_key = anthropic_key
            os.environ['ANTHROPIC_API_KEY'] = anthropic_key
            
            try:
                # Try the standard initialization (works for newer versions)
                self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            except TypeError as e:
                if "unexpected keyword argument 'proxies'" in str(e):
                    # If we get the proxies error, try importing Client directly
                    from anthropic import Client
                    self.anthropic_client = Client(api_key=anthropic_key)
                else:
                    raise
            
            logger.info("Anthropic client updated with new API key")
            
        # Refresh available models
        if openai_key or anthropic_key:
            self.get_available_models(refresh=True)
    
    def generate_text(self, 
                     model_id: str, 
                     provider: str, 
                     prompt: str,
                     system_prompt: Optional[str] = None,
                     temperature: float = 0.7,
                     max_tokens: Optional[int] = None) -> str:
        """
        Generate text using the specified model.
        
        Args:
            model_id: ID of the model to use
            provider: Provider name (openai, anthropic, ollama)
            prompt: The prompt to send to the model
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if provider == "openai":
            return self._generate_with_openai(model_id, prompt, system_prompt, temperature, max_tokens)
        elif provider == "anthropic":
            return self._generate_with_anthropic(model_id, prompt, system_prompt, temperature, max_tokens)
        elif provider == "ollama":
            return self._generate_with_ollama(model_id, prompt, system_prompt, temperature, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _generate_with_openai(self, 
                            model_id: str, 
                            prompt: str,
                            system_prompt: Optional[str],
                            temperature: float,
                            max_tokens: Optional[int]) -> str:
        """Generate text using OpenAI."""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized. Please provide an API key.")
        
        # Create a debug log file specifically for this call
        import json
        import time
        debug_log_file = f"openai_debug_{model_id}_{int(time.time())}.json"
        debug_data = {
            "timestamp": time.time(),
            "model_id": model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "system_prompt_length": len(system_prompt) if system_prompt else 0,
            "request": {},
            "response": {},
            "error": None
        }
        
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        
        # Prepare parameters
        params = {
            "model": model_id,
            "messages": messages
        }
        
        # Save the request params
        debug_data["request"] = {
            "model": model_id,
            "messages": [{"role": m["role"], "content": m["content"][:500] + "..." if len(m["content"]) > 500 else m["content"]} for m in messages]
        }
        
        # Only add temperature for models that support it
        if not ('o1' in model_id or 'o2' in model_id):
            params["temperature"] = temperature
            debug_data["request"]["temperature"] = temperature
        
        # Handle token limit parameters for different model versions
        if max_tokens:
            # Check the model ID to use the correct parameter
            o_models = ['o1', 'gpt-o1', 'o2', 'gpt-o2']
            
            # Determine if we have a high-capacity model to increase token limits
            high_capacity_models = [
                'gpt-4o', 'gpt-4-turbo', 'gpt-4-32k', 'gpt-4-1106-preview', 'gpt-4-vision',
                'claude-3-opus', 'claude-3-5-sonnet', 'claude-3-haiku', 
                'claude-3-sonnet', 'claude-3.5-sonnet'
            ]
            
            is_high_capacity = any(hc_model in model_id.lower() for hc_model in high_capacity_models)
            
            # Adjust max_tokens for high capacity models
            if is_high_capacity:
                logger.info(f"High capacity model detected: {model_id}. Setting higher token limit.")
                max_tokens = min(max_tokens * 2, 4000)  # Increase limit but cap at 4000
                
            if any(o_model in model_id for o_model in o_models):
                # o1/o2 models use max_completion_tokens
                logger.info(f"Using max_completion_tokens for o model: {model_id}")
                params["max_completion_tokens"] = max_tokens
            elif 'gpt-4o' in model_id:
                # gpt-4o uses max_tokens
                logger.info(f"Using max_tokens for standard model: {model_id}")
                params["max_tokens"] = max_tokens
            else:
                # For other models, try to use max_tokens parameter first
                # Skip internal validation and just try the API call
                # The newer OpenAI SDK handles parameter validation internally
                
                # For standard models, use max_tokens
                if 'gpt-4o' in model_id or not any(o_model in model_id for o_model in o_models):
                    logger.info(f"Using max_tokens for model: {model_id}")
                    params["max_tokens"] = max_tokens
                # For o1/o2 models, use max_completion_tokens
                else:
                    logger.info(f"Using max_completion_tokens for o model: {model_id}")
                    params["max_completion_tokens"] = max_tokens
        
        # Call the API with standard parameters
        try:
            logger.info(f"Calling OpenAI API with parameters: {params}")
            
            # Special handling for O1 model
            if 'o1' in model_id or 'o2' in model_id:
                logger.info("Using special handling for O-series model")
                try:
                    # Try with the new API format
                    response = self.openai_client.chat.completions.create(**params)
                    
                    # Save full response detail to the debug file
                    try:
                        debug_data["response"] = {
                            "id": response.id if hasattr(response, 'id') else None,
                            "object": response.object if hasattr(response, 'object') else None,
                            "model": response.model if hasattr(response, 'model') else None,
                            "choices": [{
                                "index": choice.index if hasattr(choice, 'index') else None,
                                "message": {
                                    "role": choice.message.role if hasattr(choice.message, 'role') else None,
                                    "content": choice.message.content if hasattr(choice.message, 'content') else None,
                                },
                                "finish_reason": choice.finish_reason if hasattr(choice, 'finish_reason') else None
                            } for choice in response.choices] if hasattr(response, 'choices') else [],
                            "usage": {
                                "prompt_tokens": response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else None,
                                "completion_tokens": response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else None,
                                "total_tokens": response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else None
                            } if hasattr(response, 'usage') else None,
                            "full_response": str(response),
                            "response_dir": str(dir(response))
                        }
                    except Exception as debug_error:
                        logger.error(f"Error saving debug data: {str(debug_error)}")
                        debug_data["response"] = {"error_saving": str(debug_error), "raw_response": str(response)}
                    
                    # Write the debug data to file
                    with open(debug_log_file, 'w') as f:
                        json.dump(debug_data, f, indent=2)
                    logger.info(f"Debug information saved to {debug_log_file}")
                    
                    # Debug response for O-series models
                    logger.info(f"O-series response: {response}")
                    logger.info(f"Response type: {type(response)}")
                    logger.info(f"Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')]}")
                    
                    # Try different ways to extract content from the response
                    if hasattr(response, 'choices') and response.choices:
                        logger.info(f"Choices: {response.choices}")
                        logger.info(f"Choices length: {len(response.choices)}")
                        
                        if hasattr(response.choices[0], 'message'):
                            logger.info(f"Message: {response.choices[0].message}")
                            logger.info(f"Message attributes: {[attr for attr in dir(response.choices[0].message) if not attr.startswith('_')]}")
                            
                            if hasattr(response.choices[0].message, 'content'):
                                content = response.choices[0].message.content
                                logger.info(f"Content: {content}")
                                logger.info(f"Content type: {type(content)}")
                                logger.info(f"Content repr: {repr(content)}")
                                
                                if content is None:
                                    # Handle None content
                                    logger.warning("Content is None for O-series model")
                                    # Add more debug info
                                    debug_data["error"] = "Content is None for O-series model"
                                    with open(debug_log_file, 'w') as f:
                                        json.dump(debug_data, f, indent=2)
                                    return f"[DEBUG] O-series model returned None content - see {debug_log_file}"
                                elif content == "":
                                    # Handle empty string content
                                    logger.warning("Content is empty string for O-series model")
                                    debug_data["error"] = "Content is empty string for O-series model"
                                    with open(debug_log_file, 'w') as f:
                                        json.dump(debug_data, f, indent=2)
                                    return f"[DEBUG] O-series model returned empty string - see {debug_log_file}"
                                
                                return content
                    
                    # If we couldn't extract content in the normal way, try alternate methods
                    logger.warning("Using fallback method to extract content from O-series response")
                    debug_data["error"] = "Couldn't extract content normally, using fallback"
                    with open(debug_log_file, 'w') as f:
                        json.dump(debug_data, f, indent=2)
                    return f"[DEBUG] Extraction fallback - see {debug_log_file}"
                except Exception as specific_error:
                    logger.error(f"O-series specific error: {str(specific_error)}")
                    # Save error information
                    debug_data["error"] = str(specific_error)
                    with open(debug_log_file, 'w') as f:
                        json.dump(debug_data, f, indent=2)
                    # Re-raise to be caught by the outer try/except
                    raise
            else:
                # Standard handling for other models
                response = self.openai_client.chat.completions.create(**params)
                # Save basic response info
                debug_data["response"] = {
                    "model": response.model if hasattr(response, 'model') else None,
                    "content": response.choices[0].message.content if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message') else None
                }
                with open(debug_log_file, 'w') as f:
                    json.dump(debug_data, f, indent=2)
                    
                if hasattr(response, 'choices') and response.choices and hasattr(response.choices[0], 'message'):
                    return response.choices[0].message.content
                else:
                    logger.error(f"Unexpected response structure: {response}")
                    debug_data["error"] = "Unexpected response structure"
                    with open(debug_log_file, 'w') as f:
                        json.dump(debug_data, f, indent=2)
                    return f"[ERROR] Unexpected response structure - see {debug_log_file}"
        except Exception as e:
            error_msg = f"Error generating text with OpenAI: {str(e)}"
            logger.error(error_msg)
            
            # Save error information to debug file
            debug_data["error"] = {
                "message": str(e),
                "type": type(e).__name__,
            }
            
            # Try to capture more details if available
            if hasattr(e, 'response'):
                debug_data["error"]["response"] = str(e.response)
                if hasattr(e.response, 'status_code'):
                    debug_data["error"]["status_code"] = e.response.status_code
                if hasattr(e.response, 'text'):
                    debug_data["error"]["response_text"] = e.response.text
                
            with open(debug_log_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
            logger.info(f"Error information saved to {debug_log_file}")
            
            # Return a more specific error message for common issues
            if "Unauthorized" in str(e) or "authentication" in str(e).lower():
                return f"OpenAI API authentication error: Please check your API key. Debug info: {debug_log_file}"
            elif "rate limit" in str(e).lower():
                return f"OpenAI API rate limit exceeded: Please try again later. Debug info: {debug_log_file}"
            elif "billing" in str(e).lower():
                return f"OpenAI API billing error: Please check your account. Debug info: {debug_log_file}"
            elif "this model is currently overloaded" in str(e).lower():
                return f"OpenAI API model overload: This model is currently overloaded. Please try again later or use a different model. Debug info: {debug_log_file}"
            elif "only supports a content format" in str(e).lower():
                return f"OpenAI API format error: This model may not support the requested format. Try a different model or approach. Debug info: {debug_log_file}"
            
            return f"{error_msg} - Debug info: {debug_log_file}"
    
  
    def _generate_with_anthropic(self, model_id: str, prompt: str, system_prompt: Optional[str], temperature: float, max_tokens: Optional[int]) -> str:
        """Generate text using Anthropic/Claude."""
        if not self.anthropic_client:
            raise ValueError("Anthropic client not initialized. Please provide an API key.")
        
        # Create a debug log file specifically for this call
        import json
        import time
        debug_log_file = f"anthropic_debug_{model_id}_{int(time.time())}.json"
        debug_data = {
            "timestamp": time.time(),
            "model_id": model_id,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "prompt_length": len(prompt),
            "system_prompt_length": len(system_prompt) if system_prompt else 0,
            "request": {},
            "response": {},
            "error": None
        }
        
        # Print a masked version of the API key for verification
        api_key = self.anthropic_api_key
        if api_key:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "Too short"
            logger.info(f"Using Anthropic API key: {masked_key}")
        else:
            logger.info("Anthropic API key is None or empty")
        
        # Call the API
        try:
            logger.info(f"Making Anthropic API request with model: {model_id}")
            
            # Try to determine which version of the Anthropic client we're using
            try:
                # Check if using new client format (messages.create)
                if hasattr(self.anthropic_client, 'messages'):
                    logger.info("Using newer Anthropic client API format (messages.create)")
                    
                    # Prepare parameters for newer client
                    params = {
                        "model": model_id,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature
                    }
                    
                    # Add system prompt if provided
                    if system_prompt:
                        params["system"] = system_prompt
                    
                    # Add max tokens if provided
                    if max_tokens:
                        params["max_tokens"] = max_tokens
                    
                    # Save request info
                    debug_data["request"] = {
                        "model": model_id,
                        "messages": [{"role": "user", "content_preview": prompt[:500] + "..." if len(prompt) > 500 else prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens if max_tokens else None,
                        "system": system_prompt[:500] + "..." if system_prompt and len(system_prompt) > 500 else system_prompt
                    }
                    
                    response = self.anthropic_client.messages.create(**params)
                    
                    # Save response info
                    debug_data["response"] = {
                        "model": response.model if hasattr(response, 'model') else None,
                        "id": response.id if hasattr(response, 'id') else None,
                        "type": response.type if hasattr(response, 'type') else None,
                        "content": [{"type": block.type, "text": block.text if hasattr(block, 'text') else None} 
                                  for block in response.content] if hasattr(response, 'content') else [],
                        "full_response": str(response)
                    }
                    
                    # Write debug info
                    with open(debug_log_file, 'w') as f:
                        json.dump(debug_data, f, indent=2)
                    logger.info(f"Anthropic debug information saved to {debug_log_file}")
                    
                    # Extract text from content blocks
                    content = response.content
                    text = ""
                    for block in content:
                        if block.type == "text":
                            text += block.text
                    
                    if not text:
                        logger.warning("Empty response from Anthropic")
                        debug_data["error"] = "Empty response (no text content)"
                        with open(debug_log_file, 'w') as f:
                            json.dump(debug_data, f, indent=2)
                        return f"[DEBUG] Empty response from Claude - see {debug_log_file}"
                    
                    return text
                
                # Check if using old client format (completion)
                elif hasattr(self.anthropic_client, 'completion') or hasattr(self.anthropic_client, 'completions'):
                    logger.info("Using older Anthropic client API format (completion)")
                    # Format for older client API
                    completion_method = getattr(self.anthropic_client, 'completion', None)
                    if completion_method is None:
                        completion_method = getattr(self.anthropic_client, 'completions', None).create
                    
                    # Format the prompt according to the older client requirements
                    # Check if we need to add HUMAN_PROMPT and AI_PROMPT
                    if hasattr(self.anthropic_client, 'HUMAN_PROMPT') and hasattr(self.anthropic_client, 'AI_PROMPT'):
                        formatted_prompt = f"{self.anthropic_client.HUMAN_PROMPT} {prompt} {self.anthropic_client.AI_PROMPT}"
                    else:
                        formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                    
                    response = completion_method(
                        prompt=formatted_prompt,
                        model=model_id,
                        temperature=temperature,
                        max_tokens_to_sample=max_tokens if max_tokens else 1000,
                        stream=False
                    )
                    
                    # Extract completion text from response
                    if hasattr(response, 'completion'):
                        return response.completion
                    else:
                        return str(response)
                
                else:
                    error_msg = "Unknown Anthropic client API format"
                    logger.error(error_msg)
                    debug_data["error"] = error_msg
                    with open(debug_log_file, 'w') as f:
                        json.dump(debug_data, f, indent=2)
                    raise ValueError(error_msg)
                    
            except Exception as api_error:
                logger.error(f"Error with API format detection: {str(api_error)}")
                debug_data["error"] = f"API format detection error: {str(api_error)}"
                with open(debug_log_file, 'w') as f:
                    json.dump(debug_data, f, indent=2)
                raise
                
        except Exception as e:
            error_msg = f"Error generating text with Anthropic: {str(e)}"
            logger.error(error_msg)
            
            # Save error details to debug file
            debug_data["error"] = {
                "message": str(e),
                "type": type(e).__name__
            }
            
            # Print more details if available
            if hasattr(e, 'status_code'):
                logger.error(f"Status code: {e.status_code}")
                debug_data["error"]["status_code"] = e.status_code
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                logger.error(f"Response text: {e.response.text}")
                debug_data["error"]["response_text"] = e.response.text
            
            with open(debug_log_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
            logger.info(f"Anthropic error information saved to {debug_log_file}")
            
            # Check for specific error types
            if "not_found_error" in str(e) and "model:" in str(e):
                # Return suggestion with fallback models
                return f"Model '{model_id}' not found. Try claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229, or claude-3-5-sonnet-20240620. Debug info: {debug_log_file}"
            
            # Rate limit handling
            if "rate_limit_error" in str(e):
                return f"Anthropic API rate limit exceeded. Please try again in a few seconds. Debug info: {debug_log_file}"
                
            # Authentication error
            if "auth" in str(e).lower() or "authenticate" in str(e).lower() or "unauthorized" in str(e).lower():
                return f"Authentication error with Anthropic API. Please check your API key. Debug info: {debug_log_file}"
            
            # Dependency issue
            if "proxies" in str(e).lower():
                return f"There appears to be an issue with the Anthropic library. Try reinstalling it with: pip install --upgrade anthropic. Debug info: {debug_log_file}"
            
            return f"{error_msg} - Debug info: {debug_log_file}"
    
    def _generate_with_ollama(self, 
                            model_id: str, 
                            prompt: str,
                            system_prompt: Optional[str],
                            temperature: float,
                            max_tokens: Optional[int]) -> str:
        """Generate text using Ollama."""
        # Build the payload
        payload = {
            "model": model_id,
            "prompt": prompt,
            "temperature": temperature
        }
        
        # Add system instruction if provided
        if system_prompt:
            payload["system"] = system_prompt
        
        # Add max tokens if provided
        if max_tokens:
            payload["num_predict"] = max_tokens
        
        # Call the API
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload)
            if response.status_code != 200:
                return f"Error generating text with Ollama: {response.status_code} - {response.text}"
            
            # Parse the response
            result = response.text
            # Ollama returns a stream of JSON objects, the last one has the full response
            lines = result.strip().split('\n')
            last_line = lines[-1]
            response_obj = json.loads(last_line)
            
            return response_obj.get("response", "")
        except Exception as e:
            return f"Error generating text with Ollama: {str(e)}"

# Initialize the LLM client
llm_client = EnhancedLLMClient()

def generate_keech_declaration(processed_data, period_type, custom_instructions=None):
    """
    Generate a Keech declaration from the processed timesheet data
    
    Args:
        processed_data: The processed timesheet data
        period_type: 'weekly' or 'monthly'
        custom_instructions: Any custom instructions for discounting time
    
    Returns:
        Dictionary containing Keech declaration data
    """
    logger.info(f"Generating Keech declaration with period_type={period_type}")
    # Ensure detailed_data exists in processed_data
    if 'detailed_data' not in processed_data or not processed_data['detailed_data']:
        logger.warning("No detailed_data found in processed_data")
        # Return an empty declaration structure
        return {
            "summary": {
                "total_hours": 0,
                "total_amount": 0,
                "period_type": period_type,
                "timekeeper_summary": None
            },
            "periods": [],
            "custom_instructions_applied": bool(custom_instructions)
        }
    
    # Extract entries from processed data
    all_entries = []
    for period_data in processed_data['detailed_data']:
        all_entries.extend(period_data['entries'])
    
    # Create summary by timekeeper if data exists
    timekeepers_summary = {}
    if all_entries and 'Timekeeper' in all_entries[0]:
        for entry in all_entries:
            timekeeper = entry.get('Timekeeper', 'Unknown')
            if timekeeper not in timekeepers_summary:
                timekeepers_summary[timekeeper] = {
                    'hours': 0,
                    'amount': 0
                }
            timekeepers_summary[timekeeper]['hours'] += entry.get('Hours', 0)
            if 'Amount' in entry:
                timekeepers_summary[timekeeper]['amount'] += entry.get('Amount', 0)
    
    # Apply custom instructions if provided
    discounted_entries = []
    if custom_instructions:
        logger.info(f"Applying custom instructions: {custom_instructions}")
        # Parse instructions - format could be "discount entries containing 'keyword' by 50%"
        instructions = parse_custom_instructions(custom_instructions)
        for entry in all_entries:
            # Make a copy of the entry
            adjusted_entry = entry.copy()
            
            # Apply discounts based on instructions
            for instruction in instructions:
                if should_apply_instruction(entry, instruction):
                    # Apply discount
                    if 'discount' in instruction:
                        discount_factor = 1 - (instruction['discount'] / 100)
                        adjusted_entry['Hours'] = entry['Hours'] * discount_factor
                        if 'Amount' in entry:
                            adjusted_entry['Amount'] = entry['Amount'] * discount_factor
                        adjusted_entry['Discounted'] = True
                        adjusted_entry['DiscountReason'] = instruction.get('reason', 'Custom instruction')
                        logger.info(f"Applied discount to entry: {entry.get('Description', '')[:30]}...")
                    # Remove entry
                    elif instruction.get('remove', False):
                        adjusted_entry['Hours'] = 0
                        if 'Amount' in entry:
                            adjusted_entry['Amount'] = 0
                        adjusted_entry['Removed'] = True
                        adjusted_entry['RemoveReason'] = instruction.get('reason', 'Custom instruction')
                        logger.info(f"Removed entry: {entry.get('Description', '')[:30]}...")
            
            discounted_entries.append(adjusted_entry)
    else:
        discounted_entries = all_entries
    
    # Summarize by period
    period_summary = {}
    for entry in discounted_entries:
        period = entry.get('Period')
        if period not in period_summary:
            period_summary[period] = {
                'hours': 0,
                'amount': 0,
                'entries': []
            }
        period_summary[period]['hours'] += entry.get('Hours', 0)
        if 'Amount' in entry:
            period_summary[period]['amount'] += entry.get('Amount', 0)
        period_summary[period]['entries'].append(entry)
    
    # Format the declaration for court submission
    declaration = {
        'summary': {
            'total_hours': sum(entry.get('Hours', 0) for entry in discounted_entries),
            'total_amount': sum(entry.get('Amount', 0) for entry in discounted_entries if 'Amount' in entry),
            'period_type': period_type,
            'timekeeper_summary': timekeepers_summary if timekeepers_summary else None
        },
        'periods': [
            {
                'period': period,
                'hours': data['hours'],
                'amount': data['amount'],
                'entries': data['entries']
            }
            for period, data in period_summary.items()
        ],
        'custom_instructions_applied': bool(custom_instructions)
    }
    
    logger.info("Keech declaration generation complete")
    return declaration

def parse_custom_instructions(instructions):
    """
    Parse the custom instructions into structured format
    
    Example instruction formats:
    - "Discount entries containing 'research' by 50%"
    - "Remove entries before 2023-01-01"
    - "Discount entries by timekeeper 'John Smith' by 25%"
    
    Returns a list of instruction objects with conditions and actions
    """
    parsed = []
    
    # Split into separate instructions
    instruction_list = instructions.split('\n')
    
    for instruction in instruction_list:
        instruction = instruction.strip()
        if not instruction:
            continue
            
        instruction_obj = {}
        
        # Check for discount instructions
        discount_match = re.search(r'discount.*by\s+(\d+)%', instruction, re.IGNORECASE)
        if discount_match:
            instruction_obj['discount'] = float(discount_match.group(1))
        
        # Check for removal instructions
        if re.search(r'remove', instruction, re.IGNORECASE):
            instruction_obj['remove'] = True
        
        # Check for conditions
        if 'containing' in instruction.lower():
            keyword = re.search(r'containing\s+[\'"](.+?)[\'"]', instruction)
            if keyword:
                instruction_obj['keyword'] = keyword.group(1)
        
        if 'before' in instruction.lower():
            date_match = re.search(r'before\s+(\d{4}-\d{2}-\d{2})', instruction)
            if date_match:
                instruction_obj['before_date'] = date_match.group(1)
        
        if 'after' in instruction.lower():
            date_match = re.search(r'after\s+(\d{4}-\d{2}-\d{2})', instruction)
            if date_match:
                instruction_obj['after_date'] = date_match.group(1)
        
        if 'timekeeper' in instruction.lower():
            timekeeper_match = re.search(r'timekeeper\s+[\'"](.+?)[\'"]', instruction)
            if timekeeper_match:
                instruction_obj['timekeeper'] = timekeeper_match.group(1)
        
        # Save the raw instruction as reason
        instruction_obj['reason'] = instruction
        
        if instruction_obj:
            parsed.append(instruction_obj)
    
    return parsed

def should_apply_instruction(entry, instruction):
    """
    Check if an instruction should be applied to an entry
    
    Args:
        entry: The timesheet entry
        instruction: The parsed instruction object
    
    Returns:
        Boolean indicating if the instruction applies
    """
    # Check keyword condition
    if 'keyword' in instruction:
        description = entry.get('Description', '')
        if instruction['keyword'].lower() not in description.lower():
            return False
    
    # Check date conditions
    if 'before_date' in instruction:
        try:
            entry_date = pd.to_datetime(entry.get('Date'))
            before_date = pd.to_datetime(instruction['before_date'])
            if entry_date >= before_date:
                return False
        except Exception as e:
            logger.warning(f"Error processing before_date condition: {str(e)}")
            return False
    
    if 'after_date' in instruction:
        try:
            entry_date = pd.to_datetime(entry.get('Date'))
            after_date = pd.to_datetime(instruction['after_date'])
            if entry_date <= after_date:
                return False
        except Exception as e:
            logger.warning(f"Error processing after_date condition: {str(e)}")
            return False
    
    # Check timekeeper condition
    if 'timekeeper' in instruction:
        entry_timekeeper = entry.get('Timekeeper', '')
        if instruction['timekeeper'].lower() not in entry_timekeeper.lower():
            return False
    
    return True

def ai_redact_timesheet(processed_data, custom_instructions=None):
    """
    Use AI to intelligently redact sensitive information from timesheet entries
    
    Args:
        processed_data: The processed timesheet data
        custom_instructions: Any custom redaction instructions provided by the user
    
    Returns:
        Redacted version of the timesheet data
    """
    logger.info("Starting AI-powered redaction process")
    
    # Check if processed_data is valid
    if not processed_data or not isinstance(processed_data, dict):
        logger.warning("Invalid processed_data provided for redaction")
        return processed_data  # Return the input as-is
    
    # Check if detailed_data exists
    if 'detailed_data' not in processed_data or not processed_data['detailed_data']:
        logger.warning("No detailed_data found in processed_data for redaction")
        return processed_data  # Return the input as-is
    
    # Check if we have any API keys available through our LLM client
    if not (llm_client.openai_api_key or llm_client.anthropic_api_key):
        logger.warning("No API keys found for OpenAI or Anthropic")
        logger.warning("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
        # Not using pattern-based redaction as requested
        # return fallback_redact_timesheet(processed_data)
        return processed_data  # Just return the original data instead
    
    # Create a deep copy to avoid modifying the original
    import copy
    redacted_data = copy.deepcopy(processed_data)
    
    # Initialize debug information container
    redacted_data['ai_debug'] = {
        'prompts': [],
        'responses': []
    }
    
    # Get all entries that need to be processed
    all_entries = []
    for period_data in redacted_data['detailed_data']:
        if 'entries' in period_data and period_data['entries']:
            all_entries.extend(period_data['entries'])
    
    # Process entries in batches, grouped by period
    period_entries = {}
    for entry in all_entries:
        period = entry.get('Period', 'unknown')
        if period not in period_entries:
            period_entries[period] = []
        period_entries[period].append(entry)
    
    # Process each period separately
    for period, entries in period_entries.items():
        logger.info(f"Processing {len(entries)} entries for period {period}")
        
        # Prepare data for AI
        entries_for_ai = []
        for i, entry in enumerate(entries):
            if not entry.get('Description'):  # Skip entries without descriptions
                continue
                
            # Create a serializable version of the entry
            serializable_entry = {
                'id': i,
                'description': entry.get('Description', ''),
                'timekeeper': entry.get('Timekeeper', ''),
                'hours': float(entry.get('Hours', 0)) if entry.get('Hours') is not None else 0
            }
            
            # Handle date properly (Timestamp objects aren't JSON serializable)
            date_value = entry.get('Date')
            if date_value is not None:
                if hasattr(date_value, 'isoformat'):  # If it's a datetime-like object
                    serializable_entry['date'] = date_value.isoformat()
                else:
                    serializable_entry['date'] = str(date_value)
            else:
                serializable_entry['date'] = ''
                
            entries_for_ai.append(serializable_entry)
        
        if not entries_for_ai:
            logger.warning(f"No valid entries to process for period {period}")
            continue
        
        # Create prompt for AI
        prompt = create_redaction_prompt(entries_for_ai, period, custom_instructions)
        
        try:
            # Log that we're about to call the AI service
            logger.info(f"Calling AI service for period {period} with {len(entries_for_ai)} entries")
            
            # Convert entries to string for logging (truncated)
            entries_str = str(entries_for_ai[:2])
            if len(entries_for_ai) > 2:
                entries_str += f"... and {len(entries_for_ai) - 2} more"
            logger.info(f"Example entries: {entries_str}")
            
            # Call the AI service
            ai_response = call_ai_service(prompt)
            
            # Log the AI response (truncated)
            if ai_response and 'choices' in ai_response and ai_response['choices']:
                content = ai_response['choices'][0].get('message', {}).get('content', '')
                logger.info(f"AI response content (truncated): {content[:100]}...")
            
            # Capture debug information
            if 'debug_info' in ai_response:
                redacted_data['ai_debug']['prompts'].append(ai_response['debug_info']['raw_prompt'])
                redacted_data['ai_debug']['responses'].append(ai_response['debug_info']['raw_response'])
            
            # Parse the AI response
            redacted_entries = parse_ai_redaction_response(ai_response, entries_for_ai)
            
            # Create a mapping from redacted entry ID to redacted content
            redaction_map = {}
            for redacted_entry in redacted_entries:
                if 'id' in redacted_entry and 'redacted_description' in redacted_entry:
                    redaction_map[redacted_entry['id']] = redacted_entry
            
            # Update the original entries with redacted text
            for i, entry in enumerate(entries):
                if i in redaction_map:
                    redacted_entry = redaction_map[i]
                    original_desc = entry.get('Description', '')
                    redacted_desc = redacted_entry['redacted_description']
                    
                    # Check if redaction actually happened
                    if original_desc != redacted_desc:
                        logger.info(f"Redaction applied to entry {i}: '{original_desc[:30]}...' -> '{redacted_desc[:30]}...'")
                    
                    entry['Description'] = redacted_desc
                    if 'redaction_reason' in redacted_entry:
                        entry['RedactionReason'] = redacted_entry['redaction_reason']
            
            logger.info(f"Successfully redacted entries for period {period}")
        
        except Exception as e:
            logger.error(f"Error in AI redaction for period {period}: {str(e)}")
            logger.error(f"Exception type: {type(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            logger.info("AI redaction failed for this period")
            
            # Not using pattern-based redaction as requested
            # logger.info("Falling back to pattern-based redaction for this period")
            # apply_pattern_redaction(entries)
    
    # Also redact raw data
    try:
        # Process raw data, which might have a different structure
        raw_entries = redacted_data.get('raw_data', [])
        if raw_entries:
            logger.info(f"Processing {len(raw_entries)} raw data entries")
            
            # Prepare data for AI
            raw_for_ai = []
            for i, entry in enumerate(raw_entries):
                if not entry.get('Description'):  # Skip entries without descriptions
                    continue
                    
                # Create a serializable version of the entry
                serializable_entry = {
                    'id': i,
                    'description': entry.get('Description', '')
                }
                
                raw_for_ai.append(serializable_entry)
            
            if raw_for_ai:
                # Log that we're about to call the AI service for raw data
                logger.info(f"Calling AI service for raw data with {len(raw_for_ai)} entries")
                
                # Create prompt for raw data
                raw_prompt = create_raw_redaction_prompt(raw_for_ai, custom_instructions)
                
                try:
                    # Call AI service for raw data
                    raw_ai_response = call_ai_service(raw_prompt)
                    
                    # Log the AI response (truncated)
                    if raw_ai_response and 'choices' in raw_ai_response and raw_ai_response['choices']:
                        content = raw_ai_response['choices'][0].get('message', {}).get('content', '')
                        logger.info(f"AI response for raw data (truncated): {content[:100]}...")
                    
                    # Capture debug information
                    if 'debug_info' in raw_ai_response:
                        redacted_data['ai_debug']['prompts'].append(raw_ai_response['debug_info']['raw_prompt'])
                        redacted_data['ai_debug']['responses'].append(raw_ai_response['debug_info']['raw_response'])
                    
                    # Parse and apply redactions to raw data
                    redacted_raw = parse_ai_redaction_response(raw_ai_response, raw_for_ai)
                    
                    # Create a mapping from redacted entry ID to redacted content
                    raw_redaction_map = {}
                    for redacted_entry in redacted_raw:
                        if 'id' in redacted_entry and 'redacted_description' in redacted_entry:
                            raw_redaction_map[redacted_entry['id']] = redacted_entry
                    
                    # Update raw entries
                    for i, entry in enumerate(raw_entries):
                        if i in raw_redaction_map:
                            redacted_entry = raw_redaction_map[i]
                            original_desc = entry.get('Description', '')
                            redacted_desc = redacted_entry['redacted_description']
                            
                            # Check if redaction actually happened
                            if original_desc != redacted_desc:
                                logger.info(f"Redaction applied to raw entry {i}: '{original_desc[:30]}...' -> '{redacted_desc[:30]}...'")
                            
                            entry['Description'] = redacted_desc
                    
                    logger.info(f"Successfully redacted raw data entries")
                except Exception as e:
                    logger.error(f"Error in AI redaction for raw data: {str(e)}")
                    logger.error(f"Exception type: {type(e)}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    logger.info("AI redaction failed for raw data")
                    # Not using pattern-based redaction as requested
                    # logger.info("Falling back to pattern-based redaction for raw data")
                    # apply_pattern_redaction(raw_entries)
    except Exception as e:
        logger.error(f"Error redacting raw data: {str(e)}")
        logger.info("Raw data redaction failed")
        # Not using pattern-based redaction as requested
        # logger.info("Applying pattern-based redaction to raw data")
        # apply_pattern_redaction(redacted_data.get('raw_data', []))
    
    logger.info("AI-powered redaction complete")
    return redacted_data

def create_redaction_prompt(entries, period, custom_instructions=None):
    """
    Create a prompt for the AI to redact sensitive information
    """
    system_content = """You are an expert legal assistant specializing in redacting privileged and confidential information from legal timesheets. 
    Your role is to carefully identify and redact sensitive information in attorney billing timesheet entries.

    EXPECTED FORMATS AND REQUIREMENTS:
    1. You MUST return your response as a valid JSON array
    2. Each object in the array MUST include:
       - "id": The numeric ID of the entry (preserve the original ID)
       - "redacted_description": The description with sensitive parts replaced with [REDACTED]
       - "redaction_reason": Only include this if you actually redacted something 
    3. Use EXACTLY the string "[REDACTED]" for redactions (not "redacted", not "<REDACTED>", etc.)
    4. Apply redactions aggressively when in doubt about the sensitivity of content
    5. Return ALL entries - even if not redacted (with original description as redacted_description)
    6. The response must be properly formatted JSON that can be parsed by JSON.parse()
    
    TYPES OF INFORMATION TO REDACT:
    1. Attorney-client privileged communications
    EXAMPLES: 
    10/7/2024	Text message communications with client regarding discovery responses and case strategy (13 messages)	Mark Piesner	0.60
    CORRECT REDACTION: 10/7/2024	Text message communications with client regarding [REDACTED]	Mark Piesner	0.60
    4/15/2024	Text message exchange with client regarding Facebook account hacking incident and potential evidence implications	Mark Piesner	0.20
    CORRECT REDACTION: 4/15/2024	Text message exchange with client regarding [REDACTED]	Mark Piesner	0.20
    2. Work product doctrine protected information
    EXAMPLES: 10/31/2024	Review of edited Declaration of Sanaz Saha supporting motion to stay; correspondence with law clerk regarding necessary revisions to declaration to ensure factual accuracy	Mark Piesner	0.50
    CORRECT REDACTION: 10/31/2024	Review of edited Declaration of Sanaz Saha supporting motion to stay; correspondence with law clerk regarding [REDACTED]	Mark Piesner	0.50
    3. Litigation strategy discussions and case strategy
    4. Settlement discussions and negotiation details
    5. Legal analysis, legal advice, and legal opinions
    6. Confidential client information
    7. References to internal discussions about legal approach
    8. Descriptions of arguments to be made in court
    9. Information about preparation for hearings, depositions, or trials
    10. Trade secrets or proprietary business information
    11. Personal identifiable information (PII)
    
    
    EXAMPLES OF ENTRIES NOT TO REDACT:
    4/17/2024	Phone conference with client regarding case strategy and upcoming proceedings	Mark Piesner	0.50
    REASON: It is general and not specific and therefore not privileged or confidential.
    4/18/2024	Review of case law and preparation of legal memorandum on the admissibility of expert testimony	Mark Piesner	0.30
    REASON: It is legal research and analysis, not privileged or confidential.
    4/18/2024	Communication with Trope and client regarding case developments and court filings	Mark Piesner	0.40
    REASON: It is general communication about case developments with opposing counsel, not privileged or confidential.
    
    RESPONSE FORMAT:
    Always respond with a valid JSON array containing objects for each entry. Example:
    [
      {
        "id": 0,
        "redacted_description": "Phone call with client regarding [REDACTED]",
        "redaction_reason": "Attorney-client privileged communication"
      },
      {
        "id": 1,
        "redacted_description": "Draft motion to dismiss",
        // No redaction_reason field since nothing was redacted
      }
    ]
    
    If you include a redaction_reason, make it short but descriptive.
    """
    
    # Add custom instructions to the system prompt if provided
    if custom_instructions:
        system_content += f"""
        
        IMPORTANT CUSTOM INSTRUCTIONS - FOLLOW THESE PRECISELY:
        {custom_instructions}
        
        These custom instructions override default behavior. Apply them exactly as specified.
        """
    
    prompt = [
        {"role": "system", "content": system_content}
    ]
    
    # Prepare the entries data
    entries_text = json.dumps(entries, indent=2)
    
    prompt.append({"role": "user", "content": 
                  f"Review and redact the following timesheet entries for period {period}.\n\n" +
                  f"ENTRIES TO REDACT:\n{entries_text}\n\n" +
                  f"YOUR RESPONSE MUST BE A VALID JSON ARRAY containing objects with 'id', 'redacted_description', and (if redaction occurred) 'redaction_reason'. " +
                  f"Use [REDACTED] for all redacted content. Return ALL entries, including unredacted ones. " +
                  f"Make sure the response can be parsed directly by JSON.parse()."
                 })
    
    return prompt

def create_raw_redaction_prompt(raw_entries, custom_instructions=None):
    """
    Create a prompt for redacting raw data entries
    """
    system_content = """You are an expert legal assistant specializing in redacting privileged and confidential information from legal documents. 
    Your role is to carefully identify and redact sensitive information in attorney billing timesheet entries.

    EXPECTED FORMATS AND REQUIREMENTS:
    1. You MUST return your response as a valid JSON array
    2. Each object in the array MUST include:
       - "id": The numeric ID of the entry (preserve the original ID)
       - "redacted_description": The description with sensitive parts replaced with [REDACTED]
       - "redaction_reason": Only include this if you actually redacted something
    3. Use EXACTLY the string "[REDACTED]" for redactions (not "redacted", not "<REDACTED>", etc.)
    4. Apply redactions aggressively when in doubt about the sensitivity of content
    5. Return ALL entries - even if not redacted
    6. The response must be properly formatted JSON that can be parsed by JSON.parse()
    
    TYPES OF INFORMATION TO REDACT:
    1. Attorney-client privileged communications
    EXAMPLES: 
    10/7/2024	Text message communications with client regarding discovery responses and case strategy (13 messages)	Mark Piesner	0.60
    CORRECT REDACTION: 10/7/2024	Text message communications with client regarding [REDACTED]	Mark Piesner	0.60
    4/15/2024	Text message exchange with client regarding Facebook account hacking incident and potential evidence implications	Mark Piesner	0.20
    CORRECT REDACTION: 4/15/2024	Text message exchange with client regarding [REDACTED]	Mark Piesner	0.20
    2. Work product doctrine protected information
    EXAMPLES: 10/31/2024	Review of edited Declaration of Sanaz Saha supporting motion to stay; correspondence with law clerk regarding necessary revisions to declaration to ensure factual accuracy	Mark Piesner	0.50
    CORRECT REDACTION: 10/31/2024	Review of edited Declaration of Sanaz Saha supporting motion to stay; correspondence with law clerk regarding [REDACTED]	Mark Piesner	0.50
    3. Litigation strategy discussions and case strategy
    4. Settlement discussions and negotiation details
    5. Legal analysis, legal advice, and legal opinions
    6. Confidential client information
    7. References to internal discussions about legal approach
    8. Descriptions of arguments to be made in court
    9. Information about preparation for hearings, depositions, or trials
    10. Trade secrets or proprietary business information
    11. Personal identifiable information (PII)
    
    EXAMPLES OF ENTRIES NOT TO REDACT:
    4/17/2024	Phone conference with client regarding case strategy and upcoming proceedings	Mark Piesner	0.50
    REASON: It is general and not specific and therefore not privileged or confidential.
    4/18/2024	Review of case law and preparation of legal memorandum on the admissibility of expert testimony	Mark Piesner	0.30
    REASON: It is legal research and analysis, not privileged or confidential.
    4/18/2024	Communication with Trope and client regarding case developments and court filings	Mark Piesner	0.40
    REASON: It is general communication about case developments with opposing counsel, not privileged or confidential.
    
    RESPONSE FORMAT:
    Always respond with a valid JSON array. Example:
    [
      {
        "id": 0,
        "redacted_description": "Phone call with client regarding [REDACTED]",
        "redaction_reason": "Attorney-client privileged communication"
      },
      {
        "id": 1,
        "redacted_description": "Draft motion to dismiss"
        // No redaction_reason field since nothing was redacted
      }
    ]
    """
    
    # Add custom instructions to the system prompt if provided
    if custom_instructions:
        system_content += f"""
        
        IMPORTANT CUSTOM INSTRUCTIONS - FOLLOW THESE PRECISELY:
        {custom_instructions}
        
        These custom instructions override default behavior. Apply them exactly as specified.
        """
    
    prompt = [
        {"role": "system", "content": system_content}
    ]
    
    # Prepare the entries data
    entries_text = json.dumps(raw_entries, indent=2)
    
    prompt.append({"role": "user", "content": 
                  f"Redact these entries carefully.\n\n" +
                  f"ENTRIES TO REDACT:\n{entries_text}\n\n" +
                  f"YOUR RESPONSE MUST BE A VALID JSON ARRAY containing objects with 'id', 'redacted_description', and (if redaction occurred) 'redaction_reason'. " +
                  f"Use [REDACTED] for all redacted content. Return ALL entries, including unredacted ones. " +
                  f"Do not include any text before or after the JSON array. The response must be valid JSON that can be parsed by JSON.parse()."
                 })
    
    return prompt

def call_ai_service(prompt):
    """
    Call the AI service with the given prompt using the EnhancedLLMClient
    
    This IMPORTANT function calls the AI service and returns the response.
    CRUCIAL: Debug problems with AI responses here.
    """
    # First, try to determine the provider and model to use from environment variables
    # Get model information from environment or use defaults
    model_id = os.getenv('AI_MODEL', 'gpt-4')  # Use environment variable or default to gpt-4
    provider = os.getenv('AI_PROVIDER', 'openai')  # Use environment variable or default to openai
    
    logger.info(f"Initial model selection from environment: provider={provider}, model={model_id}")
    
    # Log available API keys
    logger.info(f"OpenAI API key available: {bool(llm_client.openai_api_key)}")
    logger.info(f"Anthropic API key available: {bool(llm_client.anthropic_api_key)}")
    
    # Make sure we're using a provider that we have an API key for
    if provider == 'openai' and not llm_client.openai_api_key:
        if llm_client.anthropic_api_key:
            logger.warning("Switching to Anthropic because OpenAI API key is not available")
            provider = 'anthropic'
            # Use a default Claude model
            model_id = 'claude-3-opus-20240229'
        else:
            logger.warning("No API keys available for AI services")
            raise ValueError("No API keys available for AI services")
    
    if provider == 'anthropic' and not llm_client.anthropic_api_key:
        if llm_client.openai_api_key:
            logger.warning("Switching to OpenAI because Anthropic API key is not available")
            provider = 'openai'
            # Use a default GPT-4 model
            model_id = 'gpt-4'
        else:
            logger.warning("No API keys available for AI services")
            raise ValueError("No API keys available for AI services")
    
    # Normalize model names for popular AI services only when needed
    if provider == 'openai':
        # Only normalize incomplete model names without changing valid model IDs
        if model_id == '4':
            logger.info("Normalizing model name to full 'gpt-4'")
            model_id = 'gpt-4'
        elif model_id == '3.5':
            logger.info("Normalizing model name to full 'gpt-3.5-turbo'")
            model_id = 'gpt-3.5-turbo'
        # Log the model being used - don't change valid model IDs
        logger.info(f"Using OpenAI model: {model_id}")
    
    # Use default models if none specified
    if not model_id:
        # Use defaults based on provider
        if provider == 'openai':
            # Prefer gpt-4 over newer models for reliability with the current code
            model_id = 'gpt-4'
        elif provider == 'anthropic':
            model_id = 'claude-3-5-sonnet-20240620' if 'claude-3-5-sonnet-20240620' in [m['id'] for m in llm_client.get_flat_model_list() if m['provider'] == 'anthropic'] else 'claude-3-sonnet-20240229'
        else:
            # Default to OpenAI
            provider = 'openai'
            model_id = 'gpt-4'
            
    logger.info(f"Using AI provider: {provider}, model: {model_id}")
    
    # Extract system prompt and user prompt from the messages format
    system_prompt = None
    user_prompt = ""
    
    for message in prompt:
        if message['role'] == 'system':
            system_prompt = message['content']
        elif message['role'] == 'user':
            user_prompt = message['content']
    
    # If no user prompt was found but there are messages, use the last message
    if not user_prompt and prompt:
        user_prompt = prompt[-1]['content']
    
    # Store the raw prompt for debugging
    raw_prompt = {
        "system": system_prompt,
        "user": user_prompt,
        "provider": provider,
        "model": model_id
    }
    
    # Log a summary of the prompt (truncated for readability)
    if system_prompt:
        logger.info(f"System prompt: {system_prompt[:100]}...")
    logger.info(f"User prompt: {user_prompt[:100]}...")
    
    try:
        # Call the LLM client to generate the response
        logger.info(f"Calling {provider} API with model {model_id}")
        
        # Add timeout to prevent hanging
        import concurrent.futures
        import threading
        
        # Create a thread pool executor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Set parameters based on model type - some models don't support temperature
            def call_llm():
                if 'o1' in model_id or 'o2' in model_id:
                    logger.info(f"Omitting temperature parameter for o-series model: {model_id}")
                    return llm_client.generate_text(
                        model_id=model_id,
                        provider=provider,
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        max_tokens=2000
                    )
                else:
                    logger.info(f"Using temperature parameter for model: {model_id}")
                    return llm_client.generate_text(
                        model_id=model_id,
                        provider=provider,
                        prompt=user_prompt,
                        system_prompt=system_prompt,
                        temperature=0.1,  # Low temperature for consistent redactions
                        max_tokens=2000
                    )
            
            # Determine timeout based on model - high capacity models get more time
            high_capacity_models = [
                'gpt-4o', 'gpt-4-turbo', 'gpt-4-32k', 'gpt-4-1106-preview', 'gpt-4-vision',
                'claude-3-opus', 'claude-3-5-sonnet', 'claude-3-haiku', 
                'claude-3-sonnet', 'claude-3.5-sonnet'
            ]
            
            is_high_capacity = any(hc_model in model_id.lower() for hc_model in high_capacity_models)
            timeout_seconds = 240 if is_high_capacity else 120  # 4 minutes for high capacity models, 2 minutes otherwise
            
            logger.info(f"Using timeout of {timeout_seconds} seconds for {model_id}")
            
            # Submit the task to the executor with the calculated timeout
            future = executor.submit(call_llm)
            try:
                # Wait for the result with a timeout
                response_text = future.result(timeout=timeout_seconds)
                
                # Add additional logging to understand empty responses
                logger.info(f"API call successful. Response type: {type(response_text)}")
                logger.info(f"Response length: {len(str(response_text)) if response_text else 0}")
                logger.info(f"Is response None? {response_text is None}")
                logger.info(f"Is response empty string? {response_text == ''}")
                if response_text is None or response_text == '':
                    logger.error(f"Empty response received from {provider}/{model_id}. This might be a model limitation.")
                    # Default response for debugging - will be handled later in the code
                    response_text = "[DEBUG] Empty response from model"
            except concurrent.futures.TimeoutError:
                logger.error(f"API call to {provider}/{model_id} timed out after {timeout_seconds} seconds")
                raise TimeoutError(f"API call to {provider}/{model_id} timed out after {timeout_seconds} seconds - please try again with a different model")
        
        # Log a detailed snippet of the response for debugging
        logger.info(f"Received response from {provider}: {response_text[:300]}...")
        
        # Log full response for better debugging
        logger.info(f"FULL RESPONSE: {response_text}")
        
        # Check if the response contains redaction indicators
        has_redacted_tag = '[REDACTED]' in response_text
        has_redacted_word = 'redact' in response_text.lower()
        logger.info(f"Response contains [REDACTED] tag: {has_redacted_tag}")
        logger.info(f"Response contains 'redact' word: {has_redacted_word}")
        
        # Format the response to match the expected structure
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": response_text
                    }
                }
            ],
            "debug_info": {
                "raw_prompt": raw_prompt,
                "raw_response": response_text
            }
        }
        
        return mock_response
    except Exception as e:
        logger.error(f"AI service error with {provider}/{model_id}: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        if hasattr(e, 'status_code'):
            logger.error(f"Status code: {e.status_code}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            logger.error(f"Response text: {e.response.text}")
        raise Exception(f"AI service error with {provider}/{model_id}: {str(e)}")

def parse_ai_redaction_response(ai_response, original_entries):
    """
    Parse the AI response and extract redacted entries
    """
    try:
        logger.info(f"Parsing AI response for {len(original_entries)} entries")
        # Extract content - handle both string responses and structured responses
        content = ''
        
        # Check if we have a structured response with choices
        if isinstance(ai_response, dict) and 'choices' in ai_response:
            message_content = ai_response.get('choices', [{}])[0].get('message', {}).get('content', '')
            if message_content:
                content = message_content
        # If ai_response is a string, use it directly
        elif isinstance(ai_response, str):
            content = ai_response
            
        # Check for empty or error content
        if not content:
            logger.warning("Empty content in AI response")
            logger.warning("This might be an issue with the specific model - trying to handle empty response")
            
            # For debugging purposes, create a special array that shows the issue
            if "[DEBUG]" in str(ai_response) or "[FAILED]" in str(ai_response):
                logger.warning("Debug message detected in response")
                debug_message = str(ai_response)
                # Check if using O1 model
                if 'o1' in os.getenv('AI_MODEL', '').lower() or 'o2' in os.getenv('AI_MODEL', '').lower():
                    logger.warning("O-series model detected. These models may return empty responses for redaction.")
                    # Create a special debug response
                    return [{"id": i, 
                             "redacted_description": entry.get('description', '') + " [O-SERIES MODEL LIMITATION]", 
                             "redaction_reason": "O-series model returned empty response - try GPT-4 or Claude"
                            } for i, entry in enumerate(original_entries)]
            
            # Return original descriptions instead (unchanged)
            return [{"id": i, "redacted_description": entry.get('description', '')} for i, entry in enumerate(original_entries)]
            
        # Check for debug/failed flags
        if isinstance(content, str) and ("[DEBUG]" in content or "[FAILED]" in content or "[ERROR]" in content):
            logger.warning(f"Debug/error content detected in AI response: {content}")
            # Create a debug response to indicate the issue
            return [{"id": i, 
                     "redacted_description": entry.get('description', '') + f" [{content}]", 
                     "redaction_reason": "Model returned debug/error message"
                    } for i, entry in enumerate(original_entries)]
            
        # Check for error messages but don't trigger on square brackets which might be JSON
        if content.startswith("Error generating text") or (not content.startswith("[") and "error" in content.lower()):
            logger.warning(f"Error content in AI response: {content[:100]}...")
            # Not using pattern-based redaction as requested
            # logger.info("Falling back to pattern-based redaction")
            # return _create_pattern_based_redactions(original_entries)
            # Return original descriptions with error info
            return [{"id": i, 
                     "redacted_description": entry.get('description', '') + " [ERROR: API response error]", 
                     "redaction_reason": f"Error response: {content[:50]}..."
                    } for i, entry in enumerate(original_entries)]
            
        # Let's try to be more flexible about parsing JSON responses - if we see '[' it might be JSON
            
        # Log the first part of the content for debugging
        logger.info(f"AI response content preview: {content[:200]}...")
        
        # Multiple strategies to extract JSON
        
        # If we see the content starts with an array bracket, it might be clean JSON already
        if content.strip().startswith('['):
            try:
                direct_json = json.loads(content.strip())
                if isinstance(direct_json, list):
                    logger.info(f"Successfully parsed direct JSON with {len(direct_json)} entries")
                    
                    # Validate and return
                    valid_entries = []
                    for entry in direct_json:
                        if 'id' in entry and ('redacted_description' in entry or 'description' in entry):
                            if 'redacted_description' not in entry and 'description' in entry:
                                entry['redacted_description'] = entry['description']
                            valid_entries.append(entry)
                            
                    logger.info(f"Validated {len(valid_entries)} out of {len(direct_json)} entries")
                    if valid_entries:
                        logger.info(f"Sample parsed entry: {str(valid_entries[0])}")
                        return valid_entries
            except json.JSONDecodeError:
                logger.info("Direct JSON parsing failed, trying to extract JSON from content")
        
        # Strategy 1: Find JSON array within response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            logger.info(f"Found JSON-like content from position {json_start} to {json_end}")
            json_str = content[json_start:json_end]
            
            # Log the entire JSON string for debugging
            logger.info(f"Extracted JSON: {json_str[:200]}...")
            
            # Attempt to clean up JSON before parsing
            # Replace JavaScript/Python-style trailing commas which are invalid in JSON
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            
            # Remove any comments (JSON doesn't support them)
            json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)
            # Try to remove block comments, but be careful with this
            try:
                json_str = re.sub(r'/\*.*?\*/', '', json_str, flags=re.DOTALL)
            except Exception as regex_error:
                logger.warning(f"Error in regex comment removal: {str(regex_error)}")
            
            try:
                redacted_entries = json.loads(json_str)
                
                # Validate the response format
                if isinstance(redacted_entries, list):
                    logger.info(f"Successfully parsed {len(redacted_entries)} redacted entries from JSON")
                    
                    # Check if entries have the expected format
                    valid_entries = []
                    for entry in redacted_entries:
                        # Check if this entry has the required fields
                        if 'id' in entry:
                            # If it has a redacted_description field, use it
                            if 'redacted_description' in entry:
                                valid_entries.append(entry)
                            # If it doesn't have redacted_description but has description, create one
                            elif 'description' in entry:
                                entry['redacted_description'] = entry['description']
                                valid_entries.append(entry)
                    
                    # Log a sample of the parsed data
                    if valid_entries:
                        logger.info(f"Sample parsed entry: {str(valid_entries[0])}")
                        logger.info(f"Validated {len(valid_entries)} out of {len(redacted_entries)} entries")
                    
                    return valid_entries if valid_entries else redacted_entries
                else:
                    logger.warning(f"Parsed JSON is not a list, but a {type(redacted_entries)}")
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
                logger.error(f"JSON string attempt: {json_str[:100]}...")
                
                # Try to clean up JSON more aggressively
                try:
                    # Remove any non-JSON elements before the first [
                    clean_json = json_str[json_str.find('['):]
                    # Remove any non-JSON elements after the last ]
                    clean_json = clean_json[:clean_json.rfind(']')+1]
                    # Fix common JSON errors
                    clean_json = re.sub(r',\s*}', '}', clean_json)  # Remove trailing commas in objects
                    clean_json = re.sub(r',\s*]', ']', clean_json)  # Remove trailing commas in arrays
                    
                    logger.info(f"Attempting to parse cleaned JSON: {clean_json[:100]}...")
                    redacted_entries = json.loads(clean_json)
                    
                    if isinstance(redacted_entries, list):
                        logger.info(f"Successfully parsed {len(redacted_entries)} entries from cleaned JSON")
                        return redacted_entries
                except Exception as clean_error:
                    logger.error(f"Error parsing cleaned JSON: {str(clean_error)}")
                    # Continue to other strategies
        
        # Strategy 2: Look for JSON in code blocks
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        code_blocks = re.findall(code_block_pattern, content)
        
        if code_blocks:
            logger.info(f"Found {len(code_blocks)} code blocks in response")
            for block in code_blocks:
                try:
                    json_data = json.loads(block)
                    if isinstance(json_data, list):
                        logger.info(f"Successfully parsed {len(json_data)} entries from code block")
                        return json_data
                except json.JSONDecodeError:
                    continue
        
        # Strategy 3: Try to parse the entire content as JSON
        try:
            json_data = json.loads(content)
            if isinstance(json_data, list):
                logger.info(f"Successfully parsed {len(json_data)} entries from full content")
                return json_data
            elif isinstance(json_data, dict) and 'entries' in json_data and isinstance(json_data['entries'], list):
                logger.info(f"Found entries in JSON dictionary, {len(json_data['entries'])} items")
                return json_data['entries']
        except json.JSONDecodeError:
            logger.info("Could not parse entire content as JSON")
        
        # Final strategy: Use AI-informed pattern-based redaction
        logger.info("Using AI-informed pattern-based fallback for redaction")
        
        # Look for redacted content patterns in the AI response
        redaction_patterns = []
        
        # Try to extract redaction patterns from AI response
        # Look for common patterns like "[REDACTED]" or phrases like "confidential", "privileged", etc.
        
        # First, extract any directly mentioned terms to redact
        mentioned_terms = set()
        
        # Look for phrases like "should be redacted" or "needs redaction"
        redaction_mention_patterns = [
            r'should (?:be )?redact\w*[:\s]+([^\.]+)',
            r'needs? redact\w*[:\s]+([^\.]+)',
            r'redact(?:ed|ing)?[:\s]+([^\.]+)',
            r'redact(?:ed|ing)?.*?"([^"]+)"',
            r"redact(?:ed|ing)?.*?'([^']+)'",
            r'\[REDACTED\].*?(?:because|due to|for)[:\s]+([^\.]+)'
        ]
        
        for pattern in redaction_mention_patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                if match.group(1).strip():
                    mentioned_terms.add(match.group(1).strip())
        
        # Create regex patterns from directly mentioned terms
        if mentioned_terms:
            for term in mentioned_terms:
                # Clean up term and make it into a regex pattern
                term = term.strip().lower()
                if len(term) > 3:  # Avoid very short terms
                    redaction_patterns.append(re.escape(term))
            logger.info(f"Extracted specific redaction terms: {mentioned_terms}")
        
        # Add common legal privilege patterns
        redaction_patterns.extend([
            r'attorney.client',
            r'work\s+product',
            r'legal\s+advice',
            r'legal\s+strategy',
            r'litigation\s+strategy',
            r'settlement\s+discussion',
            r'negotiation\s+detail',
            r'internal\s+counsel',
            r'legal\s+analysis',
            r'confidential',
            r'privileged',
            r'counsel\s+regarding',
            r'attorney\s+advice',
            r'internal\s+discussion'
        ])
        
        # AGGRESSIVE REDACTION: Extract any sentences that mention redaction  
        sentences_with_redaction = []
        for sentence in re.split(r'[.!?]', content):
            if re.search(r'redact', sentence, re.IGNORECASE):
                # Look for any terms in quotes
                quoted_terms = re.findall(r'"([^"]+)"', sentence)
                for term in quoted_terms:
                    if len(term) > 3:
                        redaction_patterns.append(re.escape(term))
        
        # Combine all patterns with word boundaries for more precise matching
        pattern_strings = [fr'\b{pattern}\b' for pattern in redaction_patterns]
        combined_pattern = re.compile('|'.join(pattern_strings), re.IGNORECASE)
        
        logger.info(f"Using redaction patterns: {redaction_patterns[:5]}{'...' if len(redaction_patterns) > 5 else ''}")
        
        # Create redacted entries with our enhanced pattern
        redacted_entries = []
        for i, entry in enumerate(original_entries):
            description = entry.get('description', '')
            
            # Skip empty descriptions
            if not description:
                continue
                
            # Apply pattern-based redaction
            redacted_description = combined_pattern.sub('[REDACTED]', description)
            
            # Also search for any phrase that might need redaction
            # This is a more aggressive approach based on what the AI might have tried to communicate
            special_terms = [
                "strategy", "advice", "analysis", "discussion", "internal", 
                "privileged", "evaluation", "assessment", "counsel", "settlement",
                "confidential", "work product", "attorney"
            ]
            
            # Add more aggressive redaction for entries with sensitive terms
            for term in special_terms:
                if term.lower() in description.lower():
                    # Redact phrases around the sensitive term
                    phrase_pattern = re.compile(fr'.{{0,20}}{re.escape(term)}.{{0,30}}', re.IGNORECASE)
                    redacted_description = phrase_pattern.sub('[REDACTED]', redacted_description)
            
            # Add to redacted entries
            if redacted_description != description:
                # Count how many redactions were made
                redaction_count = redacted_description.count('[REDACTED]')
                
                redacted_entries.append({
                    'id': i,
                    'redacted_description': redacted_description,
                    'redaction_reason': f'AI-informed redaction ({redaction_count} instances)'
                })
            else:
                # For entries that weren't redacted, keep the original
                redacted_entries.append({
                    'id': i,
                    'redacted_description': description
                })
        
        logger.info(f"Created {len(redacted_entries)} AI-informed pattern-based redacted entries")
        return redacted_entries
    
    except Exception as e:
        logger.error(f"Error parsing AI response: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Not using pattern-based redaction as requested
        # Fall back to standard pattern-based redaction
        # return _create_pattern_based_redactions(original_entries)
        # Return original descriptions instead
        return [{"id": i, "redacted_description": entry.get('description', '')} for i, entry in enumerate(original_entries)]

def _create_pattern_based_redactions(original_entries):
    """Helper function to create pattern-based redactions as a fallback"""
    logger.info("Using basic pattern-based fallback for redaction")
    
    # Patterns that might indicate privileged information - using very aggressive patterns
    patterns = [
        # Legal privilege patterns
        r'confidential',
        r'privileged',
        r'attorney.client',
        r'attorney',
        r'counsel',
        r'work product',
        
        # Strategy patterns
        r'strategy',
        r'advic[e|ing]',
        r'discuss(?:ion|ed)',
        r'internal',
        
        # Litigation patterns
        r'negotiation',
        r'settlement',
        r'litigation',
        r'argument',
        r'position',
        r'legal',
        r'analysis',
        
        # Communication patterns
        r'regarding.*approach',
        r'regarding.*position',
        r'review.*strategy',
        r'communicate.*strategy',
        r'discuss.*approach',
        
        # Privacy patterns
        r'personal',
        r'private',
        r'sensitive',
        
        # Extra aggressive patterns - common legal action words followed by context
        r'draft.*regarding',
        r'prepare.*for',
        r'review.*for',
        r'analyze.*case',
        r'plan.*for'
    ]
    
    # Convert patterns to word boundary patterns for more precision
    bounded_patterns = [rf'\b{pattern}\b' for pattern in patterns]
    combined_pattern = re.compile('|'.join(bounded_patterns), re.IGNORECASE)
    
    # Create redacted entries
    redacted_entries = []
    for i, entry in enumerate(original_entries):
        description = entry.get('description', '')
        
        # Skip empty descriptions
        if not description:
            continue
            
        # Apply pattern-based redaction
        redacted_description = combined_pattern.sub('[REDACTED]', description)
        
        # Redact surrounding context for any matches (20 chars before, 30 after)
        for pattern in patterns:
            pattern_re = re.compile(rf'.{{0,20}}\b{pattern}\b.{{0,30}}', re.IGNORECASE)
            redacted_description = pattern_re.sub('[REDACTED]', redacted_description)
        
        # Add to redacted entries
        if redacted_description != description:
            # Count how many redactions were made
            redaction_count = redacted_description.count('[REDACTED]')
            
            redacted_entries.append({
                'id': i,
                'redacted_description': redacted_description,
                'redaction_reason': f'Pattern-based redaction ({redaction_count} instances)'
            })
        else:
            # For entries that weren't redacted, keep the original
            redacted_entries.append({
                'id': i,
                'redacted_description': description
            })
    
    logger.info(f"Created {len(redacted_entries)} basic pattern-based redacted entries")
    return redacted_entries

def analyze_discount_entries(processed_data, custom_instructions=None):
    """
    Use AI to analyze timesheet entries and identify entries that should be discounted
    based on project issues or inefficiencies
    
    Args:
        processed_data: The processed timesheet data
        custom_instructions: Any custom discount instructions provided by the user
    
    Returns:
        Analyzed timesheet data with discount recommendations
    """
    logger.info("Starting AI-powered discount analysis")
    
    # Check if processed_data is valid
    if not processed_data or not isinstance(processed_data, dict):
        logger.warning("Invalid processed_data provided for discount analysis")
        return processed_data  # Return the input as-is
    
    # Check if detailed_data exists
    if 'detailed_data' not in processed_data or not processed_data['detailed_data']:
        logger.warning("No detailed_data found in processed_data for discount analysis")
        return processed_data  # Return the input as-is
    
    # Check if we have any API keys available through our LLM client
    if not (llm_client.openai_api_key or llm_client.anthropic_api_key):
        logger.warning("No API keys found for OpenAI or Anthropic, cannot perform discount analysis")
        logger.warning("Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variables")
        return processed_data
    
    # Create a deep copy to avoid modifying the original
    import copy
    analyzed_data = copy.deepcopy(processed_data)
    
    # Initialize debug information container
    analyzed_data['ai_debug'] = {
        'prompts': [],
        'responses': []
    }
    
    # Get all entries that need to be processed
    all_entries = []
    for period_data in analyzed_data['detailed_data']:
        if 'entries' in period_data and period_data['entries']:
            all_entries.extend(period_data['entries'])
    
    # Process entries in batches, grouped by period
    period_entries = {}
    for entry in all_entries:
        period = entry.get('Period', 'unknown')
        if period not in period_entries:
            period_entries[period] = []
        period_entries[period].append(entry)
    
    # Process each period separately
    for period, entries in period_entries.items():
        logger.info(f"Analyzing {len(entries)} entries for period {period}")
        
        # Prepare data for AI
        entries_for_ai = []
        for i, entry in enumerate(entries):
            if not entry.get('Description'):  # Skip entries without descriptions
                continue
                
            # Create a serializable version of the entry
            serializable_entry = {
                'id': i,
                'description': entry.get('Description', ''),
                'timekeeper': entry.get('Timekeeper', ''),
                'hours': float(entry.get('Hours', 0)) if entry.get('Hours') is not None else 0
            }
            
            # Handle date properly (Timestamp objects aren't JSON serializable)
            date_value = entry.get('Date')
            if date_value is not None:
                if hasattr(date_value, 'isoformat'):  # If it's a datetime-like object
                    serializable_entry['date'] = date_value.isoformat()
                else:
                    serializable_entry['date'] = str(date_value)
            else:
                serializable_entry['date'] = ''
                
            entries_for_ai.append(serializable_entry)
        
        if not entries_for_ai:
            logger.warning(f"No valid entries to process for period {period}")
            continue
        
        # Create prompt for AI
        prompt = create_discount_analysis_prompt(entries_for_ai, period, custom_instructions)
        
        try:
            # Call the AI service
            logger.info(f"Calling AI service for discount analysis on period {period} with {len(entries_for_ai)} entries")
            ai_response = call_ai_service(prompt)
            
            # Capture debug information
            if 'debug_info' in ai_response:
                analyzed_data['ai_debug']['prompts'].append(ai_response['debug_info']['raw_prompt'])
                analyzed_data['ai_debug']['responses'].append(ai_response['debug_info']['raw_response'])
            
            # Parse the AI response
            discount_recommendations = parse_discount_response(ai_response, entries_for_ai)
            
            # Create a mapping from entry ID to discount recommendation
            discount_map = {}
            for discount_rec in discount_recommendations:
                if 'id' in discount_rec:
                    discount_map[discount_rec['id']] = discount_rec
            
            # Update the original entries with discount recommendations
            for i, entry in enumerate(entries):
                if i in discount_map:
                    recommendation = discount_map[i]
                    
                    # Add discount recommendation fields
                    if 'should_discount' in recommendation:
                        entry['ShouldDiscount'] = recommendation['should_discount']
                    
                    if 'discount_percentage' in recommendation:
                        entry['DiscountPercentage'] = recommendation['discount_percentage']
                    
                    if 'discount_reason' in recommendation:
                        entry['DiscountReason'] = recommendation['discount_reason']
                    
                    # Calculate discounted amount if applicable
                    if recommendation.get('should_discount', False) and 'discount_percentage' in recommendation:
                        original_hours = float(entry.get('Hours', 0))
                        discount_pct = float(recommendation['discount_percentage']) / 100.0
                        discounted_hours = original_hours * (1 - discount_pct)
                        
                        entry['OriginalHours'] = original_hours
                        entry['DiscountedHours'] = discounted_hours
                        
                        # Also calculate amount if available
                        if 'Rate' in entry and entry['Rate']:
                            rate = float(entry['Rate'])
                            entry['DiscountedAmount'] = discounted_hours * rate
            
            logger.info(f"Successfully analyzed entries for period {period}")
        
        except Exception as e:
            logger.error(f"Error in AI discount analysis for period {period}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Calculate total hours and amount before and after discount
    original_hours = sum(entry.get('Hours', 0) for entry in all_entries)
    discounted_hours = sum(
        entry.get('DiscountedHours', entry.get('Hours', 0)) 
        for entry in all_entries
    )
    
    # Calculate savings
    hours_saved = original_hours - discounted_hours
    percentage_saved = (hours_saved / original_hours * 100) if original_hours > 0 else 0
    
    # Add summary to the result
    analyzed_data['discount_summary'] = {
        'original_hours': original_hours,
        'discounted_hours': discounted_hours,
        'hours_saved': hours_saved,
        'percentage_saved': percentage_saved,
        'entries_discounted': sum(1 for entry in all_entries if entry.get('ShouldDiscount', False))
    }
    
    logger.info("AI-powered discount analysis complete")
    logger.info(f"Discount summary: Original hours: {original_hours}, Discounted hours: {discounted_hours}, " +
               f"Hours saved: {hours_saved}, Percentage saved: {percentage_saved:.2f}%")
    
    return analyzed_data

def create_discount_analysis_prompt(entries, period, custom_instructions=None):
    """
    Create a prompt for the AI to analyze entries for potential discounts
    """
    system_content = """Here is the description of the legal work:

<legal_work_description>
{{LEGAL_WORK_DESCRIPTION}} I want to discount for the ex-parte I filed regarding the guardian ad litem, it should have been an application, not an ex parte, so part of that work was wasted, I then filed it as an RFO which was still wrong, so we should discount for those errors. and then in the Application for GAL the Memo of P&A had some bad citations so I had to request the court disregard it, so let's discount for the time on the memo of P&a.  also, I never filed the responses to the RFOs - 4 of them filed by OC regarding teh request for custody evaluation and the other three he filed. so we should discount time spent on those projects and communications around them, since the work was not used.
</legal_work_description>

You are an experienced legal billing specialist tasked with reviewing a description of legal work and identifying areas where discounts should be applied due to errors or unused work. Your goal is to provide a detailed breakdown of recommended discounts based on the information provided.

Please analyze the description and identify areas where discounts should be applied. Focus on the following issues mentioned:

1. The ex-parte filing that should have been an application
2. The subsequent incorrect RFO filing
3. The Application for Guardian Ad Litem (GAL) with bad citations in the Memo of P&A
4. The unfinished responses to the RFOs filed by opposing counsel

Instructions:
1. Carefully read and analyze the legal work description.
2. For each identified issue, provide:
   a. A brief explanation of the error or unused work
   b. An estimate of the time that should be discounted
   c. A justification for your recommendation
3. Wrap each discount item in the following structure:
   <discount_item>
     <explanation>[Brief explanation of the error or unused work]</explanation>
     <estimated_time>[Estimate of time to be discounted]</estimated_time>
     <justification>[Justification for the recommendation]</justification>
   </discount_item>
4. After analyzing all issues, provide a summary of the total recommended discount:
   <total_discount>
     <total_time>[Estimated total time to be discounted]</total_time>
     <overall_justification>[Brief overall justification for the recommendation]</overall_justification>
   </total_discount>

Before providing your final structured response, wrap your analysis in <discount_analysis> tags inside your thinking block. In this section:
1. List out each issue mentioned in the prompt
2. For each issue, quote the relevant parts of the legal work description
3. Estimate the time spent on each incorrect or unused work
4. Provide reasoning for why this work should be discounted

This will help ensure a thorough interpretation of the information.

Begin your response with "Based on the provided description of legal work, I recommend the following discounts:"

Your final output should consist only of the structured response with discount items and total discount, and should not duplicate or rehash any of the work you did in the thinking block.
    """
    
    # Add custom instructions to the system prompt if provided
    if custom_instructions:
        system_content += f"""
        
        IMPORTANT CUSTOM INSTRUCTIONS - FOLLOW THESE PRECISELY:
        {custom_instructions}
        
        These custom instructions override default behavior. Apply them exactly as specified.
        """
    
    prompt = [
        {"role": "system", "content": system_content}
    ]
    
    # Prepare the entries data
    entries_text = json.dumps(entries, indent=2)
    
    prompt.append({"role": "user", "content": 
                  f"Analyze the following timesheet entries for period {period} and identify which entries should be discounted.\n\n" +
                  f"ENTRIES TO ANALYZE:\n{entries_text}\n\n" +
                  f"YOUR RESPONSE MUST BE A VALID JSON ARRAY containing objects with 'id', 'should_discount', and if applicable 'discount_percentage' and 'discount_reason'." +
                  f"Make sure the response can be parsed directly by JSON.parse()."
                 })
    
    return prompt

def parse_discount_response(ai_response, original_entries):
    """
    Parse the AI response for discount recommendations
    """
    try:
        logger.info(f"Parsing AI response for discount analysis with {len(original_entries)} entries")
        
        # Extract content from response
        content = ''
        if isinstance(ai_response, dict) and 'choices' in ai_response:
            message_content = ai_response.get('choices', [{}])[0].get('message', {}).get('content', '')
            if message_content:
                content = message_content
        elif isinstance(ai_response, str):
            content = ai_response
        
        # Check for empty content
        if not content:
            logger.warning("Empty content in AI discount response")
            return create_empty_discount_recommendations(original_entries)
        
        # Try direct JSON parsing first if content starts with [
        if content.strip().startswith('['):
            try:
                discount_recs = json.loads(content.strip())
                if isinstance(discount_recs, list):
                    logger.info(f"Successfully parsed direct JSON with {len(discount_recs)} discount recommendations")
                    return discount_recs
            except json.JSONDecodeError:
                logger.info("Direct JSON parsing failed, trying to extract JSON from content")
        
        # Find JSON array within response
        json_start = content.find('[')
        json_end = content.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            logger.info(f"Found JSON-like content from position {json_start} to {json_end}")
            json_str = content[json_start:json_end]
            
            # Clean up JSON
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            
            try:
                discount_recs = json.loads(json_str)
                
                if isinstance(discount_recs, list):
                    logger.info(f"Successfully parsed {len(discount_recs)} discount recommendations from JSON")
                    return discount_recs
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error: {str(e)}")
        
        # Try to extract JSON from code blocks or other formats
        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        code_blocks = re.findall(code_block_pattern, content)
        
        if code_blocks:
            for block in code_blocks:
                try:
                    discount_recs = json.loads(block)
                    if isinstance(discount_recs, list):
                        logger.info(f"Successfully parsed {len(discount_recs)} recommendations from code block")
                        return discount_recs
                except json.JSONDecodeError:
                    continue
        
        # If all parsing fails, create empty recommendations
        logger.warning("Failed to parse discount recommendations from AI response")
        return create_empty_discount_recommendations(original_entries)
        
    except Exception as e:
        logger.error(f"Error parsing discount response: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return create_empty_discount_recommendations(original_entries)

def create_empty_discount_recommendations(original_entries):
    """
    Create empty discount recommendations if parsing fails
    """
    return [{"id": i, "should_discount": False} for i in range(len(original_entries))]

def apply_pattern_redaction(entries):
    """
    Apply pattern-based redaction to entries (fallback method)
    """
    # Check if entries is valid
    if not entries or not isinstance(entries, list):
        logger.warning("Invalid entries provided for pattern redaction")
        return  # Nothing to do
    
    # Use the same aggressive patterns as in _create_pattern_based_redactions
    patterns = [
        # Legal privilege patterns
        r'confidential',
        r'privileged',
        r'attorney.client',
        r'attorney',
        r'counsel',
        r'work product',
        
        # Strategy patterns
        r'strategy',
        r'advic[e|ing]',
        r'discuss(?:ion|ed)',
        r'internal',
        
        # Litigation patterns
        r'negotiation',
        r'settlement',
        r'litigation',
        r'argument',
        r'position',
        r'legal',
        r'analysis',
        
        # Communication patterns
        r'regarding.*approach',
        r'regarding.*position',
        r'review.*strategy',
        r'communicate.*strategy',
        r'discuss.*approach',
        
        # Privacy patterns
        r'personal',
        r'private',
        r'sensitive',
        
        # Extra aggressive patterns - common legal action words followed by context
        r'draft.*regarding',
        r'prepare.*for',
        r'review.*for',
        r'analyze.*case',
        r'plan.*for'
    ]
    
    # Convert patterns to word boundary patterns for more precision
    bounded_patterns = [rf'\b{pattern}\b' for pattern in patterns]
    combined_pattern = re.compile('|'.join(bounded_patterns), re.IGNORECASE)
    
    redaction_count = 0
    for entry in entries:
        if not entry or not isinstance(entry, dict):
            continue  # Skip invalid entries
            
        description = entry.get('Description', '')
        if description and isinstance(description, str):
            # First apply the combined pattern
            redacted_desc = combined_pattern.sub('[REDACTED]', description)
            
            # Then redact surrounding context for any matches (20 chars before, 30 after)
            for pattern in patterns:
                pattern_re = re.compile(rf'.{{0,20}}\b{pattern}\b.{{0,30}}', re.IGNORECASE)
                redacted_desc = pattern_re.sub('[REDACTED]', redacted_desc)
            
            # If any redactions were made, update the entry
            if redacted_desc != description:
                redaction_count += 1
                entry['Description'] = redacted_desc
                
                # Count how many redactions were made in this entry
                instance_count = redacted_desc.count('[REDACTED]')
                entry['RedactionReason'] = f'Pattern-based redaction ({instance_count} instances)'
    
    logger.info(f"Applied pattern-based redaction to {redaction_count} entries")

def fallback_redact_timesheet(processed_data):
    """
    Fallback redaction method using pattern matching
    """
    logger.info("Using fallback pattern-based redaction")
    
    # Check if processed_data is valid
    if not processed_data or not isinstance(processed_data, dict):
        logger.warning("Invalid processed_data provided for fallback redaction")
        return processed_data  # Return the input as-is
    
    # Check if detailed_data exists
    if 'detailed_data' not in processed_data or not processed_data['detailed_data']:
        logger.warning("No detailed_data found in processed_data for fallback redaction")
        return processed_data  # Return the input as-is
    
    # Create a deep copy to avoid modifying the original
    import copy
    redacted_data = copy.deepcopy(processed_data)
    
    # Redact detailed data
    for period_data in redacted_data['detailed_data']:
        if 'entries' in period_data and period_data['entries']:
            apply_pattern_redaction(period_data['entries'])
    
    # Redact raw data
    apply_pattern_redaction(redacted_data.get('raw_data', []))
    
    logger.info("Fallback redaction complete")
    return redacted_data