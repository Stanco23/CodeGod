"""
Local Model Executor for Master AI
Supports running large local models (70B+) via Ollama, vLLM, or llama.cpp
"""

import asyncio
import json
import logging
from typing import Optional, Dict, Any, List
import subprocess
import os
from enum import Enum

logger = logging.getLogger(__name__)


class ModelBackend(Enum):
    """Supported local model backends"""
    OLLAMA = "ollama"
    VLLM = "vllm"
    LLAMACPP = "llamacpp"
    API = "api"  # Claude/GPT-4


class LocalModelExecutor:
    """
    Execute prompts using local large models for Master AI

    Supports models like:
    - Llama 3.1 70B/405B
    - Mixtral 8x7B/8x22B
    - Qwen 2.5 72B
    - DeepSeek Coder 33B
    """

    def __init__(
        self,
        model_name: str = "llama3.1:70b",
        backend: ModelBackend = ModelBackend.OLLAMA,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None
    ):
        self.model_name = model_name
        self.backend = backend
        self.api_key = api_key
        self.api_base = api_base

        if backend == ModelBackend.OLLAMA:
            self._ensure_ollama_model()
        elif backend == ModelBackend.API:
            self._setup_api_client()

    def _ensure_ollama_model(self):
        """Ensure Ollama model is available"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if self.model_name not in result.stdout:
                # Model not found - don't auto-pull, give clear error
                raise ValueError(
                    f"Model '{self.model_name}' not found in Ollama.\n\n"
                    f"Please install it first:\n"
                    f"  ollama pull {self.model_name}\n\n"
                    f"Model sizes:\n"
                    f"  - 1B-3B models: 1-2GB download, 2-4GB RAM\n"
                    f"  - 7B-8B models: 4-5GB download, 8GB+ RAM\n"
                    f"  - 70B models: 40GB download, 48GB+ RAM\n\n"
                    f"Or choose a different model with:\n"
                    f"  ./codegod --model <model-name>\n\n"
                    f"Available models: ollama list"
                )
            else:
                logger.info(f"Model {self.model_name} already available")

        except FileNotFoundError:
            logger.error("Ollama not found. Please install from https://ollama.com/")
            raise
        except ValueError:
            # Re-raise our custom error
            raise
        except Exception as e:
            logger.error(f"Failed to check model availability: {e}")
            raise

    def _setup_api_client(self):
        """Setup API client for Claude/GPT-4"""
        if "claude" in self.model_name.lower():
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
            self.api_type = "anthropic"
        elif "gpt" in self.model_name.lower():
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
            self.api_type = "openai"
        else:
            raise ValueError(f"Unknown API model: {self.model_name}")

    async def execute(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        response_format: Optional[str] = None
    ) -> str:
        """
        Execute prompt with local or API model

        Args:
            prompt: User prompt
            system_prompt: System/role prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            response_format: Expected format (e.g., "json")

        Returns:
            Model response
        """
        if self.backend == ModelBackend.OLLAMA:
            return await self._execute_ollama(prompt, system_prompt, max_tokens, temperature)
        elif self.backend == ModelBackend.API:
            return await self._execute_api(prompt, system_prompt, max_tokens, temperature)
        elif self.backend == ModelBackend.VLLM:
            return await self._execute_vllm(prompt, system_prompt, max_tokens, temperature)
        else:
            raise NotImplementedError(f"Backend {self.backend} not implemented")

    async def _execute_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Execute using Ollama"""
        try:
            # Build full prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            else:
                full_prompt = prompt

            # Use Ollama API via subprocess
            # In production, use ollama Python library for better performance
            process = await asyncio.create_subprocess_exec(
                "ollama", "run", self.model_name,
                "--verbose",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(full_prompt.encode()),
                timeout=600  # 10 minutes for complex reasoning
            )

            if process.returncode != 0:
                logger.error(f"Ollama execution failed: {stderr.decode()}")
                raise RuntimeError(f"Model execution failed: {stderr.decode()}")

            response = stdout.decode().strip()
            return response

        except asyncio.TimeoutError:
            logger.error("Model execution timed out")
            raise
        except Exception as e:
            logger.error(f"Ollama execution error: {e}")
            raise

    async def _execute_api(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Execute using Claude/GPT-4 API"""
        try:
            if self.api_type == "anthropic":
                messages = [{"role": "user", "content": prompt}]

                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt if system_prompt else "",
                    messages=messages
                )

                return response.content[0].text

            elif self.api_type == "openai":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"API execution error: {e}")
            raise

    async def _execute_vllm(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float
    ) -> str:
        """Execute using vLLM server"""
        # TODO: Implement vLLM support
        raise NotImplementedError("vLLM backend not yet implemented")


class ModelSelector:
    """
    Automatically selects best available model based on configuration
    """

    @staticmethod
    def get_best_model(
        prefer_local: bool = True,
        api_key: Optional[str] = None,
        min_parameters: int = 70  # Billion parameters
    ) -> LocalModelExecutor:
        """
        Get best available model

        Args:
            prefer_local: Prefer local models over API
            api_key: API key for Claude/GPT-4
            min_parameters: Minimum model size in billions

        Returns:
            Configured LocalModelExecutor
        """

        if prefer_local:
            # Try local models in order of preference (large to small)
            local_models = [
                # Large models (70B+)
                "llama3.1:70b",      # Meta Llama 3.1 70B
                "qwen2.5:72b",       # Qwen 2.5 72B
                "mixtral:8x7b",      # Mixtral 8x7B
                # Medium models (7B-33B)
                "deepseek-coder:33b", # DeepSeek Coder 33B
                "llama3.1:8b",       # Llama 3.1 8B
                "mistral:7b",        # Mistral 7B
                "qwen2.5:7b",        # Qwen 2.5 7B
                # Small models (1B-3B)
                "phi3:mini",         # Phi-3 Mini 3.8B
                "llama3.2:3b",       # Llama 3.2 3B
                "qwen2.5:3b",        # Qwen 2.5 3B
                "gemma2:2b",         # Gemma 2 2B
                "qwen2.5:1.5b",      # Qwen 2.5 1.5B
                "llama3.2:1b",       # Llama 3.2 1B
            ]

            # Check which models are available
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                available = result.stdout

                for model in local_models:
                    if model in available:
                        logger.info(f"Using local model: {model}")
                        return LocalModelExecutor(
                            model_name=model,
                            backend=ModelBackend.OLLAMA
                        )

                # No models available - give helpful error
                raise ValueError(
                    "No Ollama models found. Please install a model:\n\n"
                    "For small/fast (1-4GB RAM):\n"
                    "  ollama pull qwen2.5:1.5b\n"
                    "  ollama pull llama3.2:3b\n\n"
                    "For medium (8GB+ RAM):\n"
                    "  ollama pull llama3.1:8b\n"
                    "  ollama pull mistral:7b\n\n"
                    "For large/best quality (48GB+ RAM):\n"
                    "  ollama pull llama3.1:70b\n\n"
                    "Or use API models with --prefer-api"
                )

            except FileNotFoundError:
                logger.warning("Ollama not found. Falling back to API models.")

        # Fall back to API models
        if api_key:
            # Check which API service the key is for
            if api_key.startswith("sk-ant-"):
                logger.info("Using Claude Sonnet 4.5 via API")
                return LocalModelExecutor(
                    model_name="claude-sonnet-4.5",
                    backend=ModelBackend.API,
                    api_key=api_key
                )
            elif api_key.startswith("sk-"):
                logger.info("Using GPT-4 via API")
                return LocalModelExecutor(
                    model_name="gpt-4-turbo-preview",
                    backend=ModelBackend.API,
                    api_key=api_key
                )

        raise ValueError(
            "No suitable model available. Either:\n"
            "1. Install Ollama and pull a model: ollama pull llama3.1:70b\n"
            "2. Provide an API key in .env: ANTHROPIC_API_KEY or OPENAI_API_KEY"
        )


def get_master_model(model_name: Optional[str] = None, prefer_local: Optional[bool] = None) -> LocalModelExecutor:
    """
    Get Master AI model based on environment configuration

    Priority:
    1. model_name parameter (highest - from --model flag)
    2. MASTER_MODEL environment variable
    3. Auto-select best available model

    Args:
        model_name: Specific model to use (from --model flag)
        prefer_local: Whether to prefer local models over API

    Returns:
        Configured LocalModelExecutor
    """
    # Get prefer_local setting
    if prefer_local is None:
        prefer_local = os.getenv("PREFER_LOCAL", "true").lower() == "true"

    # Determine which model to use
    # Priority: parameter > env var > auto
    master_model = model_name or os.getenv("MASTER_MODEL", "auto")

    # If specific model requested
    if master_model != "auto":
        if master_model.startswith("claude-") or master_model.startswith("gpt-"):
            # API model
            api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(f"API key required for {master_model}")

            return LocalModelExecutor(
                model_name=master_model,
                backend=ModelBackend.API,
                api_key=api_key
            )
        else:
            # Local model
            return LocalModelExecutor(
                model_name=master_model,
                backend=ModelBackend.OLLAMA
            )

    # Auto-select best model
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    return ModelSelector.get_best_model(
        prefer_local=prefer_local,
        api_key=api_key
    )


# Testing
if __name__ == "__main__":
    import asyncio

    async def test():
        # Test local model
        executor = LocalModelExecutor(
            model_name="llama3.1:70b",
            backend=ModelBackend.OLLAMA
        )

        response = await executor.execute(
            prompt="Write a Python function to calculate fibonacci numbers.",
            system_prompt="You are a helpful coding assistant.",
            max_tokens=500
        )

        print("Response:", response)

    asyncio.run(test())
