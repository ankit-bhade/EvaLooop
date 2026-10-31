"""Client for models served by an OpenAI-compatible endpoint, such as OpenRouter.

Builds on OpenAILLM and adds what a full evaluation run needs from a hosted API:

  - batch_generate sends requests concurrently, so the loop runner uses its
    batch mode (one cycle of 378 tasks takes minutes instead of ~25 minutes).
  - Endpoint failures raise ModelUnavailableError instead of returning an
    "Error generating response" string, which the loop would otherwise test as
    code and count as a robustness failure.
  - extra_body is sent with every request (e.g. OpenRouter provider pinning).
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from openai import OpenAI

from .base import ModelUnavailableError
from .closed_source import OpenAILLM

logger = logging.getLogger(__name__)

# Prefixes of the strings OpenAILLM.generate returns when a request fails.
# (Its "Still received empty response" result is the model's own output, not an
# endpoint failure, so it is left to the tests like any other answer.)
ERROR_PREFIXES = ("Error generating response",)


class OpenAICompatibleLLM(OpenAILLM):
    """OpenAILLM with concurrent batching and fail-loudly endpoint errors."""

    def __init__(self, model_config: Dict[str, Any]):
        if not model_config.get("api_key"):
            raise ValueError(
                f"No API key for {model_config.get('base_url')}. Set the environment variable named by api_key_env."
            )
        super().__init__(model_config)
        # A stalled provider must fail within minutes, not the SDK's default 10 min x 3 attempts
        self.request_timeout = model_config.get("request_timeout", 90.0)
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url,
                             timeout=self.request_timeout, max_retries=1)
        self.max_concurrency = model_config.get("max_concurrency", 16)
        self.extra_body = model_config.get("extra_body")
        # Extra attempts on top of OpenAILLM's own retries, with growing waits
        self.endpoint_retries = model_config.get("endpoint_retries", 3)
        self.endpoint_retry_delay = model_config.get("endpoint_retry_delay", 10.0)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate one response; raise ModelUnavailableError if the endpoint keeps failing."""
        if self.extra_body:
            kwargs.setdefault("extra_body", self.extra_body)

        for attempt in range(self.endpoint_retries + 1):
            output = super().generate(prompt, **kwargs)
            if not output.startswith(ERROR_PREFIXES):
                return output
            if attempt < self.endpoint_retries:
                delay = self.endpoint_retry_delay * (attempt + 1)
                logger.warning(f"Endpoint error, retrying in {delay:.0f}s: {output[:200]}")
                time.sleep(delay)

        raise ModelUnavailableError(f"{self.model_id} failed after {self.endpoint_retries + 1} attempts: {output}")

    def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for many prompts concurrently, in input order."""
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as pool:
            return list(pool.map(lambda prompt: self.generate(prompt, **kwargs), prompts))
