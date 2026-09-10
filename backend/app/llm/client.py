"""
Configurable LLM Client.

Supports Hugging Face Inference API / Serverless Router, Ollama, and a
deterministic extractive fallback with robust error handling.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM client supporting Hugging Face, Ollama, and local fallback.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 45.0,
    ):
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or (settings.HUGGINGFACE_MODEL if self.provider == "huggingface" else settings.LLM_MODEL)
        self.api_key = api_key if api_key is not None else settings.HUGGINGFACE_API_KEY
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.timeout = timeout

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        """
        Generate a text response given a prompt and optional system instructions.
        """
        if self.provider == "huggingface" and self.api_key:
            try:
                return self._call_huggingface(prompt, system_instruction, temperature, max_tokens)
            except Exception as e:
                logger.warning(f"Hugging Face API call failed ({e}); falling back to local extractor.")

        if self.provider == "ollama":
            try:
                return self._call_ollama(prompt, system_instruction, temperature)
            except Exception as e:
                logger.warning(f"Ollama call failed ({e}); falling back.")

        # Fallback heuristic generator
        return self._local_fallback(prompt, system_instruction)

    def _call_huggingface(
        self,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Call Hugging Face Serverless Inference / Router API (OpenAI-compatible format or standard)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Hugging Face Router / v1 / chat / completions endpoint
        router_url = f"https://router.huggingface.co/hf-inference/models/{self.model}/v1/chat/completions"
        legacy_url = f"https://api-inference.huggingface.co/models/{self.model}"

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": max(0.01, temperature),
            "max_tokens": max_tokens,
        }

        with httpx.Client(timeout=self.timeout) as client:
            # Current Hugging Face router endpoint.
            current_router_url = "https://router.huggingface.co/v1/chat/completions"
            router_errors = []
            for url in (current_router_url, router_url):
                try:
                    response = client.post(url, headers=headers, json=payload)
                    if response.status_code == 200:
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        if content and content.strip():
                            return content.strip()
                    router_errors.append(f"{url}: {response.status_code}")
                except Exception as exc:
                    router_errors.append(f"{url}: {exc}")

            # Fallback to direct raw input format
            raw_payload = {
                "inputs": f"{system_instruction or ''}\n\nUser: {prompt}\nAssistant:",
                "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
            }
            resp = client.post(legacy_url, headers=headers, json=raw_payload)
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list) and len(result) > 0:
                    text = result[0].get("generated_text", "")
                    if "Assistant:" in text:
                        return text.split("Assistant:")[-1].strip()
                    return text.strip()
                elif isinstance(result, dict) and "generated_text" in result:
                    return result["generated_text"].strip()
            raise RuntimeError(
                f"HF API failed ({'; '.join(router_errors)}); legacy status {resp.status_code}: {resp.text[:300]}"
            )

    def _call_ollama(
        self,
        prompt: str,
        system_instruction: Optional[str],
        temperature: float,
    ) -> str:
        """Call local Ollama instance."""
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_instruction or "",
            "stream": False,
            "options": {"temperature": temperature},
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(url, json=payload)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            raise RuntimeError(f"Ollama API error: {response.text}")

    def _local_fallback(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Return a useful grounded answer when the configured provider is unavailable."""
        query_match = re.search(r"=== USER QUESTION ===\s*(.+?)(?:\n\n===|$)", prompt, re.DOTALL)
        query = query_match.group(1).strip() if query_match else "the question"
        evidence = re.findall(
            r"--- EVIDENCE #\d+ \(Page (\d+) \| Section: '([^']*)' \| ID: ([^ ]+) ---\n([\s\S]*?)(?=\n\n--- EVIDENCE|\n\n===|$)",
            prompt,
        )
        if not evidence:
            return "The configured language model is unavailable and no grounded evidence was retrieved."

        terms = {word.lower() for word in re.findall(r"[A-Za-z0-9]{3,}", query)}
        ranked = sorted(
            evidence,
            key=lambda item: sum(term in item[3].lower() for term in terms),
            reverse=True,
        )
        bullets = []
        for page, section, chunk_id, text in ranked[:4]:
            cleaned = " ".join(text.split())
            if cleaned:
                bullets.append(f"- {cleaned[:420]} [Page {page}, {section}]" )
        return f"Based on the retrieved paper evidence, the answer is summarized below:\n\n{chr(10).join(bullets)}"
