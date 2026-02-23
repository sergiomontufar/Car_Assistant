"""Simple Ollama client helper."""

from __future__ import annotations

import json
from typing import List, Optional

import requests


class OllamaClient:
    def __init__(
        self,
        url: str = "http://localhost:11434/api/chat",
        model: str = "gemma3:270m",
        stream: bool = True,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.url = url
        self.model = model
        self.stream = stream
        self.system_prompt = system_prompt

    def _build_messages(self, user_text: str) -> List[dict]:
        messages: List[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    def query(self, user_text: str) -> str:
        payload = {
            "model": self.model,
            "messages": self._build_messages(user_text),
            "stream": self.stream,
        }
        print("-> Sending user text to Ollama:\n", user_text)

        response = requests.post(self.url, json=payload, stream=self.stream, timeout=120)
        if response.status_code != 200:
            print("Error from Ollama:", response.status_code, response.text)
            return ""

        response_text = ""
        if self.stream:
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line.decode("utf-8"))
                if chunk.get("done"):
                    break
                message = chunk.get("message", {})
                content = message.get("content")
                if content:
                    response_text += content
        else:
            data = response.json()
            message = data.get("message", {})
            response_text = message.get("content", "")

        return response_text.strip()


__all__ = ["OllamaClient"]
