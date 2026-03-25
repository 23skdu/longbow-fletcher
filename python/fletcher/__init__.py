"""Fletcher Python SDK - Async embedding client."""

import asyncio
import aiohttp
from typing import List, Optional, Dict, Any


class FletcherClient:
    """Async client for Fletcher embedding server."""

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def connect(self):
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self.timeout)

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None

    def _get_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = self.api_key
        return headers

    async def embed(self, text: str, model: str = "fletcher-embed") -> List[float]:
        """Generate embedding for a single text."""
        if not self._session:
            await self.connect()

        async with self._session.post(
            f"{self.base_url}/v1/embeddings",
            json={"input": text, "model": model},
            headers=self._get_headers(),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["data"][0]["embedding"]

    async def embed_batch(
        self, texts: List[str], model: str = "fletcher-embed"
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not self._session:
            await self.connect()

        async with self._session.post(
            f"{self.base_url}/v1/embeddings/batch",
            json={"inputs": texts, "model": model},
            headers=self._get_headers(),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            dim = len(data["data"][0]["embedding"]) if data["data"] else 0
            embeddings = []
            for item in data["data"]:
                embeddings.append(item["embedding"])
            return embeddings

    async def rerank(
        self, query: str, documents: List[str], top_n: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Rerank documents based on query relevance."""
        if not self._session:
            await self.connect()

        payload = {"query": query, "documents": documents}
        if top_n:
            payload["top_n"] = top_n

        async with self._session.post(
            f"{self.base_url}/v1/rerank",
            json=payload,
            headers=self._get_headers(),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["results"]

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        if not self._session:
            await self.connect()

        async with self._session.get(
            f"{self.base_url}/v1/models/list",
            headers=self._get_headers(),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data.get("data", [])


def embed_sync(client: FletcherClient, text: str) -> List[float]:
    """Sync wrapper for embed."""
    return asyncio.run(client.embed(text))


def embed_batch_sync(client: FletcherClient, texts: List[str]) -> List[List[float]]:
    """Sync wrapper for embed_batch."""
    return asyncio.run(client.embed_batch(texts))
