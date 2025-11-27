"""
Docling Service Client - Use this to integrate with other services

Example usage:
    from client import DoclingClient

    client = DoclingClient("http://docs.jeffemmett.com")

    # Extract from URL
    result = await client.extract_url("https://example.com/doc.pdf")

    # Extract with summarization
    result = await client.extract_url(
        "https://example.com/doc.pdf",
        summarize=True,
        summarize_style="bullet_points"
    )

    # Transcribe audio
    result = await client.transcribe_url("https://example.com/audio.mp3")
"""

import httpx
import base64
from pathlib import Path
from typing import Optional, Dict, Any, Literal


OutputFormat = Literal["markdown", "json", "text", "html"]
SummarizeStyle = Literal["concise", "detailed", "bullet_points", "technical", "eli5"]


class DoclingClient:
    """Async client for Docling Service"""

    def __init__(self, base_url: str = "http://localhost:8081", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def health(self) -> dict:
        """Check service health"""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/health")
            return resp.json()

    async def stats(self) -> dict:
        """Get processing statistics"""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self.base_url}/stats")
            return resp.json()

    async def extract_url(
        self,
        url: str,
        output_format: OutputFormat = "markdown",
        summarize: bool = False,
        summarize_style: SummarizeStyle = "concise",
        index_to_search: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Extract content from a URL"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/extract",
                json={
                    "url": url,
                    "output_format": output_format,
                    "summarize": summarize,
                    "summarize_style": summarize_style,
                    "index_to_search": index_to_search,
                    "metadata": metadata,
                },
            )
            return resp.json()

    async def extract_file(
        self,
        file_path: str,
        output_format: OutputFormat = "markdown",
        summarize: bool = False,
        summarize_style: SummarizeStyle = "concise",
        index_to_search: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Extract content from a local file"""
        path = Path(file_path)
        content = base64.b64encode(path.read_bytes()).decode()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/extract",
                json={
                    "base64_content": content,
                    "filename": path.name,
                    "output_format": output_format,
                    "summarize": summarize,
                    "summarize_style": summarize_style,
                    "index_to_search": index_to_search,
                    "metadata": metadata,
                },
            )
            return resp.json()

    async def extract_bytes(
        self,
        content: bytes,
        filename: str,
        output_format: OutputFormat = "markdown",
        summarize: bool = False,
        summarize_style: SummarizeStyle = "concise",
        index_to_search: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """Extract content from bytes"""
        b64_content = base64.b64encode(content).decode()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/extract",
                json={
                    "base64_content": b64_content,
                    "filename": filename,
                    "output_format": output_format,
                    "summarize": summarize,
                    "summarize_style": summarize_style,
                    "index_to_search": index_to_search,
                    "metadata": metadata,
                },
            )
            return resp.json()

    async def transcribe_url(
        self,
        url: str,
        language: Optional[str] = None,
        summarize: bool = False,
        summarize_style: SummarizeStyle = "concise",
    ) -> dict:
        """Transcribe audio from URL"""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/transcribe",
                json={
                    "url": url,
                    "language": language,
                    "summarize": summarize,
                    "summarize_style": summarize_style,
                },
            )
            return resp.json()

    async def transcribe_file(
        self,
        file_path: str,
        language: Optional[str] = None,
        summarize: bool = False,
        summarize_style: SummarizeStyle = "concise",
    ) -> dict:
        """Transcribe audio from local file"""
        path = Path(file_path)
        content = base64.b64encode(path.read_bytes()).decode()

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/transcribe",
                json={
                    "base64_content": content,
                    "language": language,
                    "summarize": summarize,
                    "summarize_style": summarize_style,
                },
            )
            return resp.json()

    async def preview_url(self, url: str) -> dict:
        """Quick preview of URL content"""
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/url/preview",
                json=url,
            )
            return resp.json()


# Sync wrapper for convenience
class DoclingClientSync:
    """Synchronous client wrapper"""

    def __init__(self, base_url: str = "http://localhost:8081", timeout: float = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def extract_url(self, url: str, **kwargs) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/extract",
                json={"url": url, **kwargs},
            )
            return resp.json()

    def transcribe_url(self, url: str, **kwargs) -> dict:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                f"{self.base_url}/transcribe",
                json={"url": url, **kwargs},
            )
            return resp.json()
