"""
Docling Service - Document extraction and processing for the AI stack

Integrates with:
- AI Orchestrator (Ollama) for summarization
- RunPod Whisper for audio transcription
- Semantic Search for indexing extracted content
"""

import os
import asyncio
import tempfile
import base64
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

# Docling imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

# Config from environment
AI_ORCHESTRATOR_URL = os.getenv("AI_ORCHESTRATOR_URL", "http://ai-orchestrator:8080")
SEMANTIC_SEARCH_URL = os.getenv("SEMANTIC_SEARCH_URL", "http://semantic-search:8000")
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_WHISPER_ENDPOINT = os.getenv("RUNPOD_WHISPER_ENDPOINT", "lrtisuv8ixbtub")

# Supported formats
DOCUMENT_FORMATS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt", ".epub"}
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"}
AUDIO_FORMATS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}

app = FastAPI(
    title="Docling Service",
    description="Document extraction and processing service using Docling",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document converter with optimized settings
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.do_table_structure = True

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# Track processing stats
stats = {
    "documents_processed": 0,
    "pages_extracted": 0,
    "audio_transcribed": 0,
    "urls_fetched": 0,
    "errors": 0,
}


class OutputFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    TEXT = "text"
    HTML = "html"


class SummarizeStyle(str, Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"
    TECHNICAL = "technical"
    ELI5 = "eli5"  # Explain like I'm 5


class ExtractRequest(BaseModel):
    """Request to extract content from a URL or base64-encoded file"""
    url: Optional[HttpUrl] = None
    base64_content: Optional[str] = None
    filename: Optional[str] = None
    output_format: OutputFormat = OutputFormat.MARKDOWN
    summarize: bool = False
    summarize_style: SummarizeStyle = SummarizeStyle.CONCISE
    index_to_search: bool = False
    metadata: Optional[Dict[str, Any]] = None


class TranscribeRequest(BaseModel):
    """Request to transcribe audio"""
    url: Optional[HttpUrl] = None
    base64_content: Optional[str] = None
    language: Optional[str] = None  # Auto-detect if not specified
    summarize: bool = False
    summarize_style: SummarizeStyle = SummarizeStyle.CONCISE


class BatchExtractRequest(BaseModel):
    """Batch extraction request"""
    items: List[ExtractRequest]


class ExtractionResult(BaseModel):
    """Result of document extraction"""
    success: bool
    source: str
    content: Optional[str] = None
    format: OutputFormat
    metadata: Dict[str, Any] = {}
    summary: Optional[str] = None
    indexed: bool = False
    error: Optional[str] = None


# ============== Helper Functions ==============

def get_file_extension(filename: str) -> str:
    """Get lowercase file extension"""
    return Path(filename).suffix.lower()


def generate_doc_id(source: str, content: str) -> str:
    """Generate a unique document ID"""
    hash_input = f"{source}:{content[:1000]}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


async def fetch_url_content(url: str) -> tuple[bytes, str]:
    """Fetch content from URL, return bytes and detected filename"""
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()

        # Try to get filename from headers or URL
        content_disposition = resp.headers.get("content-disposition", "")
        if "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[1].strip('"\'')
        else:
            filename = url.split("/")[-1].split("?")[0] or "document"

        return resp.content, filename


async def transcribe_audio_runpod(audio_data: bytes, language: Optional[str] = None) -> dict:
    """Transcribe audio using RunPod Whisper endpoint"""
    if not RUNPOD_API_KEY:
        raise HTTPException(status_code=500, detail="RunPod API key not configured")

    # Convert audio to base64
    audio_base64 = base64.b64encode(audio_data).decode()

    payload = {
        "input": {
            "audio_base64": audio_base64,
        }
    }
    if language:
        payload["input"]["language"] = language

    async with httpx.AsyncClient(timeout=300) as client:
        # Submit job
        resp = await client.post(
            f"https://api.runpod.ai/v2/{RUNPOD_WHISPER_ENDPOINT}/run",
            headers={
                "Authorization": f"Bearer {RUNPOD_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        result = resp.json()

        if "error" in result:
            raise HTTPException(status_code=500, detail=f"RunPod error: {result['error']}")

        job_id = result.get("id")
        if not job_id:
            raise HTTPException(status_code=500, detail="No job ID returned from RunPod")

        # Poll for completion
        for _ in range(120):  # Max 10 minutes
            await asyncio.sleep(5)
            status_resp = await client.get(
                f"https://api.runpod.ai/v2/{RUNPOD_WHISPER_ENDPOINT}/status/{job_id}",
                headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
            )
            status_data = status_resp.json()

            if status_data.get("status") == "COMPLETED":
                return status_data.get("output", {})
            elif status_data.get("status") in ["FAILED", "CANCELLED"]:
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed: {status_data.get('error', 'Unknown error')}"
                )

        raise HTTPException(status_code=504, detail="Transcription timed out")


async def summarize_with_ollama(content: str, style: SummarizeStyle) -> str:
    """Summarize content using AI Orchestrator (Ollama)"""
    style_prompts = {
        SummarizeStyle.CONCISE: "Provide a concise 2-3 sentence summary of the following content:",
        SummarizeStyle.DETAILED: "Provide a detailed summary of the following content, covering all main points:",
        SummarizeStyle.BULLET_POINTS: "Summarize the following content as bullet points:",
        SummarizeStyle.TECHNICAL: "Provide a technical summary of the following content, focusing on key technical details:",
        SummarizeStyle.ELI5: "Explain the following content in simple terms that a child could understand:",
    }

    prompt = f"{style_prompts[style]}\n\n{content[:8000]}"  # Limit content for context window

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{AI_ORCHESTRATOR_URL}/api/generate/text",
                json={
                    "prompt": prompt,
                    "model": "llama3.2",
                    "max_tokens": 1024,
                    "priority": "low",  # Use free Ollama
                },
            )
            result = resp.json()
            return result.get("response", "")
        except Exception as e:
            return f"[Summarization failed: {str(e)}]"


async def index_to_semantic_search(
    doc_id: str,
    content: str,
    source: str,
    metadata: Dict[str, Any],
) -> bool:
    """Index document to semantic search service"""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{SEMANTIC_SEARCH_URL}/index",
                json={
                    "id": doc_id,
                    "content": content,
                    "metadata": {
                        "source": source,
                        "indexed_at": datetime.now().isoformat(),
                        **metadata,
                    },
                },
            )
            return resp.status_code == 200
        except Exception:
            return False


def extract_with_docling(file_path: Path, output_format: OutputFormat) -> tuple[str, dict]:
    """Extract content from document using Docling"""
    result = converter.convert(str(file_path))
    doc = result.document

    # Get metadata
    metadata = {
        "pages": len(doc.pages) if hasattr(doc, "pages") else 0,
        "tables": len(doc.tables) if hasattr(doc, "tables") else 0,
        "figures": len(doc.pictures) if hasattr(doc, "pictures") else 0,
    }

    # Export in requested format
    if output_format == OutputFormat.MARKDOWN:
        content = doc.export_to_markdown()
    elif output_format == OutputFormat.JSON:
        content = doc.export_to_dict()
    elif output_format == OutputFormat.HTML:
        content = doc.export_to_html()
    else:  # TEXT
        content = doc.export_to_markdown()  # Markdown is readable as plain text

    return content if isinstance(content, str) else str(content), metadata


# ============== API Endpoints ==============

@app.get("/")
async def root():
    """Service info and health check"""
    return {
        "service": "Docling Service",
        "version": "1.0.0",
        "status": "healthy",
        "supported_formats": {
            "documents": list(DOCUMENT_FORMATS),
            "images": list(IMAGE_FORMATS),
            "audio": list(AUDIO_FORMATS),
        },
        "integrations": {
            "ai_orchestrator": AI_ORCHESTRATOR_URL,
            "semantic_search": SEMANTIC_SEARCH_URL,
            "runpod_whisper": f"endpoint:{RUNPOD_WHISPER_ENDPOINT}",
        },
    }


@app.get("/health")
async def health():
    """Health check endpoint for Traefik"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/stats")
async def get_stats():
    """Get processing statistics"""
    return stats


@app.post("/extract", response_model=ExtractionResult)
async def extract_document(request: ExtractRequest):
    """
    Extract content from a document (URL or base64).

    Supports: PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, EPUB, images (with OCR)
    """
    try:
        # Get content
        if request.url:
            content_bytes, filename = await fetch_url_content(str(request.url))
            source = str(request.url)
        elif request.base64_content:
            content_bytes = base64.b64decode(request.base64_content)
            filename = request.filename or "document"
            source = f"base64:{filename}"
        else:
            raise HTTPException(status_code=400, detail="Provide either url or base64_content")

        ext = get_file_extension(filename)

        # Handle audio separately
        if ext in AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail="Use /transcribe endpoint for audio files"
            )

        # Write to temp file for Docling
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content_bytes)
            tmp_path = Path(tmp.name)

        try:
            # Extract content
            content, metadata = extract_with_docling(tmp_path, request.output_format)
            stats["documents_processed"] += 1
            stats["pages_extracted"] += metadata.get("pages", 1)

            # Summarize if requested
            summary = None
            if request.summarize:
                summary = await summarize_with_ollama(content, request.summarize_style)

            # Index if requested
            indexed = False
            if request.index_to_search:
                doc_id = generate_doc_id(source, content)
                indexed = await index_to_semantic_search(
                    doc_id=doc_id,
                    content=content,
                    source=source,
                    metadata={**metadata, **(request.metadata or {})},
                )

            return ExtractionResult(
                success=True,
                source=source,
                content=content,
                format=request.output_format,
                metadata=metadata,
                summary=summary,
                indexed=indexed,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        stats["errors"] += 1
        return ExtractionResult(
            success=False,
            source=str(request.url or request.filename or "unknown"),
            format=request.output_format,
            error=str(e),
        )


@app.post("/extract/upload", response_model=ExtractionResult)
async def extract_uploaded_file(
    file: UploadFile = File(...),
    output_format: OutputFormat = Form(OutputFormat.MARKDOWN),
    summarize: bool = Form(False),
    summarize_style: SummarizeStyle = Form(SummarizeStyle.CONCISE),
    index_to_search: bool = Form(False),
):
    """Extract content from an uploaded file"""
    try:
        content_bytes = await file.read()
        ext = get_file_extension(file.filename or "document")

        if ext in AUDIO_FORMATS:
            raise HTTPException(
                status_code=400,
                detail="Use /transcribe/upload endpoint for audio files"
            )

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content_bytes)
            tmp_path = Path(tmp.name)

        try:
            content, metadata = extract_with_docling(tmp_path, output_format)
            stats["documents_processed"] += 1
            stats["pages_extracted"] += metadata.get("pages", 1)

            summary = None
            if summarize:
                summary = await summarize_with_ollama(content, summarize_style)

            indexed = False
            if index_to_search:
                doc_id = generate_doc_id(file.filename or "upload", content)
                indexed = await index_to_semantic_search(
                    doc_id=doc_id,
                    content=content,
                    source=f"upload:{file.filename}",
                    metadata=metadata,
                )

            return ExtractionResult(
                success=True,
                source=f"upload:{file.filename}",
                content=content,
                format=output_format,
                metadata=metadata,
                summary=summary,
                indexed=indexed,
            )
        finally:
            tmp_path.unlink(missing_ok=True)

    except HTTPException:
        raise
    except Exception as e:
        stats["errors"] += 1
        return ExtractionResult(
            success=False,
            source=f"upload:{file.filename}",
            format=output_format,
            error=str(e),
        )


@app.post("/transcribe")
async def transcribe_audio(request: TranscribeRequest):
    """
    Transcribe audio using RunPod Whisper.

    Supports: MP3, WAV, M4A, OGG, FLAC, WEBM
    """
    try:
        if request.url:
            content_bytes, filename = await fetch_url_content(str(request.url))
            source = str(request.url)
        elif request.base64_content:
            content_bytes = base64.b64decode(request.base64_content)
            source = "base64:audio"
        else:
            raise HTTPException(status_code=400, detail="Provide either url or base64_content")

        # Transcribe
        result = await transcribe_audio_runpod(content_bytes, request.language)
        stats["audio_transcribed"] += 1

        transcript = result.get("transcription", result.get("text", ""))

        # Summarize if requested
        summary = None
        if request.summarize and transcript:
            summary = await summarize_with_ollama(transcript, request.summarize_style)

        return {
            "success": True,
            "source": source,
            "transcript": transcript,
            "language": result.get("detected_language"),
            "duration": result.get("duration"),
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        stats["errors"] += 1
        return {
            "success": False,
            "source": str(request.url or "base64"),
            "error": str(e),
        }


@app.post("/transcribe/upload")
async def transcribe_uploaded_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    summarize: bool = Form(False),
    summarize_style: SummarizeStyle = Form(SummarizeStyle.CONCISE),
):
    """Transcribe uploaded audio file"""
    try:
        content_bytes = await file.read()

        result = await transcribe_audio_runpod(content_bytes, language)
        stats["audio_transcribed"] += 1

        transcript = result.get("transcription", result.get("text", ""))

        summary = None
        if summarize and transcript:
            summary = await summarize_with_ollama(transcript, summarize_style)

        return {
            "success": True,
            "source": f"upload:{file.filename}",
            "transcript": transcript,
            "language": result.get("detected_language"),
            "duration": result.get("duration"),
            "summary": summary,
        }

    except HTTPException:
        raise
    except Exception as e:
        stats["errors"] += 1
        return {
            "success": False,
            "source": f"upload:{file.filename}",
            "error": str(e),
        }


@app.post("/batch")
async def batch_extract(request: BatchExtractRequest, background_tasks: BackgroundTasks):
    """
    Batch extract multiple documents.
    Returns immediately with job ID, processes in background.
    """
    job_id = hashlib.sha256(str(datetime.now()).encode()).hexdigest()[:16]

    # For now, process synchronously (can be enhanced with Redis queue later)
    results = []
    for item in request.items:
        result = await extract_document(item)
        results.append(result)

    return {
        "job_id": job_id,
        "total": len(request.items),
        "results": results,
    }


@app.post("/url/preview")
async def preview_url(url: HttpUrl):
    """Quick preview of URL content (first 500 chars of markdown)"""
    try:
        content_bytes, filename = await fetch_url_content(str(url))
        stats["urls_fetched"] += 1

        ext = get_file_extension(filename)

        with tempfile.NamedTemporaryFile(suffix=ext or ".html", delete=False) as tmp:
            tmp.write(content_bytes)
            tmp_path = Path(tmp.name)

        try:
            content, metadata = extract_with_docling(tmp_path, OutputFormat.MARKDOWN)
            return {
                "success": True,
                "url": str(url),
                "preview": content[:500] + ("..." if len(content) > 500 else ""),
                "full_length": len(content),
                "metadata": metadata,
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    except Exception as e:
        return {
            "success": False,
            "url": str(url),
            "error": str(e),
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
