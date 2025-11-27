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
from fastapi.responses import JSONResponse, HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Web interface for document processing"""
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Intelligence - Extract, Summarize, Index</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e0e0e0;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            text-align: center;
            padding: 40px 0 30px;
        }
        h1 {
            font-size: 2.5rem;
            background: linear-gradient(90deg, #00d9ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #888;
            font-size: 1.1rem;
        }
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        @media (max-width: 900px) {
            .main-grid { grid-template-columns: 1fr; }
        }
        .card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }
        .card h2 {
            color: #00d9ff;
            margin-bottom: 20px;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #aaa;
            font-size: 0.9rem;
        }
        input[type="text"], input[type="url"], select, textarea {
            width: 100%;
            padding: 12px 16px;
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            background: rgba(0,0,0,0.3);
            color: #fff;
            font-size: 1rem;
            transition: border-color 0.2s;
        }
        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #00d9ff;
        }
        .drop-zone {
            border: 2px dashed rgba(0,217,255,0.3);
            border-radius: 12px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: rgba(0,217,255,0.02);
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #00d9ff;
            background: rgba(0,217,255,0.08);
        }
        .drop-zone-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }
        .drop-zone p {
            color: #888;
        }
        .drop-zone .formats {
            font-size: 0.8rem;
            color: #666;
            margin-top: 10px;
        }
        .file-info {
            display: none;
            padding: 15px;
            background: rgba(0,255,136,0.1);
            border-radius: 8px;
            margin-top: 15px;
        }
        .file-info.visible { display: block; }
        .options-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            cursor: pointer;
        }
        .checkbox-group:hover {
            background: rgba(0,0,0,0.3);
        }
        input[type="checkbox"] {
            width: 18px;
            height: 18px;
            accent-color: #00d9ff;
        }
        .btn {
            width: 100%;
            padding: 14px 24px;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(90deg, #00d9ff, #00b8d4);
            color: #000;
        }
        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,217,255,0.3);
        }
        .btn-primary:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: #fff;
            margin-top: 10px;
        }
        .btn-secondary:hover {
            background: rgba(255,255,255,0.15);
        }
        .results-card {
            grid-column: 1 / -1;
        }
        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .status {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 0.85rem;
        }
        .status.processing {
            background: rgba(255,193,7,0.2);
            color: #ffc107;
        }
        .status.success {
            background: rgba(0,255,136,0.2);
            color: #00ff88;
        }
        .status.error {
            background: rgba(255,82,82,0.2);
            color: #ff5252;
        }
        .spinner {
            width: 16px;
            height: 16px;
            border: 2px solid transparent;
            border-top-color: currentColor;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 5px;
        }
        .tab {
            padding: 10px 20px;
            background: transparent;
            border: none;
            color: #888;
            cursor: pointer;
            border-radius: 8px 8px 0 0;
            transition: all 0.2s;
        }
        .tab:hover { color: #fff; }
        .tab.active {
            background: rgba(0,217,255,0.1);
            color: #00d9ff;
        }
        .tab-content {
            display: none;
            max-height: 500px;
            overflow-y: auto;
        }
        .tab-content.active { display: block; }
        .content-display {
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 20px;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
            line-height: 1.7;
        }
        .summary-display {
            background: linear-gradient(135deg, rgba(0,217,255,0.1), rgba(0,255,136,0.1));
            border-left: 3px solid #00d9ff;
            padding: 20px;
            border-radius: 0 8px 8px 0;
        }
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }
        .meta-item {
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        .meta-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00d9ff;
        }
        .meta-label {
            font-size: 0.8rem;
            color: #888;
            margin-top: 5px;
        }
        .indexed-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 16px;
            background: rgba(0,255,136,0.2);
            color: #00ff88;
            border-radius: 20px;
            font-size: 0.9rem;
        }
        .stats-bar {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 30px;
            padding: 20px;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
        }
        .stat {
            text-align: center;
        }
        .stat-value {
            font-size: 1.5rem;
            font-weight: 700;
            color: #00d9ff;
        }
        .stat-label {
            font-size: 0.75rem;
            color: #666;
        }
        #results { display: none; }
        #results.visible { display: block; }
        .hidden { display: none !important; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Document Intelligence</h1>
            <p class="subtitle">Extract, summarize, and index documents into your knowledge base</p>
        </header>

        <div class="main-grid">
            <!-- Input Card -->
            <div class="card">
                <h2><span>📄</span> Input Source</h2>

                <div class="input-group">
                    <label for="url-input">URL (webpage, PDF, document)</label>
                    <input type="url" id="url-input" placeholder="https://example.com/document.pdf">
                </div>

                <div style="text-align: center; color: #666; margin: 15px 0;">— or —</div>

                <div class="drop-zone" id="drop-zone">
                    <div class="drop-zone-icon">📁</div>
                    <p>Drop a file here or click to browse</p>
                    <p class="formats">PDF, DOCX, PPTX, XLSX, HTML, MD, TXT, EPUB, images, audio</p>
                    <input type="file" id="file-input" hidden accept=".pdf,.docx,.pptx,.xlsx,.html,.md,.txt,.epub,.png,.jpg,.jpeg,.tiff,.bmp,.gif,.mp3,.wav,.m4a,.ogg,.flac,.webm">
                </div>
                <div class="file-info" id="file-info">
                    <strong id="file-name"></strong>
                    <span id="file-size" style="color: #888; margin-left: 10px;"></span>
                </div>
            </div>

            <!-- Options Card -->
            <div class="card">
                <h2><span>⚙️</span> Processing Options</h2>

                <div class="input-group">
                    <label for="summary-style">Summary Style</label>
                    <select id="summary-style">
                        <option value="concise">Concise (2-3 sentences)</option>
                        <option value="bullet_points">Bullet Points</option>
                        <option value="detailed">Detailed</option>
                        <option value="technical">Technical</option>
                        <option value="eli5">ELI5 (Simple explanation)</option>
                    </select>
                </div>

                <div class="input-group">
                    <label for="output-format">Output Format</label>
                    <select id="output-format">
                        <option value="markdown">Markdown</option>
                        <option value="text">Plain Text</option>
                        <option value="html">HTML</option>
                        <option value="json">JSON</option>
                    </select>
                </div>

                <div class="input-group">
                    <label>Actions</label>
                    <div class="options-grid">
                        <label class="checkbox-group">
                            <input type="checkbox" id="do-summarize" checked>
                            <span>Generate Summary</span>
                        </label>
                        <label class="checkbox-group">
                            <input type="checkbox" id="do-index">
                            <span>Index to Knowledge Base</span>
                        </label>
                    </div>
                </div>

                <button class="btn btn-primary" id="process-btn" onclick="processDocument()">
                    Process Document
                </button>
                <button class="btn btn-secondary" onclick="clearAll()">Clear</button>
            </div>

            <!-- Results Card -->
            <div class="card results-card" id="results">
                <div class="results-header">
                    <h2><span>📊</span> Results</h2>
                    <div>
                        <span class="status" id="status">
                            <span class="spinner"></span>
                            Processing...
                        </span>
                        <span class="indexed-badge hidden" id="indexed-badge">
                            ✓ Indexed to Knowledge Base
                        </span>
                    </div>
                </div>

                <div class="meta-grid" id="meta-grid" style="margin-bottom: 20px;">
                    <div class="meta-item">
                        <div class="meta-value" id="meta-pages">-</div>
                        <div class="meta-label">Pages</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-value" id="meta-tables">-</div>
                        <div class="meta-label">Tables</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-value" id="meta-figures">-</div>
                        <div class="meta-label">Figures</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-value" id="meta-chars">-</div>
                        <div class="meta-label">Characters</div>
                    </div>
                </div>

                <div class="tabs">
                    <button class="tab active" onclick="switchTab('summary')">Summary</button>
                    <button class="tab" onclick="switchTab('content')">Full Content</button>
                    <button class="tab" id="transcript-tab" style="display:none;" onclick="switchTab('transcript')">Transcript</button>
                </div>

                <div class="tab-content active" id="tab-summary">
                    <div class="summary-display" id="summary-content">
                        Processing...
                    </div>
                </div>
                <div class="tab-content" id="tab-content">
                    <div class="content-display" id="full-content">
                        Loading content...
                    </div>
                </div>
                <div class="tab-content" id="tab-transcript">
                    <div class="content-display" id="transcript-content">
                    </div>
                </div>
            </div>
        </div>

        <div class="stats-bar">
            <div class="stat">
                <div class="stat-value" id="stat-docs">""" + str(stats["documents_processed"]) + """</div>
                <div class="stat-label">Documents Processed</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-pages">""" + str(stats["pages_extracted"]) + """</div>
                <div class="stat-label">Pages Extracted</div>
            </div>
            <div class="stat">
                <div class="stat-value" id="stat-audio">""" + str(stats["audio_transcribed"]) + """</div>
                <div class="stat-label">Audio Transcribed</div>
            </div>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const fileInfo = document.getElementById('file-info');
        let selectedFile = null;

        // Drag and drop
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                handleFile(e.dataTransfer.files[0]);
            }
        });
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFile(e.target.files[0]);
            }
        });

        function handleFile(file) {
            selectedFile = file;
            document.getElementById('file-name').textContent = file.name;
            document.getElementById('file-size').textContent = formatBytes(file.size);
            fileInfo.classList.add('visible');
            document.getElementById('url-input').value = '';
        }

        function formatBytes(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        function switchTab(tab) {
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById('tab-' + tab).classList.add('active');
        }

        function clearAll() {
            document.getElementById('url-input').value = '';
            selectedFile = null;
            fileInfo.classList.remove('visible');
            document.getElementById('results').classList.remove('visible');
        }

        async function processDocument() {
            const url = document.getElementById('url-input').value.trim();
            const summarize = document.getElementById('do-summarize').checked;
            const indexToSearch = document.getElementById('do-index').checked;
            const summaryStyle = document.getElementById('summary-style').value;
            const outputFormat = document.getElementById('output-format').value;

            if (!url && !selectedFile) {
                alert('Please provide a URL or upload a file');
                return;
            }

            // Show results and set processing state
            const results = document.getElementById('results');
            const status = document.getElementById('status');
            const processBtn = document.getElementById('process-btn');

            results.classList.add('visible');
            status.className = 'status processing';
            status.innerHTML = '<span class="spinner"></span> Processing...';
            processBtn.disabled = true;
            document.getElementById('indexed-badge').classList.add('hidden');

            try {
                let response;
                const isAudio = selectedFile && /\\.(mp3|wav|m4a|ogg|flac|webm)$/i.test(selectedFile.name);

                if (selectedFile) {
                    const formData = new FormData();
                    formData.append('file', selectedFile);
                    formData.append('summarize', summarize);
                    formData.append('summarize_style', summaryStyle);
                    formData.append('index_to_search', indexToSearch);
                    if (!isAudio) {
                        formData.append('output_format', outputFormat);
                    }

                    const endpoint = isAudio ? '/transcribe/upload' : '/extract/upload';
                    response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    // Check if URL points to audio
                    const urlIsAudio = /\\.(mp3|wav|m4a|ogg|flac|webm)(\\?|$)/i.test(url);
                    const endpoint = urlIsAudio ? '/transcribe' : '/extract';

                    const payload = urlIsAudio ? {
                        url: url,
                        summarize: summarize,
                        summarize_style: summaryStyle
                    } : {
                        url: url,
                        output_format: outputFormat,
                        summarize: summarize,
                        summarize_style: summaryStyle,
                        index_to_search: indexToSearch
                    };

                    response = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(payload)
                    });
                }

                const data = await response.json();

                if (data.success) {
                    status.className = 'status success';
                    status.textContent = '✓ Complete';

                    // Handle audio transcription
                    if (data.transcript !== undefined) {
                        document.getElementById('transcript-tab').style.display = 'block';
                        document.getElementById('transcript-content').textContent = data.transcript || 'No transcript available';
                        document.getElementById('full-content').textContent = data.transcript || '';
                        document.getElementById('meta-pages').textContent = '-';
                        document.getElementById('meta-tables').textContent = '-';
                        document.getElementById('meta-figures').textContent = '-';
                        document.getElementById('meta-chars').textContent = data.transcript ? data.transcript.length.toLocaleString() : '0';

                        // Switch to transcript tab
                        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                        document.getElementById('transcript-tab').classList.add('active');
                        document.getElementById('tab-transcript').classList.add('active');
                    } else {
                        document.getElementById('transcript-tab').style.display = 'none';
                        document.getElementById('full-content').textContent = data.content || 'No content extracted';

                        const meta = data.metadata || {};
                        document.getElementById('meta-pages').textContent = meta.pages || '-';
                        document.getElementById('meta-tables').textContent = meta.tables || '-';
                        document.getElementById('meta-figures').textContent = meta.figures || '-';
                        document.getElementById('meta-chars').textContent = data.content ? data.content.length.toLocaleString() : '0';
                    }

                    document.getElementById('summary-content').textContent = data.summary || 'No summary generated';

                    if (data.indexed) {
                        document.getElementById('indexed-badge').classList.remove('hidden');
                    }

                    // Update stats
                    fetchStats();
                } else {
                    status.className = 'status error';
                    status.textContent = '✗ Error';
                    document.getElementById('summary-content').textContent = 'Error: ' + (data.error || 'Unknown error');
                    document.getElementById('full-content').textContent = '';
                }
            } catch (err) {
                status.className = 'status error';
                status.textContent = '✗ Error';
                document.getElementById('summary-content').textContent = 'Error: ' + err.message;
            } finally {
                processBtn.disabled = false;
            }
        }

        async function fetchStats() {
            try {
                const resp = await fetch('/stats');
                const stats = await resp.json();
                document.getElementById('stat-docs').textContent = stats.documents_processed || 0;
                document.getElementById('stat-pages').textContent = stats.pages_extracted || 0;
                document.getElementById('stat-audio').textContent = stats.audio_transcribed || 0;
            } catch (e) {}
        }

        // Initial stats load
        fetchStats();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html)


@app.get("/api/info")
async def api_info():
    """Service info (JSON API endpoint)"""
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
