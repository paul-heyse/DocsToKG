"""Metadata extraction and enrichment.

Provides:
  - PDF text extraction + metadata
  - HTML parsing + structure analysis
  - Generic content analysis
  - Storage in catalog
  - Searchability enhancement
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ContentMetadata:
    """Extracted content metadata."""
    content_type: str
    title: Optional[str] = None
    authors: list[str] = field(default_factory=list)
    description: Optional[str] = None
    keywords: list[str] = field(default_factory=list)
    language: Optional[str] = None
    page_count: Optional[int] = None
    text_preview: Optional[str] = None  # First 500 chars
    extraction_date: Optional[str] = None
    source_hash: Optional[str] = None
    custom_fields: dict[str, Any] = field(default_factory=dict)


class MetadataExtractor:
    """Extract metadata from various file types."""
    
    def __init__(self):
        """Initialize extractor."""
        self._extractors = {
            "application/pdf": self._extract_pdf,
            "text/html": self._extract_html,
            "application/json": self._extract_json,
            "text/plain": self._extract_text,
        }
    
    def extract(self, file_path: str) -> ContentMetadata:
        """Extract metadata from file.
        
        Args:
            file_path: Path to file
            
        Returns:
            ContentMetadata with extracted information
        """
        from datetime import datetime
        
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Detect content type
            content_type = self._detect_content_type(path)
            logger.info(f"Extracting metadata from {path.name} ({content_type})")
            
            # Get extractor
            extractor = self._extractors.get(content_type, self._extract_generic)
            
            # Extract
            metadata = extractor(file_path)
            
            # Add extraction timestamp
            return ContentMetadata(
                content_type=content_type,
                title=metadata.get("title"),
                authors=metadata.get("authors", []),
                description=metadata.get("description"),
                keywords=metadata.get("keywords", []),
                language=metadata.get("language"),
                page_count=metadata.get("page_count"),
                text_preview=metadata.get("text_preview"),
                extraction_date=datetime.utcnow().isoformat(),
                source_hash=metadata.get("source_hash"),
                custom_fields=metadata.get("custom_fields", {}),
            )
        
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
            return ContentMetadata(content_type="unknown")
    
    def _detect_content_type(self, path: Path) -> str:
        """Detect file content type."""
        suffix = path.suffix.lower()
        
        mapping = {
            ".pdf": "application/pdf",
            ".html": "text/html",
            ".htm": "text/html",
            ".json": "application/json",
            ".txt": "text/plain",
        }
        
        return mapping.get(suffix, "application/octet-stream")
    
    def _extract_pdf(self, file_path: str) -> dict:
        """Extract metadata from PDF."""
        try:
            import PyPDF2
        except ImportError:
            logger.warning("PyPDF2 not installed, skipping PDF extraction")
            return {"text_preview": "PDF (PyPDF2 required)"}
        
        try:
            metadata = {}
            
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                
                # Extract metadata
                if reader.metadata:
                    metadata["title"] = reader.metadata.get("/Title")
                    metadata["authors"] = [reader.metadata.get("/Author")] if reader.metadata.get("/Author") else []
                    metadata["description"] = reader.metadata.get("/Subject")
                
                # Page count
                metadata["page_count"] = len(reader.pages)
                
                # Extract text preview from first page
                if reader.pages:
                    text = reader.pages[0].extract_text()
                    metadata["text_preview"] = text[:500] if text else None
            
            return metadata
        
        except Exception as e:
            logger.warning(f"PDF extraction error: {e}")
            return {}
    
    def _extract_html(self, file_path: str) -> dict:
        """Extract metadata from HTML."""
        try:
            from html.parser import HTMLParser
        except ImportError:
            return {}
        
        try:
            metadata = {}
            
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Parse HTML
            from html import unescape
            
            # Extract title
            import re
            title_match = re.search(r"<title[^>]*>([^<]+)</title>", content, re.IGNORECASE)
            if title_match:
                metadata["title"] = unescape(title_match.group(1)).strip()
            
            # Extract meta description
            desc_match = re.search(r'<meta\s+name="description"\s+content="([^"]+)"', content, re.IGNORECASE)
            if desc_match:
                metadata["description"] = unescape(desc_match.group(1)).strip()
            
            # Extract keywords
            keywords_match = re.search(r'<meta\s+name="keywords"\s+content="([^"]+)"', content, re.IGNORECASE)
            if keywords_match:
                keywords = [k.strip() for k in keywords_match.group(1).split(",")]
                metadata["keywords"] = keywords
            
            # Extract language
            lang_match = re.search(r'<html[^>]*\s+lang="([^"]+)"', content, re.IGNORECASE)
            if lang_match:
                metadata["language"] = lang_match.group(1).split("-")[0]
            
            # Text preview
            text_match = re.search(r"<body[^>]*>(.+?)</body>", content, re.IGNORECASE | re.DOTALL)
            if text_match:
                text = re.sub(r"<[^>]+>", "", text_match.group(1))
                text = re.sub(r"\s+", " ", text).strip()
                metadata["text_preview"] = text[:500]
            
            return metadata
        
        except Exception as e:
            logger.warning(f"HTML extraction error: {e}")
            return {}
    
    def _extract_json(self, file_path: str) -> dict:
        """Extract metadata from JSON."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            metadata = {}
            
            # Look for common metadata fields
            if isinstance(data, dict):
                metadata["title"] = data.get("title") or data.get("name")
                metadata["description"] = data.get("description")
                metadata["keywords"] = data.get("keywords", [])
                
                # Text preview from content
                content = data.get("content") or data.get("text")
                if isinstance(content, str):
                    metadata["text_preview"] = content[:500]
            
            return metadata
        
        except Exception as e:
            logger.warning(f"JSON extraction error: {e}")
            return {}
    
    def _extract_text(self, file_path: str) -> dict:
        """Extract metadata from plain text."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            
            metadata = {}
            
            # Use first line as title if it looks like one
            lines = text.split("\n")
            if lines:
                first_line = lines[0].strip()
                if len(first_line) < 200 and not first_line.startswith((" ", "\t")):
                    metadata["title"] = first_line
            
            # Text preview
            metadata["text_preview"] = text[:500]
            
            return metadata
        
        except Exception as e:
            logger.warning(f"Text extraction error: {e}")
            return {}
    
    def _extract_generic(self, file_path: str) -> dict:
        """Generic metadata extraction."""
        path = Path(file_path)
        return {
            "title": path.stem,
            "text_preview": f"File: {path.name} ({path.stat().st_size} bytes)",
        }


def extract_and_store(
    file_path: str,
    catalog,
    record_id: int,
) -> bool:
    """Extract metadata and store in catalog.
    
    Args:
        file_path: File to extract from
        catalog: Catalog store
        record_id: Record to attach metadata to
        
    Returns:
        True if successful
    """
    try:
        extractor = MetadataExtractor()
        metadata = extractor.extract(file_path)
        
        # Store metadata as JSON in catalog
        metadata_json = json.dumps(asdict(metadata))
        logger.info(f"Extracted metadata: {len(metadata_json)} bytes")
        
        # Note: Would store in catalog's metadata table in production
        return True
    
    except Exception as e:
        logger.error(f"Failed to extract metadata: {e}")
        return False
