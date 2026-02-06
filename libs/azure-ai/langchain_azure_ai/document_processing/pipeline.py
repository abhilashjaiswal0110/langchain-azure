"""Multi-modal document processing pipeline implementation.

Provides unified processing for various document types with integrated
text extraction, OCR, table extraction, and image analysis.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional, Union

from langchain_core.documents import Document
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """Supported document types."""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    IMAGE = "image"
    XLSX = "xlsx"
    MIXED = "mixed"


@dataclass
class ProcessingConfig:
    """Configuration for document processing.

    Attributes:
        extract_tables: Extract tables from documents.
        extract_images: Extract and analyze embedded images.
        extract_charts: Extract chart data (requires image analysis).
        ocr_images: Perform OCR on images.
        analyze_layout: Perform layout analysis.
        generate_summaries: Generate AI summaries of content.
        chunk_size: Maximum chunk size for text splitting.
        chunk_overlap: Overlap between chunks.
        table_output_format: Output format for tables (markdown, json, csv).
        preserve_formatting: Preserve document formatting in output.
        max_image_size_mb: Maximum image size to process in MB.
        supported_image_extensions: Supported image file extensions.
    """

    extract_tables: bool = True
    extract_images: bool = True
    extract_charts: bool = True
    ocr_images: bool = True
    analyze_layout: bool = True
    generate_summaries: bool = False
    chunk_size: int = 1000
    chunk_overlap: int = 200
    table_output_format: Literal["markdown", "json", "csv"] = "markdown"
    preserve_formatting: bool = True
    max_image_size_mb: float = 10.0
    supported_image_extensions: tuple[str, ...] = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
    )


@dataclass
class ExtractedTable:
    """Represents an extracted table from a document.

    Attributes:
        data: Table data as list of rows.
        headers: Column headers if detected.
        page_number: Page number where table was found.
        confidence: Extraction confidence score.
        markdown: Markdown representation of the table.
    """

    data: List[List[str]]
    headers: Optional[List[str]] = None
    page_number: Optional[int] = None
    confidence: float = 1.0
    markdown: str = ""

    def __post_init__(self) -> None:
        """Generate markdown representation after initialization."""
        if not self.markdown and self.data:
            self.markdown = self._to_markdown()

    def _to_markdown(self) -> str:
        """Convert table to markdown format."""
        if not self.data:
            return ""

        lines = []
        headers = self.headers or self.data[0]
        data_rows = self.data[1:] if not self.headers else self.data

        # Header row
        lines.append("| " + " | ".join(headers) + " |")
        # Separator row
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        # Data rows
        for row in data_rows:
            # Pad row if necessary
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append("| " + " | ".join(padded_row[: len(headers)]) + " |")

        return "\n".join(lines)


@dataclass
class ExtractedImage:
    """Represents an extracted or analyzed image.

    Attributes:
        source: Image source (file path, URL, or base64).
        caption: AI-generated caption.
        tags: Detected tags/labels.
        objects: Detected objects with bounding boxes.
        text_content: OCR-extracted text from image.
        page_number: Page number where image was found.
        location: Location info (bounding box coordinates).
    """

    source: str
    caption: str = ""
    tags: List[str] = field(default_factory=list)
    objects: List[dict[str, Any]] = field(default_factory=list)
    text_content: str = ""
    page_number: Optional[int] = None
    location: Optional[dict[str, Any]] = None


@dataclass
class LayoutInfo:
    """Document layout analysis information.

    Attributes:
        sections: Document sections.
        paragraphs: Paragraph information.
        reading_order: Inferred reading order.
        page_count: Total number of pages.
        header_footer: Detected headers and footers.
    """

    sections: List[dict[str, Any]] = field(default_factory=list)
    paragraphs: List[dict[str, Any]] = field(default_factory=list)
    reading_order: List[int] = field(default_factory=list)
    page_count: int = 0
    header_footer: Optional[dict[str, Any]] = None


@dataclass
class ProcessedDocument:
    """Result of document processing.

    Attributes:
        text_content: Full extracted text content.
        structured_data: Tables, key-value pairs, forms.
        images: Extracted images with analysis.
        layout: Layout analysis results.
        metadata: Document metadata.
        chunks: Text chunks ready for RAG.
        tables: Extracted tables.
        key_value_pairs: Extracted key-value pairs.
        source_path: Original document path.
        document_type: Detected document type.
    """

    text_content: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    images: List[ExtractedImage] = field(default_factory=list)
    layout: Optional[LayoutInfo] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    chunks: List[Document] = field(default_factory=list)
    tables: List[ExtractedTable] = field(default_factory=list)
    key_value_pairs: List[tuple[str, str]] = field(default_factory=list)
    source_path: str = ""
    document_type: DocumentType = DocumentType.MIXED


class MultiModalDocumentPipeline:
    """Unified pipeline for processing multi-modal documents.

    Handles PDFs, Word documents, PowerPoint, images with integrated
    text extraction, OCR, table extraction, and image analysis.

    Example:
        >>> from langchain_azure_ai.document_processing import (
        ...     MultiModalDocumentPipeline,
        ...     ProcessingConfig,
        ... )
        >>> from langchain_azure_ai.tools import (
        ...     AzureAIDocumentIntelligenceTool,
        ...     AzureAIImageAnalysisTool,
        ... )
        >>>
        >>> # Initialize tools
        >>> doc_intel = AzureAIDocumentIntelligenceTool(
        ...     endpoint="https://...",
        ...     credential="...",
        ... )
        >>> image_analysis = AzureAIImageAnalysisTool(
        ...     endpoint="https://...",
        ...     credential="...",
        ... )
        >>>
        >>> # Create pipeline
        >>> pipeline = MultiModalDocumentPipeline(
        ...     doc_intelligence_tool=doc_intel,
        ...     image_analysis_tool=image_analysis,
        ...     config=ProcessingConfig(
        ...         extract_tables=True,
        ...         extract_images=True,
        ...         chunk_size=1000,
        ...     ),
        ... )
        >>>
        >>> # Process document
        >>> result = await pipeline.process("report.pdf")
        >>> print(f"Extracted {len(result.chunks)} chunks")
        >>> print(f"Found {len(result.tables)} tables")
    """

    # File extension to document type mapping
    EXTENSION_MAP: dict[str, DocumentType] = {
        ".pdf": DocumentType.PDF,
        ".docx": DocumentType.DOCX,
        ".doc": DocumentType.DOCX,
        ".pptx": DocumentType.PPTX,
        ".ppt": DocumentType.PPTX,
        ".xlsx": DocumentType.XLSX,
        ".xls": DocumentType.XLSX,
        ".png": DocumentType.IMAGE,
        ".jpg": DocumentType.IMAGE,
        ".jpeg": DocumentType.IMAGE,
        ".gif": DocumentType.IMAGE,
        ".bmp": DocumentType.IMAGE,
        ".tiff": DocumentType.IMAGE,
        ".webp": DocumentType.IMAGE,
    }

    def __init__(
        self,
        doc_intelligence_tool: Optional[Any] = None,
        image_analysis_tool: Optional[Any] = None,
        config: Optional[ProcessingConfig] = None,
        text_splitter: Optional[Any] = None,
    ):
        """Initialize the multi-modal document pipeline.

        Args:
            doc_intelligence_tool: Azure AI Document Intelligence tool.
            image_analysis_tool: Azure AI Image Analysis tool.
            config: Processing configuration.
            text_splitter: Custom text splitter for chunking.
        """
        self.doc_intel = doc_intelligence_tool
        self.image_analysis = image_analysis_tool
        self.config = config or ProcessingConfig()
        self._text_splitter = text_splitter

        logger.info(
            f"MultiModalDocumentPipeline initialized: "
            f"doc_intel={doc_intelligence_tool is not None}, "
            f"image_analysis={image_analysis_tool is not None}"
        )

    @property
    def text_splitter(self) -> Any:
        """Get or create text splitter for chunking."""
        if self._text_splitter is None:
            try:
                from langchain_text_splitters import RecursiveCharacterTextSplitter

                self._text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
            except ImportError:
                logger.warning(
                    "langchain-text-splitters not installed. "
                    "Falling back to simple splitting."
                )
        return self._text_splitter

    async def process(
        self,
        file_path: str,
        document_type: Optional[DocumentType] = None,
        source_type: str = "local",
    ) -> ProcessedDocument:
        """Process document through multi-modal pipeline.

        Args:
            file_path: Path to document file, URL, or base64 string.
            document_type: Type hint for processing optimization.
            source_type: Source type (local, url, base64).

        Returns:
            ProcessedDocument with all extracted information.

        Raises:
            ValueError: If document type is unsupported.
            RuntimeError: If processing fails.
        """
        # Detect document type
        if document_type is None:
            document_type = self._detect_type(file_path)

        logger.info(f"Processing {document_type.value} document: {file_path[:100]}...")

        try:
            if document_type == DocumentType.IMAGE:
                return await self._process_image(file_path, source_type)
            elif document_type in (
                DocumentType.PDF,
                DocumentType.DOCX,
                DocumentType.PPTX,
                DocumentType.XLSX,
            ):
                return await self._process_document(
                    file_path, document_type, source_type
                )
            else:
                msg = f"Unsupported document type: {document_type}"
                raise ValueError(msg)

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise RuntimeError(f"Document processing failed: {e}") from e

    async def _process_document(
        self,
        file_path: str,
        doc_type: DocumentType,
        source_type: str,
    ) -> ProcessedDocument:
        """Process document-type files (PDF, Word, PPT, Excel).

        Args:
            file_path: Path to document.
            doc_type: Document type.
            source_type: Source type (local, url, base64).

        Returns:
            Processed document.
        """
        if self.doc_intel is None:
            msg = "Document Intelligence tool required for document processing"
            raise ValueError(msg)

        # Step 1: Extract content using Document Intelligence
        doc_result = await self._run_document_intelligence(file_path, source_type)

        # Step 2: Parse structured data (tables, key-value pairs)
        tables = self._extract_tables(doc_result)
        kv_pairs = self._extract_key_value_pairs(doc_result)

        # Step 3: Extract and analyze embedded images
        images: List[ExtractedImage] = []
        if self.config.extract_images and self.image_analysis:
            images = await self._extract_and_analyze_images(doc_result)

        # Step 4: Layout analysis
        layout: Optional[LayoutInfo] = None
        if self.config.analyze_layout:
            layout = self._analyze_layout(doc_result)

        # Step 5: Get text content
        text_content = doc_result.get("content", "")

        # Step 6: Create text chunks for RAG
        chunks = self._create_chunks(
            text_content=text_content,
            tables=tables,
            images=images,
            metadata={
                "source": file_path,
                "type": doc_type.value,
                "has_tables": bool(tables),
                "table_count": len(tables),
                "image_count": len(images),
            },
        )

        return ProcessedDocument(
            text_content=text_content,
            structured_data={
                "tables": [t.data for t in tables],
                "key_value_pairs": kv_pairs,
            },
            images=images,
            layout=layout,
            metadata={
                "source": file_path,
                "type": doc_type.value,
                "page_count": layout.page_count if layout else 0,
            },
            chunks=chunks,
            tables=tables,
            key_value_pairs=kv_pairs,
            source_path=file_path,
            document_type=doc_type,
        )

    async def _process_image(
        self,
        file_path: str,
        source_type: str,
    ) -> ProcessedDocument:
        """Process standalone image file.

        Args:
            file_path: Path to image.
            source_type: Source type (local, url, base64).

        Returns:
            Processed document with image analysis.
        """
        caption = ""
        tags: List[str] = []
        objects: List[dict[str, Any]] = []

        # Analyze image if tool available
        if self.image_analysis:
            analysis_result = await self._run_image_analysis(file_path, source_type)
            caption = analysis_result.get("caption", "")
            tags = analysis_result.get("tags", [])
            objects = analysis_result.get("objects", [])

        # Extract text via OCR if enabled
        text_content = ""
        if self.config.ocr_images and self.doc_intel:
            ocr_result = await self._run_document_intelligence(file_path, source_type)
            text_content = ocr_result.get("content", "")

        # Create image info
        image_info = ExtractedImage(
            source=file_path,
            caption=caption,
            tags=tags,
            objects=objects,
            text_content=text_content,
        )

        # Create content for chunking
        combined_content = self._format_image_content(image_info)

        # Create chunks
        chunks = self._create_chunks(
            text_content=combined_content,
            tables=[],
            images=[image_info],
            metadata={
                "source": file_path,
                "type": "image",
                "has_ocr": bool(text_content),
            },
        )

        return ProcessedDocument(
            text_content=text_content,
            structured_data={},
            images=[image_info],
            layout=None,
            metadata={"source": file_path, "type": "image"},
            chunks=chunks,
            tables=[],
            key_value_pairs=[],
            source_path=file_path,
            document_type=DocumentType.IMAGE,
        )

    async def _run_document_intelligence(
        self,
        source: str,
        source_type: str,
    ) -> dict[str, Any]:
        """Run document intelligence analysis.

        Args:
            source: Document source.
            source_type: Source type.

        Returns:
            Analysis result dictionary.
        """
        if self.doc_intel is None:
            return {}

        try:
            # Use the tool's internal method to get structured result
            if hasattr(self.doc_intel, "_document_analysis"):
                result = self.doc_intel._document_analysis(source, source_type)
                return result
            else:
                # Fallback to string result
                result_str = self.doc_intel._run(source, source_type)
                return {"content": result_str}
        except Exception as e:
            logger.error(f"Document Intelligence error: {e}")
            return {}

    async def _run_image_analysis(
        self,
        source: str,
        source_type: str,
    ) -> dict[str, Any]:
        """Run image analysis.

        Args:
            source: Image source.
            source_type: Source type (local, url, base64).

        Returns:
            Analysis result dictionary with caption, tags, objects.
        """
        if self.image_analysis is None:
            return {}

        try:
            # Map source_type to Azure AI Image Analysis expected values
            # Azure tool expects: "path", "url", or "base64"
            mapped_source_type = source_type
            if source_type == "local":
                mapped_source_type = "path"

            result = self.image_analysis._run(source, mapped_source_type)

            # Parse result - handle both string JSON and dict responses
            if isinstance(result, str):
                # Try to parse as JSON first
                try:
                    import json
                    parsed = json.loads(result)
                    if isinstance(parsed, dict):
                        return {
                            "caption": parsed.get("caption", parsed.get("description", "")),
                            "tags": parsed.get("tags", []),
                            "objects": parsed.get("objects", []),
                        }
                except (json.JSONDecodeError, TypeError):
                    # Treat as plain caption text
                    return {"caption": result, "tags": [], "objects": []}
            elif isinstance(result, dict):
                return {
                    "caption": result.get("caption", result.get("description", "")),
                    "tags": result.get("tags", []),
                    "objects": result.get("objects", []),
                }
            return {"caption": "", "tags": [], "objects": []}
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return {"caption": "", "tags": [], "objects": []}

    def _extract_tables(self, doc_result: dict[str, Any]) -> List[ExtractedTable]:
        """Extract tables from document analysis result.

        Args:
            doc_result: Raw document intelligence result.

        Returns:
            List of extracted tables.
        """
        tables: List[ExtractedTable] = []

        if not self.config.extract_tables:
            return tables

        raw_tables = doc_result.get("tables", [])
        for i, table_data in enumerate(raw_tables):
            if isinstance(table_data, list):
                # Already parsed as list of rows - page number unknown
                table = ExtractedTable(
                    data=table_data,
                    page_number=None,  # Page number not available from raw list
                )
                tables.append(table)
            elif isinstance(table_data, dict):
                # Parse from dict format
                rows = table_data.get("rows", [])
                table = ExtractedTable(
                    data=rows,
                    page_number=table_data.get("page_number"),
                    confidence=table_data.get("confidence", 1.0),
                )
                tables.append(table)

        logger.debug(f"Extracted {len(tables)} tables")
        return tables

    def _extract_key_value_pairs(
        self,
        doc_result: dict[str, Any],
    ) -> List[tuple[str, str]]:
        """Extract key-value pairs from document.

        Args:
            doc_result: Raw document intelligence result.

        Returns:
            List of (key, value) tuples.
        """
        kv_pairs: List[tuple[str, str]] = []

        raw_pairs = doc_result.get("key_value_pairs", [])
        for pair in raw_pairs:
            if isinstance(pair, tuple) and len(pair) == 2:
                kv_pairs.append((str(pair[0]), str(pair[1])))
            elif isinstance(pair, dict):
                key = pair.get("key", "")
                value = pair.get("value", "")
                kv_pairs.append((str(key), str(value)))

        logger.debug(f"Extracted {len(kv_pairs)} key-value pairs")
        return kv_pairs

    async def _extract_and_analyze_images(
        self,
        doc_result: dict[str, Any],
    ) -> List[ExtractedImage]:
        """Extract embedded images and analyze them.

        Args:
            doc_result: Raw document intelligence result.

        Returns:
            List of analyzed images.
        """
        images: List[ExtractedImage] = []

        if not self.config.extract_images or self.image_analysis is None:
            return images

        # Extract embedded images from document
        figures = doc_result.get("figures", [])
        for figure in figures:
            image_data = figure.get("image_data")
            if image_data:
                try:
                    # Analyze each image
                    analysis = await self._run_image_analysis(image_data, "base64")

                    images.append(
                        ExtractedImage(
                            source="embedded",
                            caption=analysis.get("caption", ""),
                            tags=analysis.get("tags", []),
                            objects=analysis.get("objects", []),
                            page_number=figure.get("page_number"),
                            location=figure.get("bounding_box"),
                        )
                    )
                except Exception as e:
                    logger.warning(f"Failed to analyze embedded image: {e}")

        logger.debug(f"Analyzed {len(images)} embedded images")
        return images

    def _analyze_layout(self, doc_result: dict[str, Any]) -> LayoutInfo:
        """Analyze document layout structure.

        Args:
            doc_result: Raw document intelligence result.

        Returns:
            Layout analysis information.
        """
        return LayoutInfo(
            sections=doc_result.get("sections", []),
            paragraphs=doc_result.get("paragraphs", []),
            reading_order=doc_result.get("reading_order", []),
            page_count=doc_result.get("page_count", len(doc_result.get("pages", []))),
            header_footer=doc_result.get("header_footer"),
        )

    def _format_image_content(self, image: ExtractedImage) -> str:
        """Format image content for chunking.

        Args:
            image: Extracted image info.

        Returns:
            Formatted content string.
        """
        parts = []

        if image.caption:
            parts.append(f"Image Caption: {image.caption}")

        if image.tags:
            parts.append(f"Tags: {', '.join(image.tags)}")

        if image.text_content:
            parts.append(f"OCR Text: {image.text_content}")

        if image.objects:
            object_names = [obj.get("name", "") for obj in image.objects if obj.get("name")]
            if object_names:
                parts.append(f"Detected Objects: {', '.join(object_names)}")

        return "\n".join(parts)

    def _create_chunks(
        self,
        text_content: str,
        tables: List[ExtractedTable],
        images: List[ExtractedImage],
        metadata: dict[str, Any],
    ) -> List[Document]:
        """Split text into chunks for RAG.

        Args:
            text_content: Full text content.
            tables: Extracted tables.
            images: Extracted images.
            metadata: Document metadata.

        Returns:
            List of Document chunks.
        """
        chunks: List[Document] = []

        # Create chunks from text content
        if text_content and self.text_splitter:
            try:
                text_chunks = self.text_splitter.create_documents(
                    [text_content],
                    metadatas=[{**metadata, "content_type": "text"}],
                )
                chunks.extend(text_chunks)
            except Exception as e:
                logger.warning(f"Text splitting failed: {e}")
                # Fallback: create single chunk
                chunks.append(
                    Document(
                        page_content=text_content,
                        metadata={**metadata, "content_type": "text"},
                    )
                )

        # Add table chunks
        for i, table in enumerate(tables):
            if table.markdown:
                chunks.append(
                    Document(
                        page_content=table.markdown,
                        metadata={
                            **metadata,
                            "content_type": "table",
                            "table_index": i,
                            "page_number": table.page_number,
                        },
                    )
                )

        # Add image description chunks
        for i, image in enumerate(images):
            image_content = self._format_image_content(image)
            if image_content:
                chunks.append(
                    Document(
                        page_content=image_content,
                        metadata={
                            **metadata,
                            "content_type": "image",
                            "image_index": i,
                            "page_number": image.page_number,
                        },
                    )
                )

        logger.debug(f"Created {len(chunks)} chunks")
        return chunks

    def _detect_type(self, file_path: str) -> DocumentType:
        """Auto-detect document type from file extension or content.

        Args:
            file_path: Path to file or data URI.

        Returns:
            Detected document type.
        """
        # Handle data URIs
        if file_path.startswith("data:"):
            mime_type = file_path.split(";")[0].split(":")[1]
            if "image" in mime_type:
                return DocumentType.IMAGE
            elif "pdf" in mime_type:
                return DocumentType.PDF
            return DocumentType.MIXED

        # Handle URLs
        if file_path.startswith(("http://", "https://")):
            # Try to extract extension from URL
            from urllib.parse import urlparse

            path = urlparse(file_path).path
            extension = Path(path).suffix.lower()
            return self.EXTENSION_MAP.get(extension, DocumentType.MIXED)

        # Handle file paths
        extension = Path(file_path).suffix.lower()
        return self.EXTENSION_MAP.get(extension, DocumentType.MIXED)

    async def process_batch(
        self,
        file_paths: List[str],
        max_concurrent: int = 5,
    ) -> List[ProcessedDocument]:
        """Process multiple documents concurrently.

        Args:
            file_paths: List of file paths to process.
            max_concurrent: Maximum concurrent processing tasks.

        Returns:
            List of processed documents.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(path: str) -> ProcessedDocument:
            async with semaphore:
                return await self.process(path)

        tasks = [process_with_semaphore(path) for path in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed: List[ProcessedDocument] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {file_paths[i]}: {result}")
            else:
                processed.append(result)

        return processed
