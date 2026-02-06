"""Multi-modal document processing pipeline for Azure AI.

This module provides unified document processing capabilities:
- PDF, Word, PowerPoint, image processing
- Table extraction and OCR
- Image analysis and captioning
- Layout-aware text chunking for RAG

Usage:
    from langchain_azure_ai.document_processing import (
        MultiModalDocumentPipeline,
        ProcessingConfig,
        ProcessedDocument,
    )

    # Initialize pipeline
    pipeline = MultiModalDocumentPipeline(
        doc_intelligence_tool=doc_intel_tool,
        image_analysis_tool=image_tool,
        config=ProcessingConfig(extract_tables=True),
    )

    # Process document
    result = await pipeline.process("document.pdf")
    print(f"Extracted {len(result.chunks)} chunks")
"""

from langchain_azure_ai.document_processing.pipeline import (
    DocumentType,
    MultiModalDocumentPipeline,
    ProcessedDocument,
    ProcessingConfig,
)

__all__ = [
    "MultiModalDocumentPipeline",
    "ProcessingConfig",
    "ProcessedDocument",
    "DocumentType",
]
