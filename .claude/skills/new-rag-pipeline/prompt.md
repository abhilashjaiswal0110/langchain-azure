# New RAG Pipeline Skill - Execution Instructions

You are executing the `new-rag-pipeline` skill for the LangChain Azure repository.

## Purpose

Create a complete RAG (Retrieval Augmented Generation) pipeline with document loading, embedding, vector storage, and querying.

## Parameters

- **pipeline_name** (required): Pipeline name in kebab-case
- **storage** (optional, default: "search"): Vector storage (search/cosmos/postgresql)
- **embeddings** (optional, default: "azure"): Embeddings provider
- **with_web_ui** (optional, default: true): Include web UI

## Execution Steps

### Step 1: Create Pipeline Directory

```bash
mkdir -p samples/rag-{{pipeline_name}}
cd samples/rag-{{pipeline_name}}
```

### Step 2: Create Pipeline Module

Create `pipeline.py`:

```python
"""{{Title}} RAG Pipeline.

This module implements a complete RAG pipeline using Azure services.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Import based on storage choice
{% if storage == "search" %}
from langchain_azure_ai.vectorstores import AzureAISearchVectorStore
{% elif storage == "cosmos" %}
from langchain_azure_ai.vectorstores import AzureCosmosDBVectorStore
{% elif storage == "postgresql" %}
from langchain_azure_postgresql.vectorstores import AzurePostgreSQLVectorStore
{% endif %}

# Import based on embeddings choice
{% if embeddings == "azure" %}
from langchain_azure_ai.embeddings import AzureOpenAIEmbeddings
{% else %}
from langchain_openai import OpenAIEmbeddings
{% endif %}


class {{PascalCase}}Pipeline:
    """{{Title}} RAG Pipeline.

    A complete pipeline for document ingestion, embedding, and retrieval.
    """

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        vectorstore: Optional[VectorStore] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        """Initialize the RAG pipeline.

        Args:
            embeddings: Embeddings model to use.
            vectorstore: Vector store for document storage.
            chunk_size: Size of document chunks.
            chunk_overlap: Overlap between chunks.
        """
        self.embeddings = embeddings or self._get_default_embeddings()
        self.vectorstore = vectorstore or self._get_default_vectorstore()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _get_default_embeddings(self) -> Embeddings:
        """Get default embeddings model."""
        {% if embeddings == "azure" %}
        return AzureOpenAIEmbeddings(
            model="text-embedding-3-large",
        )
        {% else %}
        return OpenAIEmbeddings(
            model="text-embedding-3-large",
        )
        {% endif %}

    def _get_default_vectorstore(self) -> VectorStore:
        """Get default vector store."""
        {% if storage == "search" %}
        return AzureAISearchVectorStore(
            index_name="{{pipeline_name}}",
            embedding_function=self.embeddings,
        )
        {% elif storage == "cosmos" %}
        return AzureCosmosDBVectorStore(
            collection_name="{{pipeline_name}}",
            embedding=self.embeddings,
        )
        {% elif storage == "postgresql" %}
        return AzurePostgreSQLVectorStore(
            collection_name="{{pipeline_name}}",
            embedding_function=self.embeddings,
        )
        {% endif %}

    def ingest_documents(
        self,
        documents: list[Document],
    ) -> list[str]:
        """Ingest documents into the pipeline.

        Args:
            documents: List of documents to ingest.

        Returns:
            List of document IDs.
        """
        # Split documents
        chunks = self.text_splitter.split_documents(documents)

        # Add to vectorstore
        ids = self.vectorstore.add_documents(chunks)

        return ids

    def query(
        self,
        query: str,
        k: int = 4,
    ) -> list[Document]:
        """Query the RAG pipeline.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of relevant documents.
        """
        results = self.vectorstore.similarity_search(query, k=k)
        return results

    def query_with_scores(
        self,
        query: str,
        k: int = 4,
    ) -> list[tuple[Document, float]]:
        """Query with relevance scores.

        Args:
            query: Search query.
            k: Number of results to return.

        Returns:
            List of (document, score) tuples.
        """
        results = self.vectorstore.similarity_search_with_score(query, k=k)
        return results
```

### Step 3: Create Document Embedding Script

Create `embed.py`:

```python
"""Embed documents into {{Title}} pipeline."""

import os
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)

from pipeline import {{PascalCase}}Pipeline


def load_documents(directory: str) -> list:
    """Load documents from directory.

    Args:
        directory: Path to documents directory.

    Returns:
        List of loaded documents.
    """
    pdf_loader = DirectoryLoader(
        directory,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )

    txt_loader = DirectoryLoader(
        directory,
        glob="**/*.txt",
        loader_cls=TextLoader,
    )

    pdf_docs = pdf_loader.load()
    txt_docs = txt_loader.load()

    return pdf_docs + txt_docs


def main() -> None:
    """Main embedding function."""
    # Load documents
    docs_dir = os.getenv("DOCS_DIR", "./documents")
    print(f"📂 Loading documents from {docs_dir}...")

    documents = load_documents(docs_dir)
    print(f"📄 Loaded {len(documents)} documents")

    # Create pipeline
    print("🔧 Initializing pipeline...")
    pipeline = {{PascalCase}}Pipeline()

    # Ingest documents
    print("💾 Ingesting documents...")
    ids = pipeline.ingest_documents(documents)

    print(f"✅ Successfully ingested {len(ids)} document chunks")
    print(f"🔑 Document IDs: {ids[:5]}..." if len(ids) > 5 else f"🔑 Document IDs: {ids}")


if __name__ == "__main__":
    main()
```

### Step 4: Create Query Script

Create `query.py`:

```python
"""Query {{Title}} RAG pipeline."""

import sys

from pipeline import {{PascalCase}}Pipeline


def main() -> None:
    """Main query function."""
    # Get query from args or prompt
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your query: ")

    print(f"\n🔍 Searching for: {query}\n")

    # Create pipeline
    pipeline = {{PascalCase}}Pipeline()

    # Query
    results = pipeline.query_with_scores(query, k=4)

    # Display results
    print(f"📊 Found {len(results)} results:\n")

    for i, (doc, score) in enumerate(results, 1):
        print(f"Result {i} (Score: {score:.4f})")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"Content: {doc.page_content[:200]}...")
        print("-" * 80)


if __name__ == "__main__":
    main()
```

### Step 5: Create README

Create `README.md` with setup and usage instructions.

### Step 6: Create Requirements

Create `requirements.txt`:
```
langchain-azure-ai>=0.1.0
langchain-openai>=0.1.0
{% if storage == "cosmos" %}
langchain-azure-cosmos>=0.1.0
{% elif storage == "postgresql" %}
langchain-azure-postgresql>=0.1.0
{% endif %}
langchain-community>=0.1.0
pypdf>=3.0.0
python-dotenv>=1.0.0
```

### Step 7: Create .env.example

```bash
# Azure {{Storage}} Configuration
{% if storage == "search" %}
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your-key
{% elif storage == "cosmos" %}
COSMOS_CONNECTION_STRING=your-connection-string
COSMOS_DATABASE_NAME=your-db
{% elif storage == "postgresql" %}
POSTGRESQL_CONNECTION_STRING=your-connection-string
{% endif %}

# {{Embeddings}} Configuration
{% if embeddings == "azure" %}
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
{% else %}
OPENAI_API_KEY=sk-...
{% endif %}

# Documents directory
DOCS_DIR=./documents
```

### Step 8: Create Web UI (if with_web_ui=true)

Create `app.py` with Gradio or Streamlit UI.

## Output Summary

```
✅ Successfully created {{Title}} RAG Pipeline!

📁 Files Created:
- samples/rag-{{pipeline_name}}/pipeline.py
- samples/rag-{{pipeline_name}}/embed.py
- samples/rag-{{pipeline_name}}/query.py
- samples/rag-{{pipeline_name}}/README.md
- samples/rag-{{pipeline_name}}/requirements.txt
- samples/rag-{{pipeline_name}}/.env.example
{% if with_web_ui %}
- samples/rag-{{pipeline_name}}/app.py
{% endif %}

📊 Configuration:
- Storage: {{storage}}
- Embeddings: {{embeddings}}
- Web UI: {{with_web_ui}}

📋 Next Steps:
1. cd samples/rag-{{pipeline_name}}
2. cp .env.example .env
3. Edit .env with your credentials
4. pip install -r requirements.txt
5. Place documents in ./documents/
6. Run: python embed.py
7. Query: python query.py "your question"

📚 Documentation: samples/rag-{{pipeline_name}}/README.md
```

## Success Criteria

- All files created successfully
- Templates properly filled with parameters
- README has clear setup instructions
- .env.example has all required variables
- Code follows repository standards
