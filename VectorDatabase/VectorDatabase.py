from __future__ import annotations
from TextProcessor import TextProcessor       # provides text processing (tokenisation) utilities
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv
import os

import weaviate

# ------------------------------------------------------------------
# Abstract vector store interface
# ------------------------------------------------------------------
class VectorDatabase(ABC):
    """
    Parent class for any vector store (Weaviate, Chroma, Qdrant…).
    Sub-classes must implement:
        - create_schema()
        - ingest_documents()
        - search()   (optional for now)
    """

    @abstractmethod
    def create_schema(self, class_name: str) -> str:
        ...

    @abstractmethod
    def ingest_documents(self, root: Path, class_name: str = "KnowledgeChunk") -> None:
        ...

    # Example future API
    # @abstractmethod
    # def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    #     ...


# ------------------------------------------------------------------
# Concrete Weaviate implementation
# ------------------------------------------------------------------
class WeaviateDB(VectorDatabase):
    def __init__(self, host: str = "localhost", port: int = 8080, grpc_port: int = 50051):
        self.client = weaviate.connect_to_local(host, port, grpc_port)

    def create_schema(self, class_name: str = "KnowledgeChunk") -> str:
        schema = {
            "class": class_name,
            "properties": [
                {"name": "text", "dataType": ["text"]},
                {"name": "source", "dataType": ["string"]},
                {"name": "Channel_Name", "dataType": ["string"]},
                {"name": "Channel_Members", "dataType": ["string"]},
                {"name": "Message_Date", "dataType": ["string"]},
                {"name": "Message_Author", "dataType": ["string"]},
                {"name": "Keywords", "dataType": ["string"]},
                {"name": "Summary", "dataType": ["text"]},
            ],
            "vectorizer": "text2vec-transformers",
        }
        if self.client.schema.exists(class_name):
            self.client.schema.delete_class(class_name)
        self.client.schema.create_class(schema)
        return class_name


    def ingest_documents(self, root: Path, class_name: str = "KnowledgeChunk") -> None:
        splitter = TextProcessor.chunk_text
        total = 0
        files = list(root.glob("*.txt"))

        if not files:
            print(f"⚠ No .txt files found in {root.resolve()}")
            return

        with self.client.batch.fixed_size(batch_size=100) as batch:
            for txt in files:
                meta, body = TextProcessor.parse_file(txt)
                if not body:
                    continue
                chunks = splitter(body)
                for chunk in chunks:
                    obj = {
                        "text": chunk,
                        "source": txt.name,
                        **TextProcessor.process_meta(meta)
                    }
                    batch.add_object(properties=obj, collection=class_name)
                    total += 1
        print(f"✅ Weaviate ingested {total} chunks from {len(files)} files")


    def ingest_documents_old(self, root: Path, class_name: str = "KnowledgeChunk") -> None:
        splitter = TextProcessor.chunk_text
        total = 0
        files = list(root.glob("*.txt"))

        if not files:
            print("⚠️  No .txt files found in", root.resolve())
            return

        with self.client.batch.fixed_size(batch_size=100) as batch: 
            for txt in files:
                meta, body = TextProcessor.parse_file(txt)
                if not body:
                    continue
                chunks = splitter(body)
                for chunk in chunks:
                    obj = {"text": str(chunk), "source": str(txt.name), **meta}
                    batch.add_object(obj, class_name)
                    total += 1
            batch.flush()

        print(f"✅ Weaviate ingested {total} chunks from {len(files)} files")


# read .env at module import time
#load_dotenv()

# ------------------------------------------------------------------
# 4.  Usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv()
    root = Path(os.getenv("EXPORT_FOLDER", "data"))
    db = WeaviateDB(host="localhost",port=8080, grpc_port=50051)
    db.ingest_documents(root)
    db.client.close()