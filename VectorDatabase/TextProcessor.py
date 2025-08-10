from __future__ import annotations
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any

import tiktoken
import weaviate


# ------------------------------------------------------------------
# 1.  Pure text utilities
# ------------------------------------------------------------------
class TextProcessor:
    """
    Stateless helpers for parsing our header/body format
    and chunking the body with tiktoken.
    """

    @staticmethod
    def parse_file(path: Path) -> tuple[Dict[str, str], str]:
        raw = path.read_text(encoding="utf-8")
        header, sep, body = raw.partition("---")
        if not sep:
            raise ValueError(f"No '---' separator in {path}")
        metadata = {}
        for line in header.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                metadata[key.strip()] = val.strip()
        return metadata, body.strip()

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 20,
        overlap: int = 5,
        tokenizer_name: str = "cl100k_base",
    ) -> List[str]:
        enc = tiktoken.get_encoding(tokenizer_name)
        tokens = enc.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunks.append(enc.decode(tokens[start:end]))
            start += chunk_size - overlap
        return chunks

    @staticmethod
    def map_field_name(original_name: str) -> str:
        # Define a pattern for converting field names to valid GraphQL names
        graphql_name = re.sub(r'\s+', '_', original_name)  # Replace spaces with underscores
        return graphql_name

    @staticmethod
    def process_meta(meta: Dict[str, str]) -> Dict[str, str]:
        # Map field names to valid GraphQL names
        return {TextProcessor.map_field_name(key): value for key, value in meta.items()}