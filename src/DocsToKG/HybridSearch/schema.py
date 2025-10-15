"""OpenSearch schema management for hybrid chunk documents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional

from .config import ChunkingConfig


@dataclass(slots=True)
class OpenSearchIndexTemplate:
    """Representation of an OpenSearch index template body."""

    name: str
    namespace: str
    body: Mapping[str, Any]
    chunking: ChunkingConfig

    def asdict(self) -> Dict[str, Any]:
        """Convert the template to a serializable dictionary payload.

        Args:
            None

        Returns:
            Dictionary representation suitable for persistence or logging.
        """
        return {
            "name": self.name,
            "namespace": self.namespace,
            "body": dict(self.body),
            "chunking": {
                "max_tokens": self.chunking.max_tokens,
                "overlap": self.chunking.overlap,
            },
        }


class OpenSearchSchemaManager:
    """Bootstrap and track index templates per namespace."""

    def __init__(self) -> None:
        self._templates: MutableMapping[str, OpenSearchIndexTemplate] = {}

    def bootstrap_template(
        self, namespace: str, chunking: Optional[ChunkingConfig] = None
    ) -> OpenSearchIndexTemplate:
        """Create and register an index template for a namespace.

        Args:
            namespace: Namespace identifier the template should serve.
            chunking: Optional chunking configuration override.

        Returns:
            Newly created `OpenSearchIndexTemplate` instance.
        """
        chunking_config = chunking or ChunkingConfig()
        body = {
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1,
                },
                "analysis": {
                    "analyzer": {
                        "default": {
                            "type": "standard",
                        }
                    }
                },
            },
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "namespace": {"type": "keyword"},
                    "vector_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "splade": {"type": "rank_features"},
                    "token_count": {"type": "integer"},
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "author": {"type": "keyword"},
                            "tags": {"type": "keyword"},
                            "published_at": {"type": "date"},
                            "acl": {"type": "keyword"},
                        },
                    },
                },
            },
        }
        name = f"hybrid-chunks-{namespace}"
        template = OpenSearchIndexTemplate(
            name=name, namespace=namespace, body=body, chunking=chunking_config
        )
        self._templates[namespace] = template
        return template

    def get_template(self, namespace: str) -> Optional[OpenSearchIndexTemplate]:
        """Retrieve a registered template for the given namespace.

        Args:
            namespace: Namespace identifier to look up.

        Returns:
            Matching `OpenSearchIndexTemplate`, or None if not registered.
        """
        return self._templates.get(namespace)

    def list_templates(self) -> Mapping[str, OpenSearchIndexTemplate]:
        """Return a copy of all registered templates by namespace.

        Args:
            None

        Returns:
            Mapping from namespace to `OpenSearchIndexTemplate`.
        """
        return dict(self._templates)
