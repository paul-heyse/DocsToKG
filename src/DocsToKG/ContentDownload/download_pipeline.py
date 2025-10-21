"""
Download Pipeline Integration

Orchestrates the complete download workflow:
- Loads Pydantic v2 configuration
- Builds resolver chain
- Coordinates artifact resolution and acquisition
- Manages telemetry and manifests
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from DocsToKG.ContentDownload.config import ContentDownloadConfig
from DocsToKG.ContentDownload.resolvers.registry_v2 import build_resolvers

_LOGGER = logging.getLogger(__name__)


class DownloadPipeline:
    """
    Main download pipeline orchestrator.

    Coordinates:
    - Configuration loading and validation
    - Resolver instantiation and ordering
    - Artifact processing
    - Telemetry and manifest recording
    """

    def __init__(
        self,
        config: ContentDownloadConfig,
        resolvers: Optional[list[Any]] = None,
    ):
        """
        Initialize pipeline.

        Args:
            config: ContentDownloadConfig instance
            resolvers: Optional pre-built resolver list; if None, built from config
        """
        self.config = config
        self.resolvers = resolvers or build_resolvers(config)
        _LOGGER.info(
            f"Pipeline initialized with {len(self.resolvers)} resolvers "
            f"and config hash {config.config_hash()[:8]}..."
        )

    def process_artifact(self, artifact: Any) -> dict[str, Any]:
        """
        Process a single artifact through the resolver chain.

        Args:
            artifact: Work item/artifact to resolve (dict or domain object)

        Returns:
            Outcome dict with resolution result and metadata
        """
        _LOGGER.debug(f"Processing artifact: {artifact}")

        for resolver in self.resolvers:
            try:
                # Try iter_urls pattern (existing system)
                if hasattr(resolver, "iter_urls"):
                    results = list(resolver.iter_urls(None, self.config, artifact))
                    if results:
                        _LOGGER.info(
                            f"Resolver {resolver.__class__.__name__} yielded {len(results)} results"
                        )
                        return {
                            "status": "resolved",
                            "results": results,
                            "resolver": resolver.__class__.__name__,
                        }

                # Try resolve pattern (new system)
                elif hasattr(resolver, "resolve"):
                    results = resolver.resolve(artifact)
                    if results:
                        _LOGGER.info(
                            f"Resolver {resolver.__class__.__name__} resolved {len(results)} results"
                        )
                        return {
                            "status": "resolved",
                            "results": results,
                            "resolver": resolver.__class__.__name__,
                        }

            except Exception as e:
                _LOGGER.warning(f"Resolver {resolver.__class__.__name__} failed: {e}")
                continue

        _LOGGER.warning("No resolver succeeded for artifact")
        return {"status": "unresolved", "results": [], "resolver": None}

    def process_artifacts(self, artifacts: Iterable[Any]) -> Iterable[dict[str, Any]]:
        """
        Process multiple artifacts through the pipeline.

        Args:
            artifacts: Iterable of artifacts to process

        Yields:
            Outcome dict for each artifact
        """
        for artifact in artifacts:
            yield self.process_artifact(artifact)


def build_pipeline(
    config_path: Optional[str] = None,
    config: Optional[ContentDownloadConfig] = None,
    cli_overrides: Optional[dict[str, Any]] = None,
) -> DownloadPipeline:
    """
    Build a download pipeline from configuration.

    Args:
        config_path: Optional path to config file (YAML/JSON)
        config: Optional pre-loaded ContentDownloadConfig
        cli_overrides: Optional CLI overrides dict

    Returns:
        Initialized DownloadPipeline instance

    Raises:
        ValueError: If neither config_path nor config provided
    """
    if config is None:
        if config_path is None:
            raise ValueError("Must provide either config_path or config")
        from DocsToKG.ContentDownload.config import load_config

        config = load_config(path=config_path, cli_overrides=cli_overrides)

    return DownloadPipeline(config)
