"""Legacy resolver pipeline and configuration types.

⚠️  DEPRECATED: This module is maintained for backward compatibility only.
The new system uses:
  - download_pipeline.py for the modern DownloadPipeline orchestrator
  - config/models.py for Pydantic v2 configuration
  - registry_v2.py for the modern @register_v2 decorator pattern

This module will be removed after all dependents migrate to the modern architecture.
"""
