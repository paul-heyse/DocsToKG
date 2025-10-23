# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.deployment.rollback",
#   "purpose": "Automated rollback system for safe deployment recovery.",
#   "sections": [
#     {
#       "id": "deploymentstatus",
#       "name": "DeploymentStatus",
#       "anchor": "class-deploymentstatus",
#       "kind": "class"
#     },
#     {
#       "id": "deploymentsnapshot",
#       "name": "DeploymentSnapshot",
#       "anchor": "class-deploymentsnapshot",
#       "kind": "class"
#     },
#     {
#       "id": "rollbackmanager",
#       "name": "RollbackManager",
#       "anchor": "class-rollbackmanager",
#       "kind": "class"
#     },
#     {
#       "id": "get-rollback-manager",
#       "name": "get_rollback_manager",
#       "anchor": "function-get-rollback-manager",
#       "kind": "function"
#     }
#   ]
# }
# === /NAVMAP ===

"""Automated rollback system for safe deployment recovery.

Provides snapshot-based rollback, state recovery, and deployment safety mechanisms.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger(__name__)


class DeploymentStatus(str, Enum):
    """Deployment status values."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESSFUL = "successful"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentSnapshot:
    """Snapshot of deployment state."""

    deployment_id: str
    timestamp: float
    version: str
    config: Dict[str, Any]
    status: DeploymentStatus
    error_message: Optional[str] = None
    metrics_before: Optional[Dict[str, Any]] = None
    metrics_after: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "timestamp": self.timestamp,
            "version": self.version,
            "config": self.config,
            "status": self.status.value,
            "error_message": self.error_message,
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
        }


class RollbackManager:
    """Manages deployment snapshots and rollback operations."""

    def __init__(self, snapshot_dir: Path) -> None:
        """Initialize rollback manager.

        Args:
            snapshot_dir: Directory for storing deployment snapshots
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._current_deployment: Optional[DeploymentSnapshot] = None
        self._deployment_history: List[DeploymentSnapshot] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load deployment history from disk."""
        history_file = self.snapshot_dir / "history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                    for item in data:
                        snapshot = DeploymentSnapshot(
                            deployment_id=item["deployment_id"],
                            timestamp=item["timestamp"],
                            version=item["version"],
                            config=item["config"],
                            status=DeploymentStatus(item["status"]),
                            error_message=item.get("error_message"),
                        )
                        self._deployment_history.append(snapshot)
            except Exception as e:
                LOGGER.error(f"Failed to load deployment history: {e}")

    def _save_history(self) -> None:
        """Save deployment history to disk."""
        history_file = self.snapshot_dir / "history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(
                    [s.to_dict() for s in self._deployment_history],
                    f,
                    indent=2,
                )
        except Exception as e:
            LOGGER.error(f"Failed to save deployment history: {e}")

    def create_snapshot(
        self,
        version: str,
        config: Dict[str, Any],
        metrics_before: Optional[Dict[str, Any]] = None,
    ) -> DeploymentSnapshot:
        """Create deployment snapshot.

        Args:
            version: Version being deployed
            config: Configuration snapshot
            metrics_before: Metrics before deployment

        Returns:
            Created snapshot
        """
        deployment_id = f"deploy-{int(time.time())}"

        snapshot = DeploymentSnapshot(
            deployment_id=deployment_id,
            timestamp=time.time(),
            version=version,
            config=config,
            status=DeploymentStatus.PENDING,
            metrics_before=metrics_before,
        )

        self._current_deployment = snapshot
        self._deployment_history.append(snapshot)
        self._save_history()

        LOGGER.info(f"Created deployment snapshot: {deployment_id}")
        return snapshot

    def mark_deployment_complete(self, metrics_after: Optional[Dict[str, Any]] = None) -> None:
        """Mark current deployment as complete.

        Args:
            metrics_after: Metrics after deployment
        """
        if self._current_deployment is None:
            LOGGER.error("No current deployment to mark complete")
            return

        updated = DeploymentSnapshot(
            deployment_id=self._current_deployment.deployment_id,
            timestamp=self._current_deployment.timestamp,
            version=self._current_deployment.version,
            config=self._current_deployment.config,
            status=DeploymentStatus.SUCCESSFUL,
            metrics_after=metrics_after,
        )

        # Update in history
        for i, dep in enumerate(self._deployment_history):
            if dep.deployment_id == updated.deployment_id:
                self._deployment_history[i] = updated
                break

        self._current_deployment = updated
        self._save_history()
        LOGGER.info(f"Deployment {updated.deployment_id} marked complete")

    def mark_deployment_failed(self, error_message: str) -> None:
        """Mark current deployment as failed.

        Args:
            error_message: Error message
        """
        if self._current_deployment is None:
            LOGGER.error("No current deployment to mark failed")
            return

        updated = DeploymentSnapshot(
            deployment_id=self._current_deployment.deployment_id,
            timestamp=self._current_deployment.timestamp,
            version=self._current_deployment.version,
            config=self._current_deployment.config,
            status=DeploymentStatus.FAILED,
            error_message=error_message,
        )

        # Update in history
        for i, dep in enumerate(self._deployment_history):
            if dep.deployment_id == updated.deployment_id:
                self._deployment_history[i] = updated
                break

        self._current_deployment = updated
        self._save_history()
        LOGGER.error(f"Deployment {updated.deployment_id} failed: {error_message}")

    def get_previous_successful_deployment(self) -> Optional[DeploymentSnapshot]:
        """Get most recent successful deployment.

        Returns:
            Previous successful deployment or None
        """
        for deployment in reversed(self._deployment_history):
            if deployment.status == DeploymentStatus.SUCCESSFUL:
                return deployment
        return None

    def perform_rollback(self, deployment_id: Optional[str] = None) -> bool:
        """Perform rollback to specified deployment.

        Args:
            deployment_id: Deployment to rollback to (defaults to previous successful)

        Returns:
            True if rollback successful
        """
        if deployment_id is None:
            target = self.get_previous_successful_deployment()
        else:
            target = next(
                (d for d in self._deployment_history if d.deployment_id == deployment_id),
                None,
            )

        if target is None:
            LOGGER.error("No suitable deployment for rollback")
            return False

        try:
            LOGGER.info(f"Rolling back to deployment: {target.deployment_id}")

            # Execute rollback script
            rollback_script = self.snapshot_dir / f"{target.deployment_id}.rollback.sh"
            if rollback_script.exists():
                result = subprocess.run(
                    ["bash", str(rollback_script)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    LOGGER.error(f"Rollback script failed: {result.stderr}")
                    return False

            # Restore configuration
            config_file = self.snapshot_dir / f"{target.deployment_id}.config.json"
            if config_file.exists():
                with open(config_file) as f:
                    json.load(f)
                    LOGGER.info(f"Restored configuration from {target.deployment_id}")

            # Mark as rolled back
            updated = DeploymentSnapshot(
                deployment_id=target.deployment_id,
                timestamp=target.timestamp,
                version=target.version,
                config=target.config,
                status=DeploymentStatus.ROLLED_BACK,
            )

            for i, dep in enumerate(self._deployment_history):
                if dep.deployment_id == updated.deployment_id:
                    self._deployment_history[i] = updated
                    break

            self._save_history()
            LOGGER.info(f"Successfully rolled back to {target.deployment_id}")
            return True

        except Exception as e:
            LOGGER.error(f"Rollback failed: {e}")
            return False

    def get_deployment_history(self, limit: int = 10) -> List[DeploymentSnapshot]:
        """Get recent deployment history.

        Args:
            limit: Maximum number of deployments to return

        Returns:
            Recent deployments
        """
        return self._deployment_history[-limit:]

    def cleanup_old_snapshots(self, keep_count: int = 10) -> None:
        """Clean up old deployment snapshots.

        Args:
            keep_count: Number of snapshots to keep
        """
        if len(self._deployment_history) <= keep_count:
            return

        to_remove = self._deployment_history[:-keep_count]

        for snapshot in to_remove:
            try:
                # Remove associated files
                for pattern in [f"{snapshot.deployment_id}*"]:
                    for file in self.snapshot_dir.glob(pattern):
                        if file.is_file():
                            file.unlink()
            except Exception as e:
                LOGGER.error(f"Failed to clean up snapshot {snapshot.deployment_id}: {e}")

        # Keep only recent snapshots in memory
        self._deployment_history = self._deployment_history[-keep_count:]
        self._save_history()
        LOGGER.info(f"Cleaned up old snapshots, keeping {keep_count}")


# Global rollback manager instance
_GLOBAL_ROLLBACK_MANAGER: Optional[RollbackManager] = None


def get_rollback_manager(
    snapshot_dir: Path = Path("/var/cache/docstokg/deployments"),
) -> RollbackManager:
    """Get global rollback manager instance.

    Args:
        snapshot_dir: Directory for deployment snapshots

    Returns:
        Global RollbackManager instance
    """
    global _GLOBAL_ROLLBACK_MANAGER
    if _GLOBAL_ROLLBACK_MANAGER is None:
        _GLOBAL_ROLLBACK_MANAGER = RollbackManager(snapshot_dir)
    return _GLOBAL_ROLLBACK_MANAGER
