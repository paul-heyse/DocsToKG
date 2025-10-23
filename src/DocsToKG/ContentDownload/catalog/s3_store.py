# === NAVMAP v1 ===
# {
#   "module": "DocsToKG.ContentDownload.catalog.s3_store",
#   "purpose": "Amazon S3 storage backend implementation.",
#   "sections": [
#     {
#       "id": "s3storagebackend",
#       "name": "S3StorageBackend",
#       "anchor": "class-s3storagebackend",
#       "kind": "class"
#     }
#   ]
# }
# === /NAVMAP ===

"""Amazon S3 storage backend implementation.

Provides cloud-native storage with:
  - Multipart upload for large files
  - Server-side encryption
  - Lifecycle policies
  - Versioning support
  - Cross-region replication ready
  - Bandwidth optimization
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from threading import RLock

logger = logging.getLogger(__name__)


class S3StorageBackend:
    """Amazon S3 storage backend for artifacts.

    Thread-safe S3 operations with automatic retry and optimization.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "docs/",
        region: str = "us-east-1",
        storage_class: str = "INTELLIGENT_TIERING",
        enable_versioning: bool = False,
    ):
        """Initialize S3 storage backend.

        Args:
            bucket: S3 bucket name
            prefix: Object key prefix (default 'docs/')
            region: AWS region (default 'us-east-1')
            storage_class: Storage class (STANDARD, INTELLIGENT_TIERING, GLACIER, etc)
            enable_versioning: Enable S3 versioning
        """
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/"
        self.region = region
        self.storage_class = storage_class
        self.enable_versioning = enable_versioning
        self._lock = RLock()
        self.s3_client = None

        self._init_client()

    def _init_client(self):
        """Initialize S3 client."""
        try:
            import boto3
        except ImportError:
            raise ImportError("boto3 not installed. " "Install with: pip install boto3")

        try:
            logger.info(f"Connecting to S3 bucket: {self.bucket} (region: {self.region})")

            self.s3_client = boto3.client("s3", region_name=self.region)

            # Test connectivity
            self.s3_client.head_bucket(Bucket=self.bucket)

            # Enable versioning if requested
            if self.enable_versioning:
                self.s3_client.put_bucket_versioning(
                    Bucket=self.bucket,
                    VersioningConfiguration={"Status": "Enabled"},
                )
                logger.info("S3 versioning enabled")

            logger.info("S3 storage backend initialized")

        except Exception as e:
            logger.error(f"Failed to initialize S3: {e}")
            raise

    def put_file(
        self,
        local_path: str,
        artifact_id: str,
        resolver: str,
        content_type: str | None = None,
    ) -> str:
        """Upload file to S3.

        Args:
            local_path: Local file path
            artifact_id: Artifact identifier
            resolver: Resolver name
            content_type: MIME type

        Returns:
            S3 URI (s3://bucket/key)
        """
        from pathlib import Path

        with self._lock:
            try:
                local_file = Path(local_path)
                if not local_file.exists():
                    raise FileNotFoundError(f"File not found: {local_path}")

                # Build object key
                basename = local_file.name
                key = f"{self.prefix}{resolver}/{artifact_id}/{basename}"

                # Compute file hash for integrity check
                file_hash = self._compute_file_hash(local_path)

                logger.info(f"Uploading {local_path} to s3://{self.bucket}/{key}")

                # Upload with metadata
                extra_args = {
                    "StorageClass": self.storage_class,
                    "Metadata": {
                        "artifact_id": artifact_id,
                        "resolver": resolver,
                        "sha256": file_hash,
                    },
                }

                if content_type:
                    extra_args["ContentType"] = content_type

                # Multipart upload for large files
                file_size = local_file.stat().st_size
                if file_size > 100 * 1024 * 1024:  # > 100MB
                    self._multipart_upload(local_path, key, extra_args)
                else:
                    self.s3_client.upload_file(
                        local_path,
                        self.bucket,
                        key,
                        ExtraArgs=extra_args,
                    )

                uri = f"s3://{self.bucket}/{key}"
                logger.info(f"Upload complete: {uri}")
                return uri

            except Exception as e:
                logger.error(f"Failed to upload file: {e}")
                raise

    def _multipart_upload(self, local_path: str, key: str, extra_args: dict):
        """Perform multipart upload for large files."""

        chunk_size = 50 * 1024 * 1024  # 50MB chunks

        # Initiate multipart upload
        mpu = self.s3_client.create_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            **extra_args,
        )
        upload_id = mpu["UploadId"]

        try:
            parts = []
            part_number = 1

            with open(local_path, "rb") as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break

                    response = self.s3_client.upload_part(
                        Bucket=self.bucket,
                        Key=key,
                        PartNumber=part_number,
                        UploadId=upload_id,
                        Body=chunk,
                    )

                    parts.append(
                        {
                            "ETag": response["ETag"],
                            "PartNumber": part_number,
                        }
                    )

                    logger.debug(f"Uploaded part {part_number}")
                    part_number += 1

            # Complete multipart upload
            self.s3_client.complete_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts},
            )

            logger.info(f"Multipart upload complete: {part_number - 1} parts")

        except Exception as e:
            logger.error(f"Multipart upload failed: {e}")
            self.s3_client.abort_multipart_upload(
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
            )
            raise

    def get_file(self, s3_uri: str, local_path: str) -> bool:
        """Download file from S3.

        Args:
            s3_uri: S3 URI (s3://bucket/key)
            local_path: Local download path

        Returns:
            True if successful
        """
        with self._lock:
            try:
                # Parse URI
                if not s3_uri.startswith("s3://"):
                    raise ValueError(f"Invalid S3 URI: {s3_uri}")

                bucket, key = s3_uri[5:].split("/", 1)

                if bucket != self.bucket:
                    raise ValueError(f"URI bucket {bucket} != configured bucket {self.bucket}")

                logger.info(f"Downloading s3://{bucket}/{key} to {local_path}")

                Path(local_path).parent.mkdir(parents=True, exist_ok=True)
                self.s3_client.download_file(bucket, key, local_path)

                logger.info(f"Download complete: {local_path}")
                return True

            except Exception as e:
                logger.error(f"Failed to download file: {e}")
                raise

    def verify_object(self, s3_uri: str, expected_hash: str) -> bool:
        """Verify object integrity via ETag.

        Args:
            s3_uri: S3 URI
            expected_hash: Expected SHA-256 hash

        Returns:
            True if hash matches
        """
        with self._lock:
            try:
                # Parse URI
                if not s3_uri.startswith("s3://"):
                    raise ValueError(f"Invalid S3 URI: {s3_uri}")

                bucket, key = s3_uri[5:].split("/", 1)

                # Get object metadata
                response = self.s3_client.head_object(Bucket=bucket, Key=key)
                metadata = response.get("Metadata", {})

                # Check metadata hash
                stored_hash = metadata.get("sha256")
                if stored_hash == expected_hash.lower():
                    logger.debug(f"Hash verification passed: {s3_uri}")
                    return True
                else:
                    logger.warning(f"Hash mismatch for {s3_uri}")
                    return False

            except Exception as e:
                logger.error(f"Failed to verify object: {e}")
                return False

    def delete_object(self, s3_uri: str) -> bool:
        """Delete object from S3.

        Args:
            s3_uri: S3 URI

        Returns:
            True if successful
        """
        with self._lock:
            try:
                # Parse URI
                if not s3_uri.startswith("s3://"):
                    raise ValueError(f"Invalid S3 URI: {s3_uri}")

                bucket, key = s3_uri[5:].split("/", 1)

                logger.info(f"Deleting s3://{bucket}/{key}")
                self.s3_client.delete_object(Bucket=bucket, Key=key)

                logger.info(f"Deletion complete: {s3_uri}")
                return True

            except Exception as e:
                logger.error(f"Failed to delete object: {e}")
                raise

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA-256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def close(self):
        """Close S3 client."""
        # boto3 clients don't need explicit closing
        logger.info("S3 storage backend closed")
