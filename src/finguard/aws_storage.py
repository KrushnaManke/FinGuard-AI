"""Optional AWS S3 artifact upload using the caller's configured AWS identity."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def upload_artifact(path: str | Path, bucket: str, key: str) -> str:
    if not bucket or not key or key.startswith("/") or ".." in Path(key).parts:
        raise ValueError("Provide a bucket and a safe relative object key.")
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact not found: {artifact}")
    load_dotenv()
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("Install the AWS SDK with `pip install -e '.[ops]'`.") from error
    boto3.client("s3").upload_file(str(artifact), bucket, key)
    return f"s3://{bucket}/{key}"


def download_artifact(bucket: str, key: str, path: str | Path) -> Path:
    if not bucket or not key or key.startswith("/") or ".." in Path(key).parts:
        raise ValueError("Provide a bucket and a safe relative object key.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("Install the AWS SDK with `pip install -e '.[ops]'`.") from error
    load_dotenv()
    boto3.client("s3").download_file(bucket, key, str(destination))
    return destination
