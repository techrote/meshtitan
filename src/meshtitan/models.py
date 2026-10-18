from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    cancelled = "cancelled"


TERMINAL_STATUSES = {JobStatus.succeeded, JobStatus.failed, JobStatus.cancelled}


class AssetRecord(BaseModel):
    asset_id: str
    original_name: str
    media_type: str | None = None
    bytes: int
    sha256: str
    path: str


class JobRequest(BaseModel):
    backend: str
    asset_id: str
    output_format: str = Field(default="glb", pattern=r"^(glb|obj|ply|stl)$")
    parameters: dict[str, Any] = Field(default_factory=dict)
    source_name: str | None = Field(default=None, max_length=255)
    view_assets: dict[str, str] = Field(default_factory=dict)
    parent_job_id: str | None = None


class JobRecord(BaseModel):
    job_id: str
    backend: str
    asset_id: str
    output_format: str
    parameters: dict[str, Any]
    status: JobStatus = JobStatus.queued
    created_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    cancellation_requested: bool = False
    result: dict[str, Any] | None = None
    batch_id: str | None = None
    batch_name: str | None = None
    batch_index: int | None = None
    source_name: str | None = None
    view_assets: dict[str, str] = Field(default_factory=dict)
    parent_job_id: str | None = None


class BatchRequest(BaseModel):
    name: str = Field(default="Untitled batch", min_length=1, max_length=100)
    jobs: list[JobRequest] = Field(min_length=1, max_length=200)


class CapabilityResponse(BaseModel):
    runtime_version: str
    gpu: dict[str, Any]
    backends: dict[str, Any]
    max_upload_mib: int = 32
