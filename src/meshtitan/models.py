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


class CapabilityResponse(BaseModel):
    runtime_version: str
    gpu: dict[str, Any]
    backends: dict[str, Any]
