"""Pydantic response models — must match dashboard/types/index.ts."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

ConfidenceTier = Literal["LOW", "MEDIUM", "HIGH"]
BriefStatus = Literal["DRAFT", "REVIEWED", "PUBLISHED"]
AnomalySource = Literal["SATELLITE", "GDELT", "PROCUREMENT"]

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    has_more: bool


class BriefSummary(BaseModel):
    brief_id: str
    region_id: str
    time_window_start: datetime
    time_window_end: datetime
    confidence_score: float
    confidence_tier: ConfidenceTier
    contributing_streams: list[AnomalySource]
    status: BriefStatus
    created_at: datetime


class BriefDetail(BriefSummary):
    evidence: dict[str, Any] = Field(default_factory=dict)
    agent_reasoning: str = ""
    historical_context: str = ""
    reviewer_notes: str = ""


class AnomalyEventOut(BaseModel):
    event_id: str
    source: AnomalySource
    region_id: str
    country_code: str
    lat: float
    lng: float
    timestamp: datetime
    signal_type: str
    intensity_score: float
    raw_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    detected_at: datetime


class RegionOut(BaseModel):
    region_id: str
    name: str
    country_code: str
    bbox: tuple[float, float, float, float]
    centroid_lat: float
    centroid_lng: float
    anomaly_counts: dict[str, int]
