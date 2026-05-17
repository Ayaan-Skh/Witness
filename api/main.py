from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from api import demo_data, readers
from api.schemas import (
    AnomalyEventOut,
    BriefDetail,
    BriefSummary,
    PaginatedResponse,
    RegionOut,
)
from config import API_HOST, API_PORT, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from db import get_db, init_pool


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        init_pool()
    except Exception as exc:
        print(f"Warning: database pool not initialized ({exc}). API will return errors until DB is up.")
    yield


app = FastAPI(title="Witness API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.4:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_demo_mode: Optional[bool] = None


def _use_demo() -> bool:
    global _demo_mode
    if _demo_mode is not None:
        return _demo_mode
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        _demo_mode = False
    except Exception:
        _demo_mode = True
    return _demo_mode


def _filter_demo_briefs(
  items: list[dict],
  *,
  confidence_tier: Optional[str],
  region_id: Optional[str],
  status: Optional[str],
) -> list[dict]:
    out = items
    if confidence_tier:
        out = [b for b in out if b["confidence_tier"] == confidence_tier]
    if region_id:
        out = [b for b in out if b["region_id"] == region_id]
    if status:
        out = [b for b in out if b["status"] == status]
    return out


@app.get("/health")
def health():
    return {"status": "ok", "demo_mode": _use_demo()}


@app.get("/regions", response_model=list[RegionOut])
def get_regions():
    if _use_demo():
        return demo_data.REGIONS
    with get_db() as conn:
        return readers.list_regions(conn)


@app.get("/briefs", response_model=PaginatedResponse[BriefSummary])
def get_briefs(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    confidence_tier: Optional[str] = Query(None, alias="confidence_tier"),
    region_id: Optional[str] = None,
    status: Optional[str] = None,
):
    if _use_demo():
        filtered = _filter_demo_briefs(
            demo_data.BRIEFS,
            confidence_tier=confidence_tier,
            region_id=region_id,
            status=status,
        )
        total = len(filtered)
        start = (page - 1) * page_size
        page_items = filtered[start : start + page_size]
        summaries = [{k: v for k, v in b.items() if k not in ("evidence", "agent_reasoning", "historical_context", "reviewer_notes")} for b in page_items]
        return PaginatedResponse(
            items=summaries,
            total=total,
            page=page,
            page_size=page_size,
            has_more=(page * page_size) < total,
        )

    with get_db() as conn:
        items, total = readers.list_briefs(
            conn,
            page=page,
            page_size=page_size,
            confidence_tier=confidence_tier,
            region_id=region_id,
            status=status,
        )
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


@app.get("/briefs/{brief_id}", response_model=BriefDetail)
def get_brief(brief_id: str):
    if _use_demo():
        brief = demo_data.BRIEF_BY_ID.get(brief_id)
        if not brief:
            raise HTTPException(status_code=404, detail="Brief not found")
        return brief

    with get_db() as conn:
        brief = readers.get_brief(conn, brief_id)
    if not brief:
        raise HTTPException(status_code=404, detail="Brief not found")
    return brief


@app.get("/anomalies", response_model=PaginatedResponse[AnomalyEventOut])
def get_anomalies(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    source: Optional[str] = None,
    region_id: Optional[str] = None,
    min_intensity: Optional[float] = None,
):
    if _use_demo():
        return PaginatedResponse(items=[], total=0, page=page, page_size=page_size, has_more=False)

    with get_db() as conn:
        items, total = readers.list_anomalies(
            conn,
            page=page,
            page_size=page_size,
            source=source,
            region_id=region_id,
            min_intensity=min_intensity,
        )
    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        has_more=(page * page_size) < total,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
