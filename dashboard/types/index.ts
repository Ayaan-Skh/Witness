// dashboard/src/types/index.ts
// Mirror of the FastAPI Pydantic response models.
// Any change to api/main.py response shapes must be reflected here.

export type ConfidenceTier = "LOW" | "MEDIUM" | "HIGH"
export type BriefStatus    = "DRAFT" | "REVIEWED" | "PUBLISHED"
export type AnomalySource  = "SATELLITE" | "GDELT" | "PROCUREMENT"

export interface BriefSummary {
  brief_id:             string
  region_id:            string
  time_window_start:    string
  time_window_end:      string
  confidence_score:     number
  confidence_tier:      ConfidenceTier
  contributing_streams: AnomalySource[]
  status:               BriefStatus
  created_at:           string
}

export interface BriefDetail extends BriefSummary {
  evidence:           Record<string, unknown>
  agent_reasoning:    string
  historical_context: string
  reviewer_notes:     string
}

export interface AnomalyEvent {
  event_id:        string
  source:          AnomalySource
  region_id:       string
  country_code:    string
  lat:             number
  lng:             number
  timestamp:       string
  signal_type:     string
  intensity_score: number
  raw_data:        Record<string, unknown>
  metadata:        Record<string, unknown>
  detected_at:     string
}

export interface RegionOut {
  region_id:      string
  name:           string
  country_code:   string
  bbox:           [number, number, number, number]
  centroid_lat:   number
  centroid_lng:   number
  anomaly_counts: Record<AnomalySource, number>
}

export interface PaginatedResponse<T> {
  items:     T[]
  total:     number
  page:      number
  page_size: number
  has_more:  boolean
}

// Map marker colours by confidence tier
export const TIER_COLOR: Record<ConfidenceTier, string> = {
  LOW:    "#FBBF24",  // amber
  MEDIUM: "#F97316",  // orange
  HIGH:   "#EF4444",  // red
}

export const SOURCE_COLOR: Record<AnomalySource, string> = {
  SATELLITE:   "#60A5FA",  // blue
  GDELT:       "#A78BFA",  // purple
  PROCUREMENT: "#34D399",  // green
}