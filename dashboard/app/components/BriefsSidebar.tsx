// src/components/BriefsSidebar.tsx
"use client"
import Link from "next/link"
import { format } from "date-fns"
import type { BriefSummary } from "@/types"
import { TIER_COLOR } from "@/types"
import ConfidenceBadge from "@/app/components/ConfidenceBadge"

const SOURCE_SHORT: Record<string, string> = {
  SATELLITE:   "SAT",
  GDELT:       "NEWS",
  PROCUREMENT: "PROC",
}

export default function BriefsSidebar({ briefs }: { briefs: BriefSummary[] }) {
  if (briefs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-48 text-gray-600 text-sm gap-2 px-6 text-center">
        <span>No investigation briefs match current filters.</span>
        <span className="text-xs text-gray-700">Try removing filters or broadening the date range.</span>
      </div>
    )
  }

  return (
    <ul className="divide-y divide-gray-800/60">
      {briefs.map((brief) => {
        const color = TIER_COLOR[brief.confidence_tier]
        return (
          <li key={brief.brief_id} className="relative">
            <Link
              href={`/briefs/${brief.brief_id}`}
              className="block px-4 py-3.5 hover:bg-gray-800/40 transition-colors group"
            >
              {/* Accent bar — coloured by tier, visible on hover */}
              <span
                className="absolute left-0 top-0 w-0.5 h-full opacity-0 group-hover:opacity-100 transition-opacity rounded-r"
                style={{ background: color }}
              />

              {/* Region + badge */}
              <div className="flex items-start justify-between gap-2 mb-1.5">
                <span className="text-sm font-medium text-gray-100 leading-snug group-hover:text-white transition-colors">
                  {brief.region_id
                    .replace(/_/g, " ")
                    .replace(/\b\w/g, (c) => c.toUpperCase())}
                </span>
                {/* @ts-ignore */}
                <ConfidenceBadge tier={brief.confidence_tier} score={brief.confidence_score} small />
              </div>

              {/* Time window */}
              <p className="text-xs text-gray-500 mb-2">
                {format(new Date(brief.time_window_start), "d MMM")}
                {" → "}
                {format(new Date(brief.time_window_end), "d MMM yyyy")}
              </p>

              {/* Source pills + status */}
              <div className="flex items-center gap-1.5 flex-wrap">
                {brief.contributing_streams.map((s) => (
                  <span
                    key={s}
                    className="text-xs px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 font-mono"
                  >
                    {SOURCE_SHORT[s] ?? s}
                  </span>
                ))}
                <span
                  className="ml-auto text-xs"
                  style={{ color: brief.status === "PUBLISHED" ? "#34D399" : "#6b7280" }}
                >
                  {brief.status}
                </span>
              </div>
            </Link>
          </li>
        )
      })}
    </ul>
  )
}