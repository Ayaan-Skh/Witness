"use client"

import type { ConfidenceTier } from "@/types"
import { TIER_COLOR } from "@/types"

interface Props {
  tier: ConfidenceTier
  score: number
  small?: boolean
}

export default function ConfidenceBadge({ tier, score, small }: Props) {
  const color = TIER_COLOR[tier]
  const pct = Math.round(Math.min(1, Math.max(0, score)) * 100)

  const base =
    "inline-flex items-center font-medium border shrink-0 " +
    (small ? "text-[10px] px-1.5 py-0.5 rounded" : "text-xs px-2.5 py-1 rounded-full")

  return (
    <span
      className={base}
      style={{
        color,
        borderColor: `${color}55`,
        background: `${color}14`,
      }}
    >
      {tier}
      {!small && <span className="opacity-80 ml-1">· {pct}%</span>}
    </span>
  )
}
