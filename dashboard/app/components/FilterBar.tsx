// src/components/FilterBar.tsx
"use client"
import { useRouter, usePathname } from "next/navigation"
import { SlidersHorizontal } from "lucide-react"
import type { RegionOut } from "@/types"

interface Props {
  regions:        RegionOut[]
  currentTier?:   string
  currentRegion?: string
  currentStatus?: string
}

export default function FilterBar({ regions, currentTier, currentRegion, currentStatus }: Props) {
  const router   = useRouter()
  const pathname = usePathname()

  const update = (key: string, value: string) => {
    const params = new URLSearchParams(typeof window !== "undefined" ? window.location.search : "")
    if (value) params.set(key, value)
    else        params.delete(key)
    router.push(`${pathname}?${params.toString()}`)
  }

  const clearAll = () => router.push(pathname)

  const hasFilters = !!(currentTier || currentRegion || currentStatus)

  const selectCls =
    "w-full text-xs bg-gray-800 border border-gray-700 rounded-lg px-2.5 py-1.5 " +
    "text-gray-200 focus:outline-none focus:border-gray-500 cursor-pointer transition-colors " +
    "hover:border-gray-600"

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between mb-1">
        <span className="flex items-center gap-1.5 text-xs text-gray-500">
          <SlidersHorizontal size={11} />
          Filters
        </span>
        {hasFilters && (
          <button
            onClick={clearAll}
            className="text-xs text-gray-600 hover:text-gray-400 transition-colors"
          >
            Clear all
          </button>
        )}
      </div>

      {/* Confidence tier */}
      <select
        className={selectCls}
        value={currentTier ?? ""}
        onChange={(e) => update("tier", e.target.value)}
      >
        <option value="">All confidence levels</option>
        <option value="HIGH">HIGH — 3 sources converging</option>
        <option value="MEDIUM">MEDIUM — 2 sources</option>
        <option value="LOW">LOW — 1 source</option>
      </select>

      {/* Region */}
      <select
        className={selectCls}
        value={currentRegion ?? ""}
        onChange={(e) => update("region", e.target.value)}
      >
        <option value="">All regions</option>
        {regions.map((r) => (
          <option key={r.region_id} value={r.region_id}>
            {r.name}
          </option>
        ))}
      </select>

      {/* Review status */}
      <select
        className={selectCls}
        value={currentStatus ?? ""}
        onChange={(e) => update("status", e.target.value)}
      >
        <option value="">All statuses</option>
        <option value="DRAFT">Draft — unreviewed</option>
        <option value="REVIEWED">Reviewed</option>
        <option value="PUBLISHED">Published</option>
      </select>
    </div>
  )
}