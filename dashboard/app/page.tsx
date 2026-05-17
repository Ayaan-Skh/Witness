// src/app/page.tsx
// Main dashboard — server component that fetches data,
// renders sidebar + full-screen map.

import { Suspense } from "react"
import { getBriefs, getRegions } from "@/lib/api"
import MapView from "@/app/components/MapView"
import BriefsSidebar from "@/app/components/BriefsSidebar"
import FilterBar from "@/app/components/FilterBar"
import { AlertTriangle, Inbox } from "lucide-react"
import ApiBanner from "@/app/components/ApiBanner"

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export const revalidate = 60  // ISR — refresh every 60 seconds

interface PageProps {
  searchParams: Promise<{
    tier?:   string
    region?: string
    status?: string
  }>
}

export default async function HomePage({ searchParams }: PageProps) {
  const { tier, region, status } = await searchParams

  const emptyBriefs = { items: [], total: 0, page: 1, page_size: 50, has_more: false }
  let briefsData = emptyBriefs
  let regions: Awaited<ReturnType<typeof getRegions>> = []
  let apiError: string | null = null

  try {
    const health = await fetch(`${API_BASE}/health`, { cache: "no-store" })
    if (!health.ok) {
      apiError = `API health check failed (${health.status})`
    }
  } catch {
    apiError = `Cannot reach ${API_BASE}`
  }

  if (!apiError) {
    const [briefsResult, regionsResult] = await Promise.allSettled([
      getBriefs({ page: 1, page_size: 50, confidence_tier: tier, region_id: region, status }),
      getRegions(),
    ])
    if (briefsResult.status === "fulfilled") {
      // @ts-ignore
      briefsData = briefsResult.value
    } else {
      apiError = briefsResult.reason instanceof Error ? briefsResult.reason.message : "Failed to load briefs"
    }
    if (regionsResult.status === "fulfilled") {
      regions = regionsResult.value
    } else if (!apiError) {
      apiError = regionsResult.reason instanceof Error ? regionsResult.reason.message : "Failed to load regions"
    }
  }

  return (
    <main className="flex h-screen w-screen overflow-hidden flex-col">
      {apiError && <ApiBanner message={apiError} />}
      <div className="flex flex-1 min-h-0 overflow-hidden">

      {/* ── Sidebar ─────────────────────────────────────────────────── */}
      <aside className="w-80 flex flex-col bg-gray-900 border-r border-gray-800 z-10 shrink-0">

        {/* Logo bar */}
        <div className="px-4 py-4 border-b border-gray-800 flex items-center gap-2.5">
          <div className="w-7 h-7 rounded-lg bg-amber-500/20 border border-amber-500/30 flex items-center justify-center">
            <AlertTriangle className="text-amber-400" size={14} />
          </div>
          <span className="font-semibold tracking-wide text-sm text-gray-100">Witness</span>
          <span className="ml-auto text-xs text-gray-600">OSINT</span>
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-600 hover:text-gray-400 transition-colors"
          >
            <Inbox size={14} />
          </a>
        </div>

        {/* Filters */}
        <div className="px-3 py-3 border-b border-gray-800">
          <FilterBar
            regions={regions}
            currentTier={tier}
            currentRegion={region}
            currentStatus={status}
          />
        </div>

        {/* Count line */}
        <div className="px-4 py-2 text-xs text-gray-600 border-b border-gray-800/50">
          {briefsData.total === 0
            ? "No briefs found"
            : `${briefsData.total} brief${briefsData.total !== 1 ? "s" : ""}${briefsData.has_more ? " (showing 50)" : ""}`}
        </div>

        {/* Brief list */}
        <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-700">
          <Suspense
            fallback={
              <div className="p-6 text-center text-gray-600 text-sm">Loading briefs…</div>
            }
          >
            <BriefsSidebar briefs={briefsData.items} />
          </Suspense>
        </div>

        {/* Footer disclaimer */}
        <div className="px-4 py-3 border-t border-gray-800 text-xs text-gray-700 leading-relaxed">
          Evidence only. Human review required before action.
          This system does not verify human rights violations.
        </div>
      </aside>

      {/* ── Map ─────────────────────────────────────────────────────── */}
      <div className="flex-1 relative">
        <MapView briefs={briefsData.items} regions={regions} />
      </div>

      </div>
    </main>
  )
}