// src/components/MapView.tsx
"use client"
import { useRef, useEffect, useState } from "react"
import maplibregl from "maplibre-gl"
import "maplibre-gl/dist/maplibre-gl.css"
import { useRouter } from "next/navigation"
import type { BriefSummary, RegionOut, ConfidenceTier } from "@/types"
import { TIER_COLOR } from "@/types"

const MAPTILER_KEY = process.env.NEXT_PUBLIC_MAPTILER_API_KEY?.trim() ?? ""
const HAS_MAP_KEY = MAPTILER_KEY.length > 0
const MAP_STYLE = HAS_MAP_KEY
  ? `https://api.maptiler.com/maps/dataviz-dark/style.json?key=${MAPTILER_KEY}`
  : ""

interface Props {
  briefs:  BriefSummary[]
  regions: RegionOut[]
}

export default function MapView({ briefs, regions }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const mapRef       = useRef<maplibregl.Map | null>(null)
  const markersRef   = useRef<maplibregl.Marker[]>([])
  const router       = useRouter()
  const [mapReady, setMapReady] = useState(false)

  useEffect(() => {
    if (!HAS_MAP_KEY || mapRef.current || !containerRef.current) return

    mapRef.current = new maplibregl.Map({
      container: containerRef.current,
      style:     MAP_STYLE,
      center:    [38, 14],
      zoom:      2.8,
    })

    mapRef.current.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right")
    mapRef.current.addControl(new maplibregl.ScaleControl(), "bottom-left")

    mapRef.current.on("load", () => setMapReady(true))

    return () => {
      mapRef.current?.remove()
      mapRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!mapReady || !mapRef.current) return

    markersRef.current.forEach((m) => m.remove())
    markersRef.current = []

    briefs.forEach((brief) => {
      const region = regions.find((r) => r.region_id === brief.region_id)
      if (!region) return

      const color = TIER_COLOR[brief.confidence_tier]
      const isHigh = brief.confidence_tier === "HIGH"

      const el = document.createElement("div")
      el.style.cssText = "position:relative;width:24px;height:24px;cursor:pointer;"

      if (isHigh) {
        const ring = document.createElement("div")
        ring.className = "animate-ping-slow"
        ring.style.cssText = `
          position:absolute;inset:0;border-radius:50%;
          background:${color};opacity:0.35;
        `
        el.appendChild(ring)
      }

      const dot = document.createElement("div")
      dot.style.cssText = `
        position:absolute;
        top:50%;left:50%;
        transform:translate(-50%,-50%);
        width:${isHigh ? 14 : 12}px;
        height:${isHigh ? 14 : 12}px;
        border-radius:50%;
        background:${color};
        border:2px solid rgba(255,255,255,0.25);
        box-shadow:0 0 10px ${color}90,0 0 20px ${color}40;
        transition:transform 0.15s ease;
      `
      el.appendChild(dot)

      el.addEventListener("mouseenter", () => { dot.style.transform = "translate(-50%,-50%) scale(1.3)" })
      el.addEventListener("mouseleave", () => { dot.style.transform = "translate(-50%,-50%) scale(1)" })

      const regionName = brief.region_id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
      const streams    = brief.contributing_streams.join(" · ")
      const score      = (brief.confidence_score * 100).toFixed(0)
      const dateStr    = new Date(brief.time_window_end).toLocaleDateString("en-GB", {
        day: "2-digit", month: "short", year: "numeric",
      })

      const popup = new maplibregl.Popup({ offset: 14, closeButton: false, maxWidth: "240px" })
        .setHTML(`
          <div style="
            background:#111827;border:1px solid #374151;border-radius:10px;
            padding:12px 14px;font-family:system-ui,sans-serif;min-width:200px;
          ">
            <div style="font-weight:600;color:#f9fafb;font-size:13px;margin-bottom:6px;">
              ${regionName}
            </div>
            <div style="font-size:11px;color:#9ca3af;margin-bottom:3px;">
              Confidence:
              <span style="color:${color};font-weight:600;margin-left:4px;">
                ${brief.confidence_tier} (${score}%)
              </span>
            </div>
            <div style="font-size:11px;color:#9ca3af;margin-bottom:3px;">
              Sources: <span style="color:#e5e7eb;">${streams}</span>
            </div>
            <div style="font-size:11px;color:#9ca3af;margin-bottom:10px;">
              Until: <span style="color:#e5e7eb;">${dateStr}</span>
            </div>
            <div style="
              font-size:11px;text-align:center;padding:5px 10px;border-radius:6px;
              background:${color}22;color:${color};border:1px solid ${color}44;cursor:pointer;
            ">
              Open brief →
            </div>
          </div>
        `)

      const marker = new maplibregl.Marker({ element: el })
        .setLngLat([region.centroid_lng, region.centroid_lat])
        .setPopup(popup)
        .addTo(mapRef.current!)

      el.addEventListener("click", () => router.push(`/briefs/${brief.brief_id}`))
      markersRef.current.push(marker)
    })
  }, [mapReady, briefs, regions, router])

  return (
    <div className="relative w-full h-full">
      <div ref={containerRef} className="w-full h-full" />

      <div className="absolute bottom-8 right-4 z-10 bg-gray-900/90 backdrop-blur-sm rounded-xl border border-gray-700/60 p-3 text-xs min-w-[130px]">
        <p className="text-gray-500 uppercase tracking-widest text-xs mb-2.5 font-medium">
          Confidence
        </p>
        {(["HIGH", "MEDIUM", "LOW"] as ConfidenceTier[]).map((tier) => (
          <div key={tier} className="flex items-center gap-2 mb-1.5 last:mb-0">
            <div
              className="w-3 h-3 rounded-full shrink-0"
              style={{
                background: TIER_COLOR[tier],
                boxShadow:  `0 0 6px ${TIER_COLOR[tier]}80`,
              }}
            />
            <span className="text-gray-300">{tier}</span>
          </div>
        ))}
        <div className="border-t border-gray-700/60 mt-2.5 pt-2.5 text-gray-600">
          {briefs.length} brief{briefs.length !== 1 ? "s" : ""} shown
        </div>
      </div>

      {!HAS_MAP_KEY && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-950/80 z-20">
          <div className="text-center text-gray-400 text-sm max-w-sm px-4">
            <p className="font-medium text-gray-200 mb-2">Map API key not set</p>
            <p className="mb-2">
              Get a free key at{" "}
              <a
                href="https://cloud.maptiler.com/account/keys/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-amber-400 hover:underline"
              >
                cloud.maptiler.com
              </a>
            </p>
            <p>
              Add to <code className="text-amber-400">dashboard/.env.local</code>:
            </p>
            <code className="mt-2 block text-xs text-amber-400/90">
              NEXT_PUBLIC_MAPTILER_API_KEY=your_key_here
            </code>
          </div>
        </div>
      )}
    </div>
  )
}
