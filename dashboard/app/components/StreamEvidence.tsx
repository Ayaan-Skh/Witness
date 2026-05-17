// src/components/StreamEvidence.tsx
import { Satellite, Newspaper, FileText } from "lucide-react"
import type { AnomalySource } from "@/types"

const ICONS: Record<AnomalySource, React.ReactNode> = {
  SATELLITE:   <Satellite size={14} />,
  GDELT:       <Newspaper size={14} />,
  PROCUREMENT: <FileText  size={14} />,
}

const LABELS: Record<AnomalySource, string> = {
  SATELLITE:   "Satellite Imagery",
  GDELT:       "News Intelligence (GDELT)",
  PROCUREMENT: "Government Procurement (OCDS)",
}

const SOURCE_ACCENT: Record<AnomalySource, string> = {
  SATELLITE:   "#60A5FA",
  GDELT:       "#A78BFA",
  PROCUREMENT: "#34D399",
}

interface Props {
  evidence: Record<string, unknown>
  streams:  AnomalySource[]
}

export default function StreamEvidence({ evidence, streams }: Props) {
  if (!streams.length) {
    return <p className="text-sm text-gray-500">No evidence recorded.</p>
  }

  return (
    <div className="space-y-4">
      {streams.map((stream) => {
        const data   = (evidence[stream] ?? {}) as Record<string, unknown>
        const accent = SOURCE_ACCENT[stream]

        return (
          <div
            key={stream}
            className="rounded-xl border bg-gray-900 overflow-hidden"
            style={{ borderColor: `${accent}33` }}
          >
            {/* Header bar */}
            <div
              className="flex items-center gap-2 px-4 py-2.5 text-sm font-medium"
              style={{ background: `${accent}12`, color: accent, borderBottom: `1px solid ${accent}22` }}
            >
              {ICONS[stream]}
              {LABELS[stream]}
            </div>

            <div className="p-4 space-y-3">
              {/* Summary text */}
              {typeof data.summary === "string" && data.summary.trim() !== "" ? (
                <p className="text-sm text-gray-300 leading-relaxed">{data.summary}</p>
              ) : null}

              {/* Key metric grid — show up to 6 non-summary fields */}
              {(() => {
                const metrics = Object.entries(data)
                  .filter(([k]) => k !== "summary" && k !== "events" && k !== "top_concerning_themes")
                  .slice(0, 6)

                if (metrics.length === 0) return null

                return (
                  <div className="grid grid-cols-2 gap-2">
                    {metrics.map(([key, val]) => (
                      <div key={key} className="bg-gray-800 rounded-lg px-3 py-2">
                        <div className="text-xs text-gray-500 mb-0.5 capitalize">
                          {key.replace(/_/g, " ")}
                        </div>
                        <div className="text-sm text-gray-100 font-mono truncate">
                          {typeof val === "number"
                            ? val % 1 === 0 ? val.toString() : val.toFixed(3)
                            : String(val ?? "—")}
                        </div>
                      </div>
                    ))}
                  </div>
                )
              })()}

              {/* Concerning themes for GDELT */}
              {stream === "GDELT" && Array.isArray(data.top_concerning_themes) && data.top_concerning_themes.length > 0 && (
                <div>
                  <p className="text-xs text-gray-500 mb-1.5">Concerning CAMEO codes detected</p>
                  <div className="flex gap-1.5 flex-wrap">
                    {(data.top_concerning_themes as Array<{ cameo_code: string; mention_count: number }>)
                      .map((t) => (
                        <span
                          key={t.cameo_code}
                          className="text-xs px-2 py-0.5 rounded-full bg-red-950/50 text-red-300 border border-red-800/40"
                        >
                          {t.cameo_code} · {t.mention_count.toLocaleString()} mentions
                        </span>
                      ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}