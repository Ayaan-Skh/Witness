// src/app/briefs/[id]/page.tsx
import { getBrief } from "@/lib/api"
import { notFound } from "next/navigation"
import Link from "next/link"
import { format } from "date-fns"
import {
  ArrowLeft, Satellite, Newspaper, FileText,
  Brain, Clock, ShieldAlert, Users,
} from "lucide-react"
import type { AnomalySource } from "@/types"
import { TIER_COLOR } from "@/types"
import ConfidenceBadge from "@/app/components/ConfidenceBadge"
import StreamEvidence from "@/app/components/StreamEvidence"

// Force dynamic so we always get fresh data for the detail page
export const dynamic = "force-dynamic"

const SOURCE_ICON: Record<AnomalySource, React.ReactNode> = {
  SATELLITE:   <Satellite size={13} />,
  GDELT:       <Newspaper size={13} />,
  PROCUREMENT: <FileText  size={13} />,
}

export default async function BriefPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const brief = await getBrief(id).catch(() => null)
  if (!brief) notFound()

  const tierColor   = TIER_COLOR[brief.confidence_tier]
  const regionLabel = brief.region_id.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">

      {/* ── Sticky header ─────────────────────────────────────────── */}
      <header className="sticky top-0 z-20 bg-gray-900/95 backdrop-blur border-b border-gray-800 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-4">
          <Link
            href="/"
            className="w-8 h-8 rounded-lg bg-gray-800 hover:bg-gray-700 flex items-center justify-center transition-colors text-gray-400 hover:text-gray-100"
          >
            <ArrowLeft size={16} />
          </Link>

          <div className="flex-1 min-w-0">
            <h1 className="font-semibold text-base text-gray-100 truncate">{regionLabel}</h1>
            <p className="text-xs text-gray-500">
              {format(new Date(brief.time_window_start), "d MMM yyyy")}
              {" → "}
              {format(new Date(brief.time_window_end), "d MMM yyyy")}
              {" · "}
              {brief.contributing_streams.length} source{brief.contributing_streams.length !== 1 ? "s" : ""}
            </p>
          </div>
{/* @ts-ignore */}
          <ConfidenceBadge tier={brief.confidence_tier} score={brief.confidence_score} />

          <span
            className="text-xs px-2.5 py-1 rounded-full border"
            style={{
              color:        brief.status === "PUBLISHED" ? "#34D399" : "#9ca3af",
              borderColor:  brief.status === "PUBLISHED" ? "#34D39940" : "#374151",
              background:   brief.status === "PUBLISHED" ? "#34D39912" : "transparent",
            }}
          >
            {brief.status}
          </span>
        </div>
      </header>

      {/* ── Content ───────────────────────────────────────────────── */}
      <div className="max-w-4xl mx-auto px-6 py-8 space-y-8">

        {/* Contributing streams */}
        <section>
          <SectionTitle icon={<ShieldAlert size={13} />} label="Data Sources" />
          <div className="flex gap-2 flex-wrap mt-3">
            {brief.contributing_streams.map((stream) => (
              <span
                key={stream}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-gray-800 text-gray-200 border border-gray-700"
              >
                {SOURCE_ICON[stream as AnomalySource]}
                {stream}
              </span>
            ))}
          </div>
        </section>

        {/* Evidence by stream */}
        <section>
          <SectionTitle icon={<Satellite size={13} />} label="Evidence by Stream" />
          <div className="mt-3">
            <StreamEvidence
              evidence={brief.evidence}
              streams={brief.contributing_streams as AnomalySource[]}
            />
          </div>
        </section>

        {/* Agent reasoning */}
        <section>
          <SectionTitle icon={<Brain size={13} />} label="Agent Reasoning" />
          <div className="mt-3 bg-gray-900 rounded-xl border border-gray-800 p-5">
            {brief.agent_reasoning ? (
              <pre className="text-sm text-gray-300 whitespace-pre-wrap font-sans leading-relaxed">
                {brief.agent_reasoning}
              </pre>
            ) : (
              <p className="text-sm text-gray-600">No reasoning recorded.</p>
            )}
          </div>
        </section>

        {/* Historical context */}
        {brief.historical_context && (
          <section>
            <SectionTitle icon={<Clock size={13} />} label="Historical Context" />
            <div className="mt-3 bg-gray-900 rounded-xl border border-gray-800 p-5">
              <p className="text-sm text-gray-300 leading-relaxed">
                {brief.historical_context}
              </p>
            </div>
          </section>
        )}

        {/* Reviewer notes */}
        {brief.reviewer_notes && (
          <section>
            <SectionTitle icon={<Users size={13} />} label="Reviewer Notes" />
            <div className="mt-3 rounded-xl border border-amber-800/30 bg-amber-950/20 p-5">
              <p className="text-sm text-amber-200 leading-relaxed">
                {brief.reviewer_notes}
              </p>
            </div>
          </section>
        )}

        {/* Ethical footer */}
        <div
          className="rounded-xl border p-4 text-xs leading-relaxed"
          style={{ borderColor: `${tierColor}22`, background: `${tierColor}08`, color: "#6b7280" }}
        >
          <span className="font-medium" style={{ color: tierColor }}>Important: </span>
          This brief was generated automatically from public data sources and presents
          correlational evidence only — not verified facts. All briefs require human review
          before any action is taken. Witness cannot confirm the occurrence of human rights
          violations. Confidence tier <strong style={{ color: tierColor }}>{brief.confidence_tier}</strong>{" "}
          reflects statistical convergence across independent signals, not ground truth.
        </div>

      </div>
    </div>
  )
}

// ── Small section heading helper ──────────────────────────────────────────────
function SectionTitle({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-gray-500 font-medium">
      {icon}
      {label}
    </div>
  )
}