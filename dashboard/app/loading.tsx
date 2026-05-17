export default function Loading() {
  return (
    <div className="flex h-screen w-screen items-center justify-center bg-gray-950 text-gray-400">
      <div className="flex flex-col items-center gap-3">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-gray-600 border-t-amber-500" />
        <p className="text-sm">Loading…</p>
      </div>
    </div>
  )
}
