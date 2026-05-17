import Link from "next/link"

export default function BriefNotFound() {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 bg-gray-950 p-8 text-gray-300">
      <h1 className="text-lg font-semibold text-gray-100">Brief not found</h1>
      <p className="text-sm text-gray-500">This investigation brief does not exist or was removed.</p>
      <Link href="/" className="text-sm text-amber-400 hover:underline">
        ← Back to dashboard
      </Link>
    </div>
  )
}
