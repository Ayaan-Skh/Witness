interface Props {
  message: string
}

export default function ApiBanner({ message }: Props) {
  return (
    <div className="bg-amber-950/90 border-b border-amber-800/50 px-4 py-2 text-xs text-amber-200 z-30">
      <strong className="font-medium">API unavailable — </strong>
      {message}{" "}
      <span className="text-amber-400/80">
        Start backend: <code className="text-amber-300">uvicorn api.main:app --reload --port 8000</code>
        {" · "}
        Seed data: <code className="text-amber-300">python scripts/seed_demo_data.py</code>
      </span>
    </div>
  )
}
