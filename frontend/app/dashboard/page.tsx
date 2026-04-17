export default function DashboardPage() {
  return (
    <main className="min-h-screen bg-slate-950 text-white p-8">
      <p className="text-sm uppercase tracking-[0.3em] text-cyan-400">
        IIoT Monitoring System
      </p>
      <h1 className="mt-3 text-4xl font-bold">Dashboard</h1>
      <p className="mt-3 text-slate-400">
        Login successful. Here we can later add telemetry, alerts, devices, and predictions.
      </p>
    </main>
  );
}