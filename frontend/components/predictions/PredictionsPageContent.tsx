"use client";

import { useEffect, useMemo, useState } from "react";
import pageStyles from "@/components/dashboard-pages/section.module.css";
import { fetchJson } from "@/components/dashboard-pages/api";

type Prediction = {
  id: number;
  device_id?: string;
  machine_id?: number;
  prediction?: number;
  prediction_label?: string;
  risk_score?: number;
  risk_level?: string;
  model_name?: string;
  created_at?: string;
  event_time?: string;
};

export default function PredictionsPageContent() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!apiUrl) {
      setError("NEXT_PUBLIC_API_URL is not configured");
      setLoading(false);
      return;
    }

    const load = async () => {
      try {
        const data = await fetchJson<Prediction[]>(`${apiUrl}/api/v1/predictions`);
        setPredictions(Array.isArray(data) ? data : []);
        setError("");
      } catch (err) {
        const errorObj = err as Error;
        setError(errorObj.message || "Failed to load predictions");
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [apiUrl]);

  const sortedPredictions = useMemo(() => {
    return [...predictions].sort((a, b) => {
      const aTime = new Date(a.event_time || a.created_at || 0).getTime();
      const bTime = new Date(b.event_time || b.created_at || 0).getTime();
      return bTime - aTime;
    });
  }, [predictions]);

  const getStatusClass = (value?: string) => {
    const label = value?.toUpperCase();
    if (label === "CRITICAL" || label === "FAILURE") return pageStyles.badgeCritical;
    if (label === "WARNING") return pageStyles.badgeWarning;
    return pageStyles.badgeNormal;
  };

  return (
    <main className={pageStyles.page}>
      <div className={pageStyles.header}>
        <p className={pageStyles.kicker}>Inference</p>
        <h1 className={pageStyles.title}>Predictions</h1>
        <p className={pageStyles.subtitle}>
          Latest prediction results from edge and cloud models.
        </p>
      </div>

      {loading ? (
        <div className={pageStyles.stateBox}>Loading predictions...</div>
      ) : error ? (
        <div className={pageStyles.errorBox}>{error}</div>
      ) : sortedPredictions.length === 0 ? (
        <div className={pageStyles.stateBox}>No predictions found.</div>
      ) : (
        <div className={pageStyles.tableWrap}>
          <table className={pageStyles.table}>
            <thead>
              <tr>
                <th>Device</th>
                <th>Machine</th>
                <th>Status</th>
                <th>Risk Score</th>
                <th>Model</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {sortedPredictions.map((item) => {
                const label =
                  item.risk_level ||
                  item.prediction_label ||
                  (item.prediction === 1 ? "FAILURE" : "NORMAL");

                return (
                  <tr key={item.id}>
                    <td>{item.device_id || "—"}</td>
                    <td>{item.machine_id ?? "—"}</td>
                    <td>
                      <span className={`${pageStyles.badge} ${getStatusClass(label)}`}>
                        {label}
                      </span>
                    </td>
                    <td>
                      {typeof item.risk_score === "number"
                        ? item.risk_score.toFixed(3)
                        : "—"}
                    </td>
                    <td>{item.model_name || "—"}</td>
                    <td>{item.event_time || item.created_at || "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}