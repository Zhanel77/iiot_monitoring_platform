"use client";

import { useEffect, useMemo, useState } from "react";
import pageStyles from "@/components/dashboard-pages/section.module.css";
import { fetchJson } from "@/components/dashboard-pages/api";
import { useSearchParams } from "next/navigation";

type ShapFactor = {
  feature?: string;
  shap_value?: number;
  effect?: string;
};

type Prediction = {
  id: number;
  device_id?: string;
  machine_id?: number;
  risk_level?: string;
  risk_score?: number;
  event_time?: string;
  created_at?: string;
  top_factors?: ShapFactor[];
};

export default function AlertsPageContent() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const searchParams = useSearchParams();
  const alertId = searchParams.get("alertId");
  const [selectedAlert, setSelectedAlert] = useState<any | null>(null);

  useEffect(() => {
    if (!apiUrl || !alertId) return;

    const loadAlertDetail = async () => {
      try {
        const data = await fetchJson<any>(`${apiUrl}/api/v1/alerts/${alertId}`);
        setSelectedAlert(data);
      } catch (err) {
        console.error("Failed to load alert detail", err);
      }
    };

    loadAlertDetail();
  }, [apiUrl, alertId]);

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
        setError(errorObj.message || "Failed to load alerts");
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [apiUrl]);

  const alerts = useMemo(() => {
    return [...predictions]
      .sort((a, b) => {
        const aTime = new Date(a.event_time || a.created_at || 0).getTime();
        const bTime = new Date(b.event_time || b.created_at || 0).getTime();
        return bTime - aTime;
      })
      .filter((item) => {
        const level = item.risk_level?.toUpperCase();
        return level === "CRITICAL" || level === "WARNING";
      })
      .slice(0, 10);
  }, [predictions]);

  return (
    <main className={pageStyles.page}>
      <div className={pageStyles.header}>
        <p className={pageStyles.kicker}>Layer</p>
        <h1 className={pageStyles.title}>Alerts</h1>
        <p className={pageStyles.subtitle}>
          Critical and warning events generated from cloud prediction outputs.
        </p>
      </div>

      {selectedAlert && (
        <div className={pageStyles.card}>
          <div className={pageStyles.cardTop}>
            <span className={`${pageStyles.badge} ${pageStyles.badgeCritical}`}>
              {selectedAlert.severity}
            </span>
            <span className={pageStyles.mutedText}>
              Model: {selectedAlert.model_type || "—"}
            </span>
          </div>

          <h3 className={pageStyles.cardTitle}>
            Alert Details — Machine {selectedAlert.machine_id}
          </h3>

          <p className={pageStyles.cardText}>{selectedAlert.message}</p>

          <p className={pageStyles.cardText}>
            Risk score:{" "}
            {typeof selectedAlert.risk_score === "number"
              ? (selectedAlert.risk_score * 100).toFixed(1) + "%"
              : "—"}
          </p>

          {Array.isArray(selectedAlert?.top_factors) &&
          selectedAlert.top_factors.length > 0 ? (
            <section style={{ marginTop: 28, display: "grid", gap: 22 }}>
              <div
                style={{
                  padding: 24,
                  borderRadius: 24,
                  background: "linear-gradient(135deg, #111827, #07111f)",
                  border: "1px solid rgba(56, 189, 248, 0.25)",
                }}
              >
                <h2 style={{ marginBottom: 12 }}>Engineer Summary</h2>

                <p style={{ color: "#cbd5e1", fontSize: 18 }}>
                  The system detected abnormal machine behavior. The main risk contributors are:
                </p>

                <ul style={{ marginTop: 14, color: "#fda4af", fontSize: 18 }}>
                  {selectedAlert.top_factors
                    .filter((f: any) => f.effect === "increase")
                    .slice(0, 2)
                    .map((factor: any, index: number) => (
                      <li key={index}>
                        {factor.feature.replaceAll("_", " ")} = {factor.feature_value}
                      </li>
                    ))}
                </ul>
              </div>

              <div
                style={{
                  padding: 24,
                  borderRadius: 24,
                  background: "#0b1220",
                  border: "1px solid rgba(148, 163, 184, 0.2)",
                }}
              >
                <h2 style={{ marginBottom: 12 }}>Recommended Actions</h2>

                <ul style={{ color: "#dbeafe", fontSize: 17, lineHeight: 1.8 }}>
                  {selectedAlert.top_factors.some((f: any) => f.feature === "Tool Wear") && (
                    <li>Inspect tool condition and consider replacement.</li>
                  )}

                  {selectedAlert.top_factors.some((f: any) => f.feature === "Torque Load") && (
                    <li>Check torque load and inspect drivetrain or mechanical resistance.</li>
                  )}

                  {selectedAlert.top_factors.some((f: any) => f.feature === "Rotation Speed") && (
                    <li>Verify rotational speed stability and operating range.</li>
                  )}

                  {selectedAlert.top_factors.some((f: any) => f.feature === "Air Temperature") && (
                    <li>Check surrounding temperature and cooling conditions.</li>
                  )}

                  <li>Review the latest sensor readings before stopping the machine.</li>
                </ul>
              </div>

              <div>
                <h2 style={{ marginBottom: 16 }}>Technical Details</h2>

                <div style={{ display: "grid", gap: 14 }}>
                  {selectedAlert.top_factors.map((factor: any, index: number) => {
                    const isIncrease = factor.effect === "increase";
                    const absValue = Math.abs(Number(factor.shap_value || 0));
                    const width = Math.min(100, absValue * 35);

                    return (
                      <div
                        key={index}
                        style={{
                          padding: 18,
                          borderRadius: 20,
                          background: "#070d1f",
                          border: "1px solid rgba(148, 163, 184, 0.18)",
                        }}
                      >
                        <div style={{ display: "flex", justifyContent: "space-between" }}>
                          <b>{factor.feature.replaceAll("_", " ")}</b>
                          <span style={{ color: isIncrease ? "#fb7185" : "#60a5fa" }}>
                            {isIncrease ? "+" : ""}
                            {Number(factor.shap_value).toFixed(3)}
                          </span>
                        </div>

                        <p style={{ color: "#9fb4d8" }}>
                          Value: {factor.feature_value} ·{" "}
                          {isIncrease ? "increases failure risk" : "reduces failure risk"}
                        </p>

                        <div style={{ height: 8, background: "#1f2937", borderRadius: 999 }}>
                          <div
                            style={{
                              width: `${width}%`,
                              height: "100%",
                              background: isIncrease ? "#fb7185" : "#60a5fa",
                              borderRadius: 999,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </section>
          ) : (
            <p>No SHAP explanation available</p>
          )}

          {Array.isArray(selectedAlert.top_factors) &&
            selectedAlert.top_factors.length > 0 && (
              <div className={pageStyles.cardText}>
                <b>SHAP explanation:</b>
                {selectedAlert.top_factors.map((factor: any, index: number) => (
                  <div key={index}>
                    {factor.feature}: {factor.feature_value} — {factor.effect} risk
                  </div>
                ))}
              </div>
            )}

          {selectedAlert.features_used && (
            <pre className={pageStyles.cardText}>
              {JSON.stringify(selectedAlert.features_used, null, 2)}
            </pre>
          )}
        </div>
      )}

      {!selectedAlert && (
      <>
      {loading ? (
        <div className={pageStyles.stateBox}>Loading alerts...</div>
      ) : error ? (
        <div className={pageStyles.errorBox}>{error}</div>
      ) : alerts.length === 0 ? (
        <div className={pageStyles.stateBox}>No active alerts found.</div>
      ) : (
        <div className={pageStyles.grid}>
          {alerts.map((alert) => {
            const topFactors = Array.isArray(alert.top_factors)
              ? alert.top_factors.slice(0, 2).map((item) => item.feature).filter(Boolean)
              : [];

            return (
              <div key={alert.id} className={pageStyles.card}>
                <div className={pageStyles.cardTop}>
                  <span
                    className={`${pageStyles.badge} ${
                      alert.risk_level?.toUpperCase() === "CRITICAL"
                        ? pageStyles.badgeCritical
                        : pageStyles.badgeWarning
                    }`}
                  >
                    {alert.risk_level}
                  </span>

                  <span className={pageStyles.mutedText}>
                    Risk:{" "}
                    {typeof alert.risk_score === "number"
                      ? alert.risk_score.toFixed(3)
                      : "—"}
                  </span>
                </div>

                <h3 className={pageStyles.cardTitle}>
                  {alert.device_id || `Machine ${alert.machine_id ?? "—"}`}
                </h3>

                <p className={pageStyles.cardText}>
                  Cloud layer detected abnormal behavior that may require operator attention.
                </p>

                {topFactors.length > 0 && (
                  <p className={pageStyles.cardText}>
                    SHAP insight: {topFactors.join(" and ")} are the top factors.
                  </p>
                )}

                <p className={pageStyles.cardFooter}>
                  {alert.event_time || alert.created_at || "—"}
                </p>
              </div>
            );
          })}
        </div>
      )}
      </>
      )}
    </main>
  );
}