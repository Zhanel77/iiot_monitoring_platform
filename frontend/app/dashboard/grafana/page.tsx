"use client";

import styles from "../dashboard.module.css";

export default function GrafanaPage() {
  return (
    <div className={styles.page}>
      <div style={{ marginBottom: "16px" }}>
        <h1 style={{ fontSize: "28px", fontWeight: 700 }}>
          Grafana Monitoring
        </h1>
        <p style={{ color: "#9ca3af", marginTop: "6px" }}>
          Real-time industrial telemetry dashboards from Grafana.
        </p>
      </div>

      <div
        style={{
          width: "100%",
          height: "calc(100vh - 160px)",
          borderRadius: "16px",
          overflow: "hidden",
          border: "1px solid rgba(255,255,255,0.1)",
          background: "#111827",
        }}
      >
        <iframe
          src="http://localhost:3000"
          width="100%"
          height="100%"
          style={{ border: "none" }}
          title="Grafana Dashboard"
        />
      </div>
    </div>
  );
}
