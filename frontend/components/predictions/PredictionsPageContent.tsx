"use client";

import { useEffect, useMemo, useState } from "react";
import pageStyles from "@/components/dashboard-pages/section.module.css";
import { fetchJson } from "@/components/dashboard-pages/api";

type TelemetryEvent = {
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
  features_used?: Record<string, unknown>;
};

export default function PredictionsPageContent() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

  const [events, setEvents] = useState<TelemetryEvent[]>([]);
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
        const data = await fetchJson<TelemetryEvent[]>(`${apiUrl}/api/v1/predictions`);
        setEvents(Array.isArray(data) ? data : []);
        setError("");
      } catch (err) {
        const errorObj = err as Error;
        setError(errorObj.message || "Failed to load telemetry events");
      } finally {
        setLoading(false);
      }
    };

    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  const sortedEvents = useMemo(() => {
    return [...events].sort((a, b) => {
      const aTime = new Date(a.event_time || a.created_at || 0).getTime();
      const bTime = new Date(b.event_time || b.created_at || 0).getTime();
      return bTime - aTime;
    });
  }, [events]);

  const stats = useMemo(() => {
    const warning = events.filter((item) => getStatus(item) === "WARNING").length;
    const critical = events.filter((item) => getStatus(item) === "CRITICAL").length;
    const normal = events.filter((item) => getStatus(item) === "NORMAL").length;

    return {
      total: events.length,
      normal,
      warning,
      critical,
    };
  }, [events]);

  const getStatusClass = (value?: string) => {
    const label = value?.toUpperCase();

    if (label === "CRITICAL" || label === "FAILURE") return pageStyles.badgeCritical;
    if (label === "WARNING") return pageStyles.badgeWarning;

    return pageStyles.badgeNormal;
  };

  return (
    <main className={pageStyles.page}>
      <div className={pageStyles.header}>
        <p className={pageStyles.kicker}>Live Operations</p>
        <h1 className={pageStyles.title}>Telemetry Events</h1>
        <p className={pageStyles.subtitle}>
          Recent machine events, operating status, risk level, and latest sensor readings from the monitoring pipeline.
        </p>
      </div>

      {loading ? (
        <div className={pageStyles.stateBox}>Loading telemetry events...</div>
      ) : error ? (
        <div className={pageStyles.errorBox}>{error}</div>
      ) : sortedEvents.length === 0 ? (
        <div className={pageStyles.stateBox}>No telemetry events found.</div>
      ) : (
        <>
          <div className={pageStyles.grid}>
            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Total Events</p>
              <h3 className={pageStyles.cardTitle}>{stats.total}</h3>
              <p className={pageStyles.cardText}>Processed monitoring records</p>
            </div>

            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Normal Events</p>
              <h3 className={pageStyles.cardTitle}>{stats.normal}</h3>
              <p className={pageStyles.cardText}>Machine state inside expected range</p>
            </div>

            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Warning Events</p>
              <h3 className={pageStyles.cardTitle}>{stats.warning}</h3>
              <p className={pageStyles.cardText}>Requires monitoring attention</p>
            </div>

            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Critical Events</p>
              <h3 className={pageStyles.cardTitle}>{stats.critical}</h3>
              <p className={pageStyles.cardText}>Requires immediate inspection</p>
            </div>
          </div>

          <div className={pageStyles.tableWrap}>
            <table className={pageStyles.table}>
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Machine</th>
                  <th>Operating State</th>
                  <th>Risk</th>
                  <th>Sensor Snapshot</th>
                  <th>Time</th>
                </tr>
              </thead>

              <tbody>
                {sortedEvents.map((item) => {
                  const label = getStatus(item);

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
                      <td>{formatSensorSnapshot(item.features_used)}</td>
                      <td>{formatDate(item.event_time || item.created_at)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </main>
  );
}

function getStatus(item: TelemetryEvent) {
  return (
    item.risk_level ||
    item.prediction_label ||
    (item.prediction === 1 ? "CRITICAL" : "NORMAL")
  ).toUpperCase();
}

function formatSensorSnapshot(features?: Record<string, unknown>) {
  if (!features) return "No sensor snapshot";

  const temp = formatValue(features.Air_temperature_K);
  const rpm = formatValue(features.Rotational_speed_rpm);
  const torque = formatValue(features.Torque_Nm);
  const wear = formatValue(features.Tool_wear_min);

  return `Temp ${temp}K · RPM ${rpm} · Torque ${torque}Nm · Wear ${wear}min`;
}

function formatValue(value: unknown) {
  if (typeof value === "number") return value.toFixed(1);
  if (typeof value === "string") return value;
  return "—";
}

function formatDate(value?: string) {
  if (!value) return "—";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString();
}