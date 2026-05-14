"use client";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import pageStyles from "@/components/dashboard-pages/section.module.css";
import { fetchJson } from "@/components/dashboard-pages/api";

type FailureFactor = {
  feature?: string;
  feature_value?: number | string;
  shap_value?: number;
  effect?: string;
};

type AlarmEvent = {
  id: number;
  device_id?: string;
  machine_id?: number;
  risk_level?: string;
  risk_score?: number;
  event_time?: string;
  created_at?: string;
  top_factors?: FailureFactor[];
  alert_type?: string;
  message?: string;
  severity?: string;
};

type AlarmDetail = {
  id: number;
  device_id?: string;
  machine_id?: number;
  alert_type?: string;
  severity?: string;
  message?: string;
  status?: string;
  created_at?: string;
  prediction_id?: number;
  risk_score?: number;
  risk_level?: string;
  model_type?: string;
  features_used?: Record<string, unknown> | null;
  top_factors?: FailureFactor[] | null;
};

export default function AlertsPageContent() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  const searchParams = useSearchParams();
  const alertId = searchParams.get("alertId");

  const [events, setEvents] = useState<AlarmEvent[]>([]);
  const [selectedAlarm, setSelectedAlarm] = useState<AlarmDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!apiUrl) {
      setError("NEXT_PUBLIC_API_URL is not configured");
      setLoading(false);
      return;
    }

    const load = async () => {
      try {
        const data = await fetchJson<AlarmEvent[]>(`${apiUrl}/api/v1/alerts/open`);
        setEvents(Array.isArray(data) ? data : []);
        setError("");
      } catch (err) {
        const errorObj = err as Error;
        setError(errorObj.message || "Failed to load alarm events");
      } finally {
        setLoading(false);
      }
    };

    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  useEffect(() => {
    if (!apiUrl || !alertId) {
      setSelectedAlarm(null);
      return;
    }

    const loadDetail = async () => {
      try {
        setDetailLoading(true);

        const data = await fetchJson<AlarmDetail>(
          `${apiUrl}/api/v1/alerts/${alertId}`
        );

        setSelectedAlarm(data);
      } catch (err) {
        console.error("Failed to load alarm detail", err);
      } finally {
        setDetailLoading(false);
      }
    };

    loadDetail();

    const interval = setInterval(loadDetail, 10000);

    return () => clearInterval(interval);
  }, [apiUrl, alertId]);

  const alarms = useMemo(() => {
    return [...events]
      .sort((a, b) => {
        const aTime = new Date(a.event_time || a.created_at || 0).getTime();
        const bTime = new Date(b.event_time || b.created_at || 0).getTime();
        return bTime - aTime;
      })
      .filter((item) => {
        const level = item.risk_level?.toUpperCase();
        return level === "CRITICAL" || level === "WARNING";
      })
      .slice(0, 30);
  }, [events]);

  const stats = useMemo(() => {
    return {
      total: alarms.length,
      critical: alarms.filter((item) => item.risk_level?.toUpperCase() === "CRITICAL").length,
      warning: alarms.filter((item) => item.risk_level?.toUpperCase() === "WARNING").length,
    };
  }, [alarms]);

  const alertChartData = useMemo(() => {
    const buckets: Record<
      string,
      {
        time: string;
        critical: number;
        warning: number;
        cloud: number;
        edge: number;
      }
    > = {};

    alarms.forEach((alarm) => {
      const rawDate = alarm.created_at || alarm.event_time;
      if (!rawDate) return;

      const date = new Date(rawDate);
      const key = date.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      });

      if (!buckets[key]) {
        buckets[key] = {
          time: key,
          critical: 0,
          warning: 0,
          cloud: 0,
          edge: 0,
        };
      }

      const level = alarm.risk_level?.toUpperCase();

      if (level === "CRITICAL") buckets[key].critical += 1;
      if (level === "WARNING") buckets[key].warning += 1;

      if (alarm.alert_type === "cloud_shap_risk") {
        buckets[key].cloud += 1;
      } else {
        buckets[key].edge += 1;
      }
    });

    return Object.values(buckets).slice(-10);
  }, [alarms]);

  const alertTimeline = useMemo(() => {
    return alarms.slice(0, 8);
  }, [alarms]);

  const factors = selectedAlarm?.top_factors || [];
  const increasingFactors = factors.filter((item) => item.effect === "increase").slice(0, 2);

  return (
    <main className={pageStyles.page}>
      <div className={pageStyles.header}>
        <p className={pageStyles.kicker}>Plant Safety</p>
        <h1 className={pageStyles.title}>Alarm Center</h1>
        <p className={pageStyles.subtitle}>
          Active warning and critical machine states with operator actions and failure cause explanation.
        </p>
      </div>

      {selectedAlarm && (
        <div className={pageStyles.card}>
          <div className={pageStyles.cardTop}>
            <span
              className={`${pageStyles.badge} ${
                selectedAlarm.risk_level?.toUpperCase() === "CRITICAL" ||
                selectedAlarm.severity?.toUpperCase() === "CRITICAL"
                  ? pageStyles.badgeCritical
                  : pageStyles.badgeWarning
              }`}
            >
              {selectedAlarm.risk_level || selectedAlarm.severity || "ALARM"}
            </span>

            <span className={pageStyles.mutedText}>
              Status: {selectedAlarm.status || "open"}
            </span>
          </div>

          <h3 className={pageStyles.cardTitle}>
            Machine {selectedAlarm.machine_id} Alarm Detail
          </h3>

          <p className={pageStyles.cardText}>
            {buildEngineerExplanation(selectedAlarm)}
          </p>
          <div className={pageStyles.grid}>
            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Risk Level</p>
              <h3 className={pageStyles.cardTitle}>
                {selectedAlarm.risk_level || selectedAlarm.severity || "—"}
              </h3>
              <p className={pageStyles.cardText}>Current alarm classification</p>
            </div>

            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Risk Score</p>
              <h3 className={pageStyles.cardTitle}>
                {typeof selectedAlarm.risk_score === "number"
                  ? (selectedAlarm.risk_score * 100).toFixed(1) + "%"
                  : "—"}
              </h3>
              <p className={pageStyles.cardText}>Estimated machine failure risk</p>
            </div>
          </div>

          {detailLoading ? (
            <div className={pageStyles.stateBox}>Loading alarm explanation...</div>
          ) : factors.length > 0 ? (
            <section style={{ marginTop: 28, display: "grid", gap: 22 }}>
              <div
                style={{
                  padding: 24,
                  borderRadius: 24,
                  background: "linear-gradient(135deg, #111827, #07111f)",
                  border: "1px solid rgba(56, 189, 248, 0.25)",
                }}
              >
                <h2 style={{ marginBottom: 12, color: "#f8fafc" }}>Engineer Summary</h2>

                <p style={{ color: "#cbd5e1", fontSize: 18 }}>
                  The alarm is mainly related to the following operating parameters:
                </p>

                <ul style={{ marginTop: 14, color: "#fda4af", fontSize: 18 }}>
                  {factors.slice(0, 3).map((factor, index) => (
                    <li key={index}>
                      {formatFeatureName(factor.feature)} = {factor.feature_value}{" "}
                      {factor.effect === "increase"
                        ? "increased failure risk"
                        : "reduced failure risk"}{" "}
                      (SHAP:{" "}
                      {typeof factor.shap_value === "number"
                        ? factor.shap_value.toFixed(3)
                        : "—"}
                      )
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
                <h2 style={{ marginBottom: 12, color: "#f8fafc" }}>Recommended Operator Actions</h2>

                <ul style={{ color: "#dbeafe", fontSize: 17, lineHeight: 1.8 }}>
                  {hasFactor(factors, "Tool_wear_min") && (
                    <li>Inspect tool condition and consider replacement.</li>
                  )}

                  {hasFactor(factors, "Torque_Nm") && (
                    <li>Check torque load and inspect drivetrain or mechanical resistance.</li>
                  )}

                  {hasFactor(factors, "Rotational_speed_rpm") && (
                    <li>Verify rotational speed stability and operating range.</li>
                  )}

                  {hasFactor(factors, "Air_temperature_K") && (
                    <li>Check ambient temperature and cooling conditions.</li>
                  )}

                  <li>Review the latest sensor readings before stopping the machine.</li>
                </ul>
              </div>

              <div>
                <h2 style={{ marginBottom: 16, color: "#f8fafc" }}>Technical Failure Factors</h2>

                <div style={{ display: "grid", gap: 14 }}>
                  {factors.map((factor, index) => {
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
                          <b>{formatFeatureName(factor.feature)}</b>
                          <span style={{ color: isIncrease ? "#fb7185" : "#60a5fa" }}>
                            {isIncrease ? "+" : ""}
                            {typeof factor.shap_value === "number"
                              ? factor.shap_value.toFixed(3)
                              : "—"}
                          </span>
                        </div>

                        <p style={{ color: "#9fb4d8" }}>
                          Current value: {factor.feature_value} ·{" "}
                          {isIncrease
                            ? "increases failure risk"
                            : "reduces failure risk"}
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
            <div className={pageStyles.stateBox}>
              No failure cause explanation is available for this alarm yet.
            </div>
          )}
        </div>
      )}

      {!selectedAlarm && (
        <>
          <div className={pageStyles.grid}>
            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Active Alarms</p>
              <h3 className={pageStyles.cardTitle}>{stats.total}</h3>
              <p className={pageStyles.cardText}>Warning and critical machine states</p>
            </div>

            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Critical</p>
              <h3 className={pageStyles.cardTitle}>{stats.critical}</h3>
              <p className={pageStyles.cardText}>Immediate inspection required</p>
            </div>

            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Warning</p>
              <h3 className={pageStyles.cardTitle}>{stats.warning}</h3>
              <p className={pageStyles.cardText}>Requires operator attention</p>
            </div>
          </div>

          <div className={pageStyles.card} style={{ marginTop: 24 }}>
            <h2 className={pageStyles.cardTitle}>Real-Time Alert Activity</h2>
            <p className={pageStyles.cardText}>
              Real-time machine risk activity and predictive maintenance alarm trends across industrial equipment.
            </p>

            <div style={{ width: "100%", height: 320, marginTop: 20 }}>
              <ResponsiveContainer>
                <LineChart data={alertChartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="time" />
                  <YAxis allowDecimals={false} />
                  <Tooltip />

                  <Line
                    type="monotone"
                    dataKey="critical"
                    name="Critical"
                    strokeWidth={3}
                    dot
                  />

                  <Line
                    type="monotone"
                    dataKey="warning"
                    name="Warning"
                    strokeWidth={3}
                    dot
                  />

                  <Line
                    type="monotone"
                    name="AI Failure Analysis"
                    strokeWidth={3}
                    dot
                  />

                  <Line
                    type="monotone"
                    dataKey="edge"
                    name="Real-Time Machine Detection"
                    strokeWidth={3}
                    dot
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className={pageStyles.card} style={{ marginTop: 24 }}>
            <h2 className={pageStyles.cardTitle}>Alert Timeline</h2>
            <p className={pageStyles.cardText}>
              Chronological history of recent machine alerts.
            </p>

            <div style={{ display: "grid", gap: 14, marginTop: 18 }}>
              {alertTimeline.map((alarm) => (
                <div
                  key={alarm.id}
                  style={{
                    display: "grid",
                    gridTemplateColumns: "90px 1fr",
                    gap: 14,
                    alignItems: "start",
                    padding: 14,
                    borderRadius: 16,
                    background: "#070d1f",
                    border: "1px solid rgba(148, 163, 184, 0.18)",
                  }}
                >
                  <strong style={{ color: "#93c5fd" }}>
                    {formatTime(alarm.created_at || alarm.event_time)}
                  </strong>

                  <div>
                    <b style={{ color: "#f8fafc" }}>
                      {alarm.alert_type === "cloud_shap_risk"
                        ? "Cloud SHAP critical"
                        : "Edge alert"}
                    </b>

                    <p style={{ marginTop: 4, color: "#cbd5e1" }}>
                      Machine {alarm.machine_id} ·{" "}
                      {alarm.risk_level || alarm.severity || "ALERT"} · Risk{" "}
                      {typeof alarm.risk_score === "number"
                        ? alarm.risk_score.toFixed(3)
                        : "—"}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {loading ? (
            <div className={pageStyles.stateBox}>Loading alarm center...</div>
          ) : error ? (
            <div className={pageStyles.errorBox}>{error}</div>
          ) : alarms.length === 0 ? (
            <div className={pageStyles.stateBox}>No active alarms found.</div>
          ) : (
            <div className={pageStyles.grid}>
              {alarms.map((alarm) => {
                const level = alarm.risk_level?.toUpperCase() || "WARNING";

                return (
                  <div
                    key={alarm.id}
                    className={pageStyles.card}
                    onClick={() => {
                      window.location.href = `/dashboard/alerts?alertId=${alarm.id}`;
                    }}
                    style={{ cursor: "pointer" }}
                  >
                    <div className={pageStyles.cardTop}>
                      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                        <span
                          className={`${pageStyles.badge} ${
                            level === "CRITICAL"
                              ? pageStyles.badgeCritical
                              : pageStyles.badgeWarning
                          }`}
                        >
                          {level}
                        </span>

                        <span className={pageStyles.mutedText}>
                          {alarm.alert_type === "cloud_shap_risk"
                            ? "Cloud SHAP"
                            : "Edge Alert"}
                        </span>
                      </div>

                      <span className={pageStyles.mutedText}>
                        Risk:{" "}
                        {typeof alarm.risk_score === "number"
                          ? alarm.risk_score.toFixed(3)
                          : "—"}
                      </span>
                    </div>

                    <h3 className={pageStyles.cardTitle}>
                      {alarm.device_id || `Machine ${alarm.machine_id ?? "—"}`}
                    </h3>

                    <p className={pageStyles.cardText}>
                      {alarm.message || buildAlarmExplanation(alarm.top_factors || [])}
                    </p>

                    <p className={pageStyles.cardFooter}>
                      {formatDate(alarm.event_time || alarm.created_at)}
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

function buildAlarmExplanation(factors?: FailureFactor[]) {
  if (!factors || factors.length === 0) {
    return "Abnormal machine behavior detected.";
  }

  const increased = factors.filter((factor) => factor.effect === "increase");

  if (increased.length === 0) {
    return "The monitoring system detected abnormal machine behavior. Current operating parameters remain close to safe ranges, however additional inspection is recommended to prevent potential equipment failure.";
  }

  const reasons = increased.slice(0, 2).map((factor) => {
    const name = formatFeatureName(factor.feature);
    const value = factor.feature_value ?? "—";

    if (factor.feature === "Tool_wear_min") {
      return `${name} reached ${value} min`;
    }

    if (factor.feature === "Torque_Nm") {
      return `${name} increased to ${value} Nm`;
    }

    if (factor.feature === "Rotational_speed_rpm") {
      return `${name} changed to ${value} rpm`;
    }

    if (factor.feature === "temp_diff") {
      return `${name} reached ${value} K`;
    }

    if (factor.feature === "power_kw") {
      return `${name} increased to ${Number(value).toFixed(2)} kW`;
    }

    return `${name} changed to ${value}`;
  });

  return `Failure risk increased because ${reasons.join(" and ")}.`;
}

function hasFactor(factors: FailureFactor[], featureName: string) {
  return factors.some((factor) => factor.feature === featureName);
}

function formatFeatureName(name?: string) {
  if (!name) return "Unknown parameter";

  const dictionary: Record<string, string> = {
    Tool_wear_min: "Tool Wear",
    Torque_Nm: "Torque Load",
    Rotational_speed_rpm: "Rotation Speed",
    Air_temperature_K: "Air Temperature",
    temp_diff: "Temperature Difference",
    power_kw: "Power Consumption",
  };

  return dictionary[name] || name.replaceAll("_", " ");
}

function formatDate(value?: string) {
  if (!value) return "—";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleString();
}

function formatTime(value?: string) {
  if (!value) return "—";

  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;

  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

function buildEngineerExplanation(alarm?: AlarmDetail | null) {
  if (!alarm) {
    return "Machine operating condition requires additional inspection.";
  }

  const factors = alarm.top_factors || [];
  const increased = factors.filter((factor) => factor.effect === "increase");

  if (increased.length === 0) {
    return "High-risk machine behavior was detected from the overall operating pattern. Current parameters remain close to normal ranges, however preventive inspection is recommended.";
  }

  const reasons = increased.slice(0, 3).map((factor) => {
    const name = formatFeatureName(factor.feature);
    const value = formatFactorValue(factor);

    if (factor.feature === "Tool_wear_min") {
      return `${name} reached ${value}, indicating possible tool degradation`;
    }

    if (factor.feature === "Torque_Nm") {
      return `${name} increased to ${value}, indicating possible mechanical overload`;
    }

    if (factor.feature === "Rotational_speed_rpm") {
      return `${name} changed to ${value}, indicating unstable rotational behavior`;
    }

    if (factor.feature === "temp_diff") {
      return `${name} reached ${value}, indicating thermal imbalance`;
    }

    if (factor.feature === "power_kw") {
      return `${name} increased to ${value}, indicating elevated power load`;
    }

    return `${name} reached ${value}`;
  });

  return `The monitoring system recommends inspection because ${reasons.join(
    "; "
  )}.`;
}

function formatFactorValue(factor: FailureFactor) {
  const value = factor.feature_value;

  if (value === undefined || value === null) {
    return "—";
  }

  if (factor.feature === "Torque_Nm") {
    return `${Number(value).toFixed(1)} Nm`;
  }

  if (factor.feature === "Rotational_speed_rpm") {
    return `${Number(value).toFixed(0)} rpm`;
  }

  if (factor.feature === "Tool_wear_min") {
    return `${Number(value).toFixed(0)} min`;
  }

  if (factor.feature === "temp_diff") {
    return `${Number(value).toFixed(1)} K`;
  }

  if (factor.feature === "power_kw") {
    return `${Number(value).toFixed(2)} kW`;
  }

  return String(value);
}