"use client";

import { useEffect, useMemo, useState } from "react";
import styles from "./dashboard.module.css";
import { fetchJson } from "@/components/dashboard-pages/api";

type UserInfo = {
  id?: number;
  full_name?: string;
  email?: string;
  role?: string;
};

type Device = {
  id: number;
  device_id?: string;
  machine_id?: number;
  name?: string;
  status?: string;
};

type ShapFactor = {
  feature?: string;
  feature_value?: number | string;
  shap_value?: number;
  effect?: string;
};

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
  top_factors?: ShapFactor[];
  features_used?: Record<string, unknown>;
};

type HealthResponse = {
  status: string;
};

export default function DashboardMain() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [user, setUser] = useState<UserInfo | null>(null);
  const [devices, setDevices] = useState<Device[]>([]);
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
        const [healthData, meData, devicesData, predictionsData] =
          await Promise.all([
            fetchJson<HealthResponse>(`${apiUrl}/health`),
            fetchJson<UserInfo>(`${apiUrl}/api/v1/auth/me`),
            fetchJson<Device[]>(`${apiUrl}/api/v1/devices`),
            fetchJson<Prediction[]>(`${apiUrl}/api/v1/predictions`),
          ]);

        setHealth(healthData);
        setUser(meData);
        setDevices(Array.isArray(devicesData) ? devicesData : []);
        setPredictions(Array.isArray(predictionsData) ? predictionsData : []);
        setError("");
      } catch (err) {
        const errorObj = err as Error;
        setError(errorObj.message || "Failed to load dashboard");
      } finally {
        setLoading(false);
      }
    };

    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  const sortedPredictions = useMemo(() => {
    return [...predictions].sort((a, b) => {
      const aTime = new Date(a.event_time || a.created_at || 0).getTime();
      const bTime = new Date(b.event_time || b.created_at || 0).getTime();
      return bTime - aTime;
    });
  }, [predictions]);

  const recentPredictions = useMemo(() => sortedPredictions.slice(0, 5), [sortedPredictions]);

  const stats = useMemo(() => {
    const activeAlerts = predictions.filter((item) => {
      const level = item.risk_level?.toUpperCase();
      return level === "WARNING" || level === "CRITICAL";
    }).length;

    const criticalCount = predictions.filter(
      (item) => item.risk_level?.toUpperCase() === "CRITICAL"
    ).length;

    const normalCount = predictions.filter((item) => {
      const label = (
        item.risk_level ||
        item.prediction_label ||
        (item.prediction === 1 ? "FAILURE" : "NORMAL")
      ).toUpperCase();

      return label === "NORMAL";
    }).length;

    return {
      totalDevices: devices.length,
      totalPredictions: predictions.length,
      activeAlerts,
      healthText: health?.status === "ok" ? "Stable" : "Unavailable",
      criticalCount,
      normalCount,
    };
  }, [devices, predictions, health]);

  const latestPredictionWithShap = useMemo(() => {
    return sortedPredictions.find(
      (item) => Array.isArray(item.top_factors) && item.top_factors.length > 0
    );
  }, [sortedPredictions]);

  const shapFactors = useMemo(() => {
    const factors = latestPredictionWithShap?.top_factors || [];

    return factors.slice(0, 5).map((item) => {
      const raw = typeof item.shap_value === "number" ? Math.abs(item.shap_value) : 0;
      const width = `${Math.max(12, Math.min(raw * 100, 100))}%`;

      return {
        feature: item.feature || "Unknown feature",
        impact: item.effect === "decrease" ? "decrease" : "increase",
        value:
          typeof item.shap_value === "number"
            ? item.shap_value > 0
              ? `+${item.shap_value.toFixed(3)}`
              : item.shap_value.toFixed(3)
            : "—",
        width,
      };
    });
  }, [latestPredictionWithShap]);

  const notifications = useMemo(() => {
    const result: { level: "CRITICAL" | "WARNING" | "SHAP"; text: string }[] = [];

    sortedPredictions
      .filter((item) => item.risk_level?.toUpperCase() === "CRITICAL")
      .slice(0, 2)
      .forEach((item) => {
        result.push({
          level: "CRITICAL",
          text: `High failure risk detected for ${item.device_id || `machine ${item.machine_id ?? "unknown"}`}`,
        });
      });

    sortedPredictions
      .filter((item) => item.risk_level?.toUpperCase() === "WARNING")
      .slice(0, 2)
      .forEach((item) => {
        result.push({
          level: "WARNING",
          text: `Warning threshold exceeded for ${item.device_id || `machine ${item.machine_id ?? "unknown"}`}`,
        });
      });

    if (shapFactors.length > 0) {
      result.push({
        level: "SHAP",
        text: `${shapFactors
          .slice(0, 2)
          .map((item) => item.feature)
          .join(" and ")} are the top factors in the latest explained prediction`,
      });
    }

    return result.slice(0, 4);
  }, [sortedPredictions, shapFactors]);

  const chartPoints = useMemo(() => {
    const chartData = [...sortedPredictions]
      .reverse()
      .slice(-8)
      .map((item) => ({
        time: item.event_time || item.created_at || "",
        risk:
          typeof item.risk_score === "number"
            ? Math.max(0, Math.min(item.risk_score, 1))
            : 0,
      }));

    if (chartData.length === 0) return [];

    return chartData.map((item, index) => {
      const x =
        chartData.length === 1 ? 20 : 20 + (index * 260) / (chartData.length - 1);
      const y = 120 - item.risk * 90;

      return {
        x,
        y,
        label: formatTime(item.time),
      };
    });
  }, [sortedPredictions]);

  const chartPath = useMemo(() => {
    if (chartPoints.length === 0) return "";
    return chartPoints
      .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
      .join(" ");
  }, [chartPoints]);

  const getPredictionLabel = (item: Prediction) => {
    return (
      item.risk_level ||
      item.prediction_label ||
      (item.prediction === 1 ? "FAILURE" : "NORMAL")
    );
  };

  if (loading) {
    return <main className={styles.page}><div className={styles.loading}>Loading dashboard...</div></main>;
  }

  return (
    <main className={styles.page}>
      <div className={styles.topHeader}>
        <div>
          <p className={styles.kicker}>IIoT Monitoring System</p>
          <h1 className={styles.title}>Dashboard</h1>
          <p className={styles.subtitle}>
            Real-time overview of industrial devices, cloud alerts, and explainable AI insights.
          </p>
        </div>

        <div className={styles.liveBadge}>
          <span className={styles.liveDot} />
          Auto-refresh every 15s
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.statsGrid}>
        <div className={styles.statCard}>
          <p className={styles.statLabel}>Connected Devices</p>
          <h2 className={styles.statValue}>{stats.totalDevices}</h2>
          <span className={styles.statHint}>Registered in the system</span>
        </div>

        <div className={styles.statCard}>
          <p className={styles.statLabel}>Predictions</p>
          <h2 className={styles.statValue}>{stats.totalPredictions}</h2>
          <span className={styles.statHint}>Loaded from API</span>
        </div>

        <div className={styles.statCard}>
          <p className={styles.statLabel}>Active Alerts</p>
          <h2 className={styles.statValueDanger}>{stats.activeAlerts}</h2>
          <span className={styles.statHint}>Warning and critical states</span>
        </div>

        <div className={styles.statCard}>
          <p className={styles.statLabel}>System Health</p>
          <h2 className={styles.statValueSuccess}>{stats.healthText}</h2>
          <span className={styles.statHint}>
            {user?.full_name ? `Operator: ${user.full_name}` : "Backend status"}
          </span>
        </div>
      </section>

      <section className={styles.topGrid}>
        <div className={styles.chartCard}>
          <div className={styles.cardHeader}>
            <div>
              <h3 className={styles.cardTitle}>Prediction Trend</h3>
              <p className={styles.cardSubtitle}>Recent risk score history</p>
            </div>
            <span className={styles.cardTag}>Live data</span>
          </div>

          <div className={styles.realChartBox}>
            {chartPoints.length === 0 ? (
              <p className={styles.emptyText}>No chart data available.</p>
            ) : (
              <>
                <svg viewBox="0 0 300 140" className={styles.chartSvg} preserveAspectRatio="none">
                  <path
                    d={chartPath}
                    fill="none"
                    stroke="url(#lineGradient)"
                    strokeWidth="3"
                    strokeLinecap="round"
                  />
                  {chartPoints.map((point, index) => (
                    <circle key={index} cx={point.x} cy={point.y} r="4" className={styles.chartCircle} />
                  ))}
                  <defs>
                    <linearGradient id="lineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#22d3ee" />
                      <stop offset="100%" stopColor="#60a5fa" />
                    </linearGradient>
                  </defs>
                </svg>

                <div className={styles.chartLabels}>
                  {chartPoints.map((point, index) => (
                    <span key={index}>{point.label}</span>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>

        <div className={styles.notificationsCard}>
          <div className={styles.cardHeader}>
            <div>
              <h3 className={styles.cardTitle}>Active Machine Alerts</h3>
              <p className={styles.cardSubtitle}>Generated from predictions and SHAP</p>
            </div>
          </div>

          <div className={styles.notificationList}>
            {notifications.length === 0 ? (
              <p className={styles.emptyText}>No notifications.</p>
            ) : (
              notifications.map((item, index) => (
                <div key={index} className={styles.notificationItem}>
                  <span
                    className={`${styles.notificationBadge} ${
                      item.level === "CRITICAL"
                        ? styles.notificationCritical
                        : item.level === "WARNING"
                        ? styles.notificationWarning
                        : styles.notificationInfo
                    }`}
                  >
                    {item.level}
                  </span>
                  <p className={styles.notificationText}>{item.text}</p>
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <section className={styles.bottomGrid}>
        <div className={styles.panel}>
          <div className={styles.cardHeader}>
            <div>
              <h3 className={styles.cardTitle}>Latest Machine Status</h3>
              <p className={styles.cardSubtitle}>Latest results from devices</p>
            </div>
          </div>

          <div className={styles.predictionList}>
            {recentPredictions.length === 0 ? (
              <p className={styles.emptyText}>No predictions available.</p>
            ) : (
              recentPredictions.map((item) => {
                const label = getPredictionLabel(item).toUpperCase();

                return (
                  <div key={item.id} className={styles.predictionItem}>
                    <div>
                      <p className={styles.predictionDevice}>
                        {item.device_id || `Machine ${item.machine_id ?? "—"}`}
                      </p>
                      <p className={styles.predictionMeta}>
                        {item.model_name || "Model unavailable"} · {formatDate(item.event_time || item.created_at)}
                      </p>
                    </div>

                    <div className={styles.predictionRight}>
                      <span
                        className={`${styles.statusBadge} ${
                          label === "CRITICAL" || label === "FAILURE"
                            ? styles.badgeCritical
                            : label === "WARNING"
                            ? styles.badgeWarning
                            : styles.badgeNormal
                        }`}
                      >
                        {label}
                      </span>
                      <span className={styles.scoreText}>
                        Risk: {typeof item.risk_score === "number" ? item.risk_score.toFixed(3) : "—"}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        <div className={styles.panel}>
          <div className={styles.cardHeader}>
            <div>
              <h3 className={styles.cardTitle}>Root Cause Analysis</h3>
              <p className={styles.cardSubtitle}>Top factors from latest explained prediction</p>
            </div>
          </div>

          <div className={styles.summaryMiniGrid}>
            <div className={styles.summaryMiniCard}>
              <span className={styles.summaryMiniLabel}>Normal</span>
              <strong className={styles.summaryMiniValueGreen}>{stats.normalCount}</strong>
            </div>
            <div className={styles.summaryMiniCard}>
              <span className={styles.summaryMiniLabel}>Critical</span>
              <strong className={styles.summaryMiniValueRed}>{stats.criticalCount}</strong>
            </div>
          </div>

          <div className={styles.shapList}>
            {shapFactors.length === 0 ? (
              <p className={styles.emptyText}>No factors found in prediction data.</p>
            ) : (
              shapFactors.map((item, index) => (
                <div key={index} className={styles.shapCard}>
                  <div className={styles.shapTop}>
                    <div>
                      <p className={styles.shapFeature}>{item.feature}</p>
                      <p className={styles.shapImpact}>Impact: {item.impact}</p>
                    </div>

                    <span
                      className={`${styles.shapValue} ${
                        item.impact === "increase" ? styles.shapIncrease : styles.shapDecrease
                      }`}
                    >
                      {item.value}
                    </span>
                  </div>

                  <div className={styles.shapBar}>
                    <div
                      className={`${styles.shapFill} ${
                        item.impact === "increase"
                          ? styles.shapFillIncrease
                          : styles.shapFillDecrease
                      }`}
                      style={{ width: item.width }}
                    />
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </section>
    </main>
  );
}

function formatTime(value?: string) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function formatDate(value?: string) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}