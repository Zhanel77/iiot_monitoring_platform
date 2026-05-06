"use client";

import { useEffect, useMemo, useState } from "react";
import pageStyles from "@/components/dashboard-pages/section.module.css";
import { fetchJson } from "@/components/dashboard-pages/api";

type Device = {
  id: number;
  device_id?: string;
  machine_id?: number;
  name?: string;
  status?: string;
};

type Prediction = {
  id: number;
  device_id?: string;
  machine_id?: number;
  risk_score?: number;
  risk_level?: string;
  prediction_label?: string;
  event_time?: string;
  created_at?: string;
  features_used?: Record<string, unknown>;
};

export default function DevicesPageContent() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

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
        const [devicesData, predictionsData] = await Promise.all([
          fetchJson<Device[]>(`${apiUrl}/api/v1/devices`),
          fetchJson<Prediction[]>(`${apiUrl}/api/v1/predictions`),
        ]);

        setDevices(Array.isArray(devicesData) ? devicesData : []);
        setPredictions(Array.isArray(predictionsData) ? predictionsData : []);
        setError("");
      } catch (err) {
        const errorObj = err as Error;
        setError(errorObj.message || "Failed to load machines");
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

  const machineRows = useMemo(() => {
    return devices.map((machine) => {
      const latest = sortedPredictions.find(
        (p) => p.machine_id === machine.machine_id || p.device_id === machine.device_id
      );

      const level = (
        latest?.risk_level ||
        latest?.prediction_label ||
        machine.status ||
        "NORMAL"
      ).toUpperCase();

      const action = getRecommendedAction(level, latest?.features_used);

      return {
        ...machine,
        latest,
        level,
        action,
      };
    });
  }, [devices, sortedPredictions]);

  const stats = useMemo(() => {
    const critical = machineRows.filter((m) => m.level === "CRITICAL").length;
    const warning = machineRows.filter((m) => m.level === "WARNING").length;
    const normal = machineRows.filter((m) => m.level === "NORMAL" || m.level === "REGISTERED").length;

    return {
      total: machineRows.length,
      normal,
      warning,
      critical,
    };
  }, [machineRows]);

  return (
    <main className={pageStyles.page}>
      <div className={pageStyles.header}>
        <p className={pageStyles.kicker}>Plant Monitoring</p>
        <h1 className={pageStyles.title}>Machines</h1>
        <p className={pageStyles.subtitle}>
          Operational status of monitored industrial machines, latest telemetry state, and recommended operator actions.
        </p>
      </div>

      {loading ? (
        <div className={pageStyles.stateBox}>Loading machines...</div>
      ) : error ? (
        <div className={pageStyles.errorBox}>{error}</div>
      ) : machineRows.length === 0 ? (
        <div className={pageStyles.stateBox}>No machines found.</div>
      ) : (
        <>
          <div className={pageStyles.machineGrid}>
            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Machines Online</p>
              <h3 className={pageStyles.cardTitle}>{stats.total}</h3>
              <p className={pageStyles.cardText}>Registered and monitored assets</p>
            </div>

            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Normal</p>
              <h3 className={pageStyles.cardTitle}>{stats.normal}</h3>
              <p className={pageStyles.cardText}>Operating inside expected range</p>
            </div>

            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Warning</p>
              <h3 className={pageStyles.cardTitle}>{stats.warning}</h3>
              <p className={pageStyles.cardText}>Requires operator attention</p>
            </div>

            <div className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Critical</p>
              <h3 className={pageStyles.cardTitle}>{stats.critical}</h3>
              <p className={pageStyles.cardText}>Immediate inspection recommended</p>
            </div>
          </div>

          <div className={pageStyles.machineGrid}>
            {machineRows.map((machine) => (
              <div key={machine.id} className={pageStyles.machineCard}>
                <div className={pageStyles.machineHeader}>
                  <h3 className={pageStyles.machineName}>
                    {machine.name || `Machine ${machine.machine_id ?? machine.id}`}
                  </h3>
                    <span
                      className={`${pageStyles.machineStatus} ${
                        machine.level === "CRITICAL"
                          ? pageStyles.machineStatusCritical
                          : machine.level === "WARNING"
                          ? pageStyles.machineStatusWarning
                          : pageStyles.machineStatusNormal
                      }`}
                    >
                      {machine.level}
                    </span>

                  <span className={pageStyles.mutedText}>
                    Risk:{" "}
                    {typeof machine.latest?.risk_score === "number"
                      ? machine.latest.risk_score.toFixed(3)
                      : "—"}
                  </span>
                </div>

                <h3 className={pageStyles.cardTitle}>
                  {machine.name || `Machine ${machine.machine_id ?? machine.id}`}
                </h3>

                <p className={pageStyles.cardText}>
                  Asset ID: {machine.device_id || "—"}
                </p>

                <div className={pageStyles.machineTelemetry}>
                  <div className={pageStyles.telemetryCard}>
                    <span className={pageStyles.telemetryLabel}>Risk Score</span>
                    <strong className={pageStyles.telemetryValue}>
                      {typeof machine.latest?.risk_score === "number"
                        ? machine.latest.risk_score.toFixed(3)
                        : "—"}
                    </strong>
                  </div>

                  <div className={pageStyles.telemetryCard}>
                    <span className={pageStyles.telemetryLabel}>Machine ID</span>
                    <strong className={pageStyles.telemetryValue}>
                      {machine.machine_id ?? "—"}
                    </strong>
                  </div>
                </div>

                <div className={pageStyles.machineAction}>
                  <strong>Recommended operator action:</strong>
                  <br />
                  {machine.action}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
    </main>
  );
}

function getRecommendedAction(level: string, features?: Record<string, unknown>) {
  const hasToolWear = hasFeature(features, "Tool_wear_min");
  const hasTorque = hasFeature(features, "Torque_Nm");

  if (level === "CRITICAL") {
    if (hasToolWear) return "Inspect tool condition and plan replacement.";
    if (hasTorque) return "Inspect drivetrain and reduce mechanical load.";
    return "Stop or inspect the machine before continuing production.";
  }

  if (level === "WARNING") {
    return "Monitor closely and review latest sensor readings.";
  }

  return "No immediate action required.";
}

function hasFeature(features: Record<string, unknown> | undefined, key: string) {
  return Boolean(features && Object.prototype.hasOwnProperty.call(features, key));
}

function formatTelemetry(features?: Record<string, unknown>) {
  if (!features) return "Telemetry: unavailable";

  const temp = features.Air_temperature_K;
  const rpm = features.Rotational_speed_rpm;
  const torque = features.Torque_Nm;

  return `Telemetry: Temp ${formatValue(temp)} K · RPM ${formatValue(rpm)} · Torque ${formatValue(torque)} Nm`;
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