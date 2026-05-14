"use client";

import { useEffect, useRef } from "react";
import { notifications } from "@mantine/notifications";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Prediction = {
  id: number;
  device_id?: string;
  machine_id?: number;
  risk_level?: string;
  risk_score?: number;
  model_type?: string;
  model_name?: string;
};

export function MachineShapNotifications() {
  const shownRef = useRef<Set<number>>(new Set());

  useEffect(() => {
    const saved = sessionStorage.getItem("shown-machine-alerts");
    if (saved) {
      shownRef.current = new Set(JSON.parse(saved));
    }

    const loadMachineAlerts = async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/predictions`, {
          cache: "no-store",
        });

        const predictions: Prediction[] = await res.json();

        if (!Array.isArray(predictions)) return;

        const alerts = predictions
          .filter((item) => {
            const level = item.risk_level?.toUpperCase();
            return level === "CRITICAL" || level === "WARNING";
          })
          .slice(0, 5);

        alerts.forEach((alert) => {
          if (shownRef.current.has(alert.id)) return;

          shownRef.current.add(alert.id);
          sessionStorage.setItem(
            "shown-machine-alerts",
            JSON.stringify([...shownRef.current])
          );

          const level = alert.risk_level?.toUpperCase();

          notifications.show({
            title:
              level === "CRITICAL"
                ? `🚨 Critical Machine ${alert.machine_id}`
                : `⚠ Warning Machine ${alert.machine_id}`,
            message: (
              <div
                onClick={() => {
                  window.location.href = "/dashboard/alerts";
                }}
                style={{ cursor: "pointer" }}
              >
                AI detected {level?.toLowerCase()} risk for{" "}
                {alert.device_id || `Machine ${alert.machine_id}`}. Risk score:{" "}
                {typeof alert.risk_score === "number"
                  ? alert.risk_score.toFixed(4)
                  : "—"}
              </div>
            ),
            color: level === "CRITICAL" ? "red" : "orange",
            autoClose: 10000,
            withBorder: true,
          });
        });
      } catch (error) {
        console.error("Failed to load machine SHAP alerts", error);
      }
    };

    loadMachineAlerts();
    const interval = setInterval(loadMachineAlerts, 15000);

    return () => clearInterval(interval);
  }, []);

  return null;
}