"use client";

import { useEffect, useRef } from "react";
import { notifications } from "@mantine/notifications";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function AlertNotifications() {
  const shown = useRef(new Set<number>());

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/alerts/open`, {
          cache: "no-store",
        });

        const alerts = await res.json();

        if (!Array.isArray(alerts) || alerts.length === 0) return;

        const newest = alerts[0];

        if (!shown.current.has(newest.id)) {
          shown.current.add(newest.id);

          notifications.show({
            title: `⚠ ${newest.severity?.toUpperCase() ?? "ALERT"}: Machine ${newest.machine_id}`,
            message: "Edge detected high risk. Click to view cloud SHAP explanation.",
            color: newest.severity === "critical" ? "red" : "orange",
            autoClose: 8000,
            withBorder: true,
            onClick: () => {
              window.location.href = `/dashboard/alerts?alertId=${newest.id}`;
            },
          });
        }
      } catch (error) {
        console.error("Failed to load alerts", error);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return null;
}