"use client";

import { useEffect, useRef } from "react";
import { notifications } from "@mantine/notifications";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export function AlertNotifications() {
  const shown = useRef(new Set<number>());

  useEffect(() => {
    const interval = setInterval(async () => {
      const res = await fetch(`${API_URL}/api/v1/alerts/open`, {
        cache: "no-store",
      });

      const alerts = await res.json();
      const cloudAlerts = alerts.filter((alert: any) => alert.model_type === "cloud");
      const newest = cloudAlerts[0];

      alerts.forEach((alert: any) => {
        if (newest && !shown.current.has(newest.id)) {
            shown.current.add(newest.id);

            notifications.show({
                title: `⚠ ${newest.severity?.toUpperCase()}: Machine ${newest.machine_id}`,
                message: "Click to view explanation",
                color: newest.severity === "critical" ? "red" : "yellow",
                autoClose: 8000,
                withBorder: true,
                onClick: () => {
                window.location.href = `/dashboard/alerts?alertId=${newest.id}`;
                },
            });
        }
      });
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return null;
}