"use client";

import { useEffect, useRef } from "react";
import { notifications } from "@mantine/notifications";
import { usePathname } from "next/navigation";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const STORAGE_KEY = "shown-machine-alerts";
const NOTIFICATION_ID = "machine-alert-notification";

type MachineAlert = {
  id: number;
  device_id?: string;
  machine_id?: number;
  risk_level?: string;
  risk_score?: number;
  severity?: string;
  alert_type?: string;
  message?: string;
  created_at?: string;
};

export function MachineShapNotifications() {
  const pathname = usePathname();
  const shownRef = useRef<Set<number>>(new Set());
  const initializedRef = useRef(false);
  const queueRef = useRef<MachineAlert[]>([]);
  const isShowingRef = useRef(false);

  useEffect(() => {
    if (!pathname.startsWith("/dashboard")) {
      return;
    }

    const saved = sessionStorage.getItem(STORAGE_KEY);
    if (saved) {
      shownRef.current = new Set(JSON.parse(saved));
    }

    const showNextAlert = () => {
      if (isShowingRef.current) return;

      const nextAlert = queueRef.current.shift();
      if (!nextAlert) return;

      isShowingRef.current = true;

      const level =
        nextAlert.risk_level?.toUpperCase() ||
        nextAlert.severity?.toUpperCase() ||
        "WARNING";

      notifications.hide(NOTIFICATION_ID);

      notifications.show({
        id: NOTIFICATION_ID,
        title:
          level === "CRITICAL"
            ? `🚨 Critical Machine ${nextAlert.machine_id}`
            : `⚠ Warning Machine ${nextAlert.machine_id}`,
        message: (
          <div
            onClick={() => {
              window.location.href = `/dashboard/alerts?alertId=${nextAlert.id}`;
            }}
            style={{ cursor: "pointer" }}
          >
            {nextAlert.message ||
              `AI detected ${level.toLowerCase()} risk for ${
                nextAlert.device_id || `Machine ${nextAlert.machine_id}`
              }.`}
          </div>
        ),
        color: level === "CRITICAL" ? "red" : "orange",
        autoClose: 7000,
        withBorder: true,
        onClose: () => {
          isShowingRef.current = false;

          setTimeout(() => {
            showNextAlert();
          }, 2500);
        },
      });
    };

    const loadMachineAlerts = async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/alerts/open`, {
          cache: "no-store",
        });

        const data: MachineAlert[] = await res.json();

        if (!Array.isArray(data)) return;

        const alerts = data
          .filter((item) => {
            const level =
              item.risk_level?.toUpperCase() || item.severity?.toUpperCase();

            return (
              (level === "CRITICAL" || level === "WARNING") &&
              item.alert_type === "cloud_shap_risk"
            );
          })
          .sort((a, b) => {
            const aTime = new Date(a.created_at || 0).getTime();
            const bTime = new Date(b.created_at || 0).getTime();
            return aTime - bTime;
          });

        // первый заход: просто запоминаем старые alerts, НЕ показываем пачку
        if (!initializedRef.current) {
          alerts.forEach((alert) => shownRef.current.add(alert.id));

          sessionStorage.setItem(
            STORAGE_KEY,
            JSON.stringify([...shownRef.current])
          );

          initializedRef.current = true;
          return;
        }

        const newAlerts = alerts.filter((alert) => {
          return !shownRef.current.has(alert.id);
        });

        if (newAlerts.length === 0) return;

        newAlerts.forEach((alert) => {
          shownRef.current.add(alert.id);
          queueRef.current.push(alert);
        });

        sessionStorage.setItem(
          STORAGE_KEY,
          JSON.stringify([...shownRef.current])
        );

        showNextAlert();
      } catch (error) {
        console.error("Failed to load machine SHAP alerts", error);
      }
    };

    loadMachineAlerts();

    const interval = setInterval(loadMachineAlerts, 15000);

    return () => {
      clearInterval(interval);
    };
  }, [pathname]);

  return null;
}