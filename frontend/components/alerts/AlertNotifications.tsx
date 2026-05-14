"use client";

import { useEffect, useRef } from "react";
import { notifications } from "@mantine/notifications";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type WeatherStatus = {
  device_id: string;
  machine_id?: number;
  weather_impact?: string | null;
  environmental_reasons?: string[];
  outside_temp_c?: number | null;
  wind_speed?: number | null;
  weather_main?: string | null;
};

export function AlertNotifications() {
  const shownMachineAlerts = useRef(new Set<number>());
  const shownWeatherAlerts = useRef(new Set<string>());

  useEffect(() => {
    const loadMachineAlerts = async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/alerts/open`, {
          cache: "no-store",
        });

        const alerts = await res.json();

        if (!Array.isArray(alerts) || alerts.length === 0) return;

        alerts.forEach((alert) => {
          if (shownMachineAlerts.current.has(alert.id)) {
            return;
          }

          shownMachineAlerts.current.add(alert.id);

          notifications.show({
            title:
              alert.severity === "critical"
                ? `🚨 Critical Machine ${alert.machine_id}`
                : `⚠ Warning Machine ${alert.machine_id}`,

            message: (
              <div
                onClick={() => {
                  window.location.href = `/dashboard/alerts?alertId=${alert.id}`;
                }}
                style={{ cursor: "pointer" }}
              >
                {alert.message ||
                  "Operational risk exceeded normal threshold."}
              </div>
            ),

            color:
              alert.severity === "critical"
                ? "red"
                : "orange",

            autoClose: 10000,
            withBorder: true,
          });
        });
      } catch (error) {
        console.error("Failed to load machine alerts", error);
      }
    };

    const loadWeatherAlerts = async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/weather/devices-status`, {
          cache: "no-store",
        });

        const weatherItems: WeatherStatus[] = await res.json();

        if (!Array.isArray(weatherItems) || weatherItems.length === 0) return;

        weatherItems
          .filter((item) => {
            const impact = item.weather_impact?.toUpperCase();
            return impact === "MEDIUM" || impact === "HIGH";
          })
          .forEach((item) => {
            const alertKey = `${item.device_id}-${item.weather_impact}-${item.environmental_reasons?.join("_")}`;

            if (shownWeatherAlerts.current.has(alertKey)) return;

            shownWeatherAlerts.current.add(alertKey);

            const reasons =
              item.environmental_reasons && item.environmental_reasons.length > 0
                ? item.environmental_reasons.join(", ")
                : `${item.weather_main ?? "Weather anomaly"} detected`;

            notifications.show({
              title: `🌦 WEATHER IMPACT: ${item.device_id}`,
              message: (
                <div
                  onClick={() => {
                    window.location.href = "/dashboard/weather";
                  }}
                  style={{ cursor: "pointer" }}
                >
                  {reasons}. Temp: {item.outside_temp_c ?? "—"}°C, wind:{" "}
                  {item.wind_speed ?? "—"} m/s. Click to view weather monitoring.
                </div>
              ),
              color: item.weather_impact?.toUpperCase() === "HIGH" ? "red" : "orange",
              autoClose: 10000,
              withBorder: true,
            });
          });
      } catch (error) {
        console.error("Failed to load weather alerts", error);
      }
    };

    loadMachineAlerts();
    loadWeatherAlerts();

    const machineInterval = setInterval(loadMachineAlerts, 5000);
    const weatherInterval = setInterval(loadWeatherAlerts, 60000);

    return () => {
      clearInterval(machineInterval);
      clearInterval(weatherInterval);
    };
  }, []);

  return null;
}