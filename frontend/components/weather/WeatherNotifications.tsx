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

export function WeatherNotifications() {
  const shownRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const saved = sessionStorage.getItem("shown-weather-alerts");
    if (saved) {
      shownRef.current = new Set(JSON.parse(saved));
    }

    const loadWeatherAlerts = async () => {
      try {
        const res = await fetch(`${API_URL}/api/v1/weather/devices-status`, {
          cache: "no-store",
        });

        const items: WeatherStatus[] = await res.json();

        if (!Array.isArray(items)) return;

        items
          .filter((item) => {
            const impact = item.weather_impact?.toUpperCase();
            return impact === "MEDIUM" || impact === "HIGH";
          })
          .forEach((item) => {
            const key = `${item.device_id}-${item.weather_impact}`;

            if (shownRef.current.has(key)) return;

            shownRef.current.add(key);
            sessionStorage.setItem(
              "shown-weather-alerts",
              JSON.stringify([...shownRef.current])
            );

            const reasons =
              item.environmental_reasons?.length
                ? item.environmental_reasons.join(", ")
                : item.weather_main || "Weather anomaly detected";

            notifications.show({
              title: `🌦 Weather Impact · ${item.device_id}`,
              message: (
                <div
                  onClick={() => {
                    window.location.href = "/dashboard/weather";
                  }}
                  style={{ cursor: "pointer" }}
                >
                  {reasons}. Temp: {item.outside_temp_c ?? "—"}°C, wind:{" "}
                  {item.wind_speed ?? "—"} m/s.
                </div>
              ),
              color:
                item.weather_impact?.toUpperCase() === "HIGH"
                  ? "red"
                  : "orange",
              autoClose: 10000,
              withBorder: true,
            });
          });
      } catch (error) {
        console.error("Failed to load weather alerts", error);
      }
    };

    loadWeatherAlerts();
    const interval = setInterval(loadWeatherAlerts, 60000);

    return () => clearInterval(interval);
  }, []);

  return null;
}