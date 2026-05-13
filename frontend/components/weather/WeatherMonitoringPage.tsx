"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { fetchJson } from "@/components/dashboard-pages/api";
import styles from "./weather.module.css";

const WeatherMap = dynamic(
  () => import("@/components/dashboard/WeatherMap"),
  { ssr: false }
);

type DeviceWeatherStatus = {
  device_id: string;
  machine_id?: number;
  latitude?: number | null;
  longitude?: number | null;
  risk_score?: number | null;
  risk_level?: string | null;
  outside_temp_c?: number | null;
  outside_humidity?: number | null;
  outside_pressure?: number | null;
  wind_speed?: number | null;
  weather_main?: string | null;
  weather_impact?: string | null;
  environmental_reasons?: string[];
};

export default function WeatherMonitoringPage() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  const [items, setItems] = useState<DeviceWeatherStatus[]>([]);
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
        const data = await fetchJson<DeviceWeatherStatus[]>(
          `${apiUrl}/api/v1/weather/devices-status`
        );

        setItems(Array.isArray(data) ? data : []);
        setError("");
      } catch (err) {
        setError((err as Error).message || "Failed to load weather data");
      } finally {
        setLoading(false);
      }
    };

    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, [apiUrl]);

  if (loading) {
    return <main className={styles.page}>Loading weather monitoring...</main>;
  }

  return (
    <main className={styles.page}>
      <div className={styles.header}>
        <div>
          <p className={styles.kicker}>Climate-aware IIoT Monitoring</p>
          <h1 className={styles.title}>Weather Monitoring</h1>
          <p className={styles.subtitle}>
            External weather conditions linked with machine risk and device locations.
          </p>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <section className={styles.mapPanel}>
        <WeatherMap devices={items} />
      </section>

      <section className={styles.tablePanel}>
        <h2 className={styles.sectionTitle}>Device Environmental Status</h2>

        <div className={styles.table}>
          {items.map((item) => (
            <div key={item.device_id} className={styles.row}>
              <div>
                <strong>{item.device_id}</strong>
                <p>Machine {item.machine_id ?? "—"}</p>
              </div>

              <span>{item.outside_temp_c?.toFixed(1) ?? "—"}°C</span>
              <span>{item.outside_humidity ?? "—"}%</span>
              <span>{item.wind_speed ?? "—"} m/s</span>
              <span>{item.weather_main ?? "—"}</span>

              <b className={getImpactClass(item.weather_impact)}>
                {item.weather_impact ?? "—"}
              </b>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}

function getImpactClass(impact?: string | null) {
  const value = impact?.toUpperCase();

  if (value === "HIGH") return styles.high;
  if (value === "MEDIUM") return styles.medium;

  return styles.low;
}