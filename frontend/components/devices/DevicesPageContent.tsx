"use client";

import { useEffect, useState } from "react";
import pageStyles from "@/components/dashboard-pages/section.module.css";
import { fetchJson } from "@/components/dashboard-pages/api";

type Device = {
  id: number;
  device_id?: string;
  machine_id?: number;
  name?: string;
  status?: string;
};

export default function DevicesPageContent() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  const [devices, setDevices] = useState<Device[]>([]);
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
        const data = await fetchJson<Device[]>(`${apiUrl}/api/v1/devices`);
        setDevices(Array.isArray(data) ? data : []);
        setError("");
      } catch (err) {
        const errorObj = err as Error;
        setError(errorObj.message || "Failed to load devices");
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [apiUrl]);

  return (
    <main className={pageStyles.page}>
      <div className={pageStyles.header}>
        <p className={pageStyles.kicker}>Monitoring</p>
        <h1 className={pageStyles.title}>Devices</h1>
        <p className={pageStyles.subtitle}>
          Connected industrial devices registered in the system.
        </p>
      </div>

      {loading ? (
        <div className={pageStyles.stateBox}>Loading devices...</div>
      ) : error ? (
        <div className={pageStyles.errorBox}>{error}</div>
      ) : devices.length === 0 ? (
        <div className={pageStyles.stateBox}>No devices found.</div>
      ) : (
        <div className={pageStyles.grid}>
          {devices.map((device) => (
            <div key={device.id} className={pageStyles.card}>
              <p className={pageStyles.cardLabel}>Device</p>
              <h3 className={pageStyles.cardTitle}>
                {device.device_id || device.name || `Device ${device.id}`}
              </h3>

              <div className={pageStyles.metaBlock}>
                <span>Machine ID: {device.machine_id ?? "—"}</span>
                <span>Status: {device.status || "registered"}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </main>
  );
}