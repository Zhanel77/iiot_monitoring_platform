"use client";

import {
  MapContainer,
  Marker,
  Popup,
  TileLayer,
  Circle,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
// @ts-ignore
import "leaflet.heat";
import { useEffect } from "react";
import L from "leaflet";

type DeviceWeatherStatus = {
  device_id: string;
  machine_id?: number;
  latitude?: number | null;
  longitude?: number | null;
  risk_score?: number | null;
  risk_level?: string | null;
  outside_temp_c?: number | null;
  outside_humidity?: number | null;
  wind_speed?: number | null;
  weather_main?: string | null;
  weather_impact?: string | null;
};

type Props = {
  devices: DeviceWeatherStatus[];
};

const markerIcon = new L.Icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});

function getCircleColor(device: DeviceWeatherStatus) {
  const level = device.weather_impact?.toUpperCase();

  if (level === "HIGH") return "#ef4444";
  if (level === "MEDIUM") return "#f59e0b";

  return "#22c55e";
}

function getRadius(device: DeviceWeatherStatus) {
  const temp = device.outside_temp_c ?? 0;

  return 15000 + temp * 800;
}

function HeatLayer({ devices }: Props) {
  const map = useMap();

  useEffect(() => {
    const heatPoints = devices
      .filter((d) => d.latitude && d.longitude)
      .map((device) => {
        const impact =
          device.weather_impact?.toUpperCase() === "HIGH"
            ? 1.0
            : device.weather_impact?.toUpperCase() === "MEDIUM"
            ? 0.6
            : 0.25;

        return [
          device.latitude!,
          device.longitude!,
          impact,
        ];
      });

    // @ts-ignore
    const heat = L.heatLayer(heatPoints, {
      radius: 45,
      blur: 30,
      maxZoom: 8,
    });

    heat.addTo(map);

    return () => {
      map.removeLayer(heat);
    };
  }, [devices, map]);

  return null;
}

export default function WeatherMap({ devices }: Props) {
  return (
    <div
      style={{
        width: "100%",
        height: "500px",
        borderRadius: "24px",
        overflow: "hidden",
      }}
    >
      <MapContainer
        center={[48.0196, 66.9237]}
        zoom={5}
        style={{ width: "100%", height: "100%" }}
      >
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        <HeatLayer devices={devices} />

        {devices.map((device) => {
          if (!device.latitude || !device.longitude) return null;

          return (
            <div key={device.device_id}>
              <Circle
                center={[device.latitude, device.longitude]}
                radius={getRadius(device)}
                pathOptions={{
                  color: getCircleColor(device),
                  fillColor: getCircleColor(device),
                  fillOpacity: 0.35,
                }}
              />

              <Marker
                position={[device.latitude, device.longitude]}
                icon={markerIcon}
              >
                <Popup>
                  <div style={{ minWidth: 220 }}>
                    <h3>{device.device_id}</h3>

                    <p>
                      <strong>Risk:</strong>{" "}
                      {device.risk_score?.toFixed(3) ?? "—"}
                    </p>

                    <p>
                      <strong>Status:</strong>{" "}
                      {device.risk_level ?? "—"}
                    </p>

                    <p>
                      <strong>Temperature:</strong>{" "}
                      {device.outside_temp_c ?? "—"}°C
                    </p>

                    <p>
                      <strong>Humidity:</strong>{" "}
                      {device.outside_humidity ?? "—"}%
                    </p>

                    <p>
                      <strong>Wind:</strong>{" "}
                      {device.wind_speed ?? "—"} m/s
                    </p>

                    <p>
                      <strong>Weather:</strong>{" "}
                      {device.weather_main ?? "—"}
                    </p>

                    <p>
                      <strong>Impact:</strong>{" "}
                      {device.weather_impact ?? "—"}
                    </p>
                  </div>
                </Popup>
              </Marker>
            </div>
          );
        })}
      </MapContainer>
    </div>
  );
}