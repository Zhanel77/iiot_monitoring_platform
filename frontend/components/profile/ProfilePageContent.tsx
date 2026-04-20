"use client";

import { useEffect, useState } from "react";
import pageStyles from "@/components/dashboard-pages/section.module.css";
import { fetchJson } from "@/components/dashboard-pages/api";

type UserInfo = {
  id?: number;
  full_name?: string;
  email?: string;
  role?: string;
  is_active?: boolean;
};

export default function ProfilePageContent() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;
  const [user, setUser] = useState<UserInfo | null>(null);
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
        const data = await fetchJson<UserInfo>(`${apiUrl}/api/v1/auth/me`);
        setUser(data);
        setError("");
      } catch (err) {
        const errorObj = err as Error;
        setError(errorObj.message || "Failed to load profile");
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [apiUrl]);

  return (
    <main className={pageStyles.page}>
      <div className={pageStyles.header}>
        <p className={pageStyles.kicker}>Account</p>
        <h1 className={pageStyles.title}>Profile</h1>
        <p className={pageStyles.subtitle}>
          Information about the currently authenticated user.
        </p>
      </div>

      {loading ? (
        <div className={pageStyles.stateBox}>Loading profile...</div>
      ) : error ? (
        <div className={pageStyles.errorBox}>{error}</div>
      ) : !user ? (
        <div className={pageStyles.stateBox}>Profile data not available.</div>
      ) : (
        <div className={pageStyles.profileCard}>
          <div className={pageStyles.profileAvatar}>
            {(user.full_name || "U").charAt(0).toUpperCase()}
          </div>

          <div className={pageStyles.profileDetails}>
            <div className={pageStyles.profileRow}>
              <span className={pageStyles.profileLabel}>Full name</span>
              <span className={pageStyles.profileValue}>{user.full_name || "—"}</span>
            </div>

            <div className={pageStyles.profileRow}>
              <span className={pageStyles.profileLabel}>Email</span>
              <span className={pageStyles.profileValue}>{user.email || "—"}</span>
            </div>

            <div className={pageStyles.profileRow}>
              <span className={pageStyles.profileLabel}>Role</span>
              <span className={pageStyles.profileValue}>{user.role || "—"}</span>
            </div>

            <div className={pageStyles.profileRow}>
              <span className={pageStyles.profileLabel}>Status</span>
              <span className={pageStyles.profileValue}>
                {user.is_active ? "Active" : "Inactive"}
              </span>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}