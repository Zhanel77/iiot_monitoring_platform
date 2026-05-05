"use client";

import { useEffect, useMemo, useState } from "react";
import pageStyles from "@/components/dashboard-pages/section.module.css";
import { fetchJson, getAccessToken } from "@/components/dashboard-pages/api";
import {
  getRolePermissions,
  type UserRole,
} from "@/components/dashboard-pages/permissions";

type UserInfo = {
  id?: number;
  full_name?: string;
  email?: string;
  role?: string;
  is_active?: boolean;
};

export default function RegisterUserPageContent() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

  const [currentUser, setCurrentUser] = useState<UserInfo | null>(null);

  const [form, setForm] = useState({
    full_name: "",
    email: "",
    password: "",
    role: "viewer" as UserRole,
  });

  const [loadingUser, setLoadingUser] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  useEffect(() => {
    if (!apiUrl) {
      setError("NEXT_PUBLIC_API_URL is not configured");
      setLoadingUser(false);
      return;
    }

    const loadCurrentUser = async () => {
      try {
        const user = await fetchJson<UserInfo>(`${apiUrl}/api/v1/auth/me`);
        setCurrentUser(user);

        const permissions = getRolePermissions(user.role);
        const firstAllowedRole =
          permissions.allowedRolesToCreate[0] ?? "viewer";

        setForm((prev) => ({
          ...prev,
          role: firstAllowedRole,
        }));
      } catch (err) {
        const errorObj = err as Error;
        setError(errorObj.message || "Failed to load current user");
      } finally {
        setLoadingUser(false);
      }
    };

    loadCurrentUser();
  }, [apiUrl]);

  const allowedRoles = useMemo(() => {
    return getRolePermissions(currentUser?.role).allowedRolesToCreate;
  }, [currentUser?.role]);

  const handleChange =
    (field: "full_name" | "email" | "password" | "role") =>
    (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      setForm((prev) => ({
        ...prev,
        [field]: e.target.value,
      }));
    };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");

    if (!apiUrl) {
      setError("NEXT_PUBLIC_API_URL is not configured");
      return;
    }

    if (allowedRoles.length === 0) {
      setError("You do not have permission to register users");
      return;
    }

    if (!allowedRoles.includes(form.role)) {
      setError("You cannot create a user with this role");
      return;
    }

    const token = getAccessToken();
    if (!token) {
      setError("Authentication token not found");
      return;
    }

    try {
      setSubmitting(true);

      const response = await fetch(`${apiUrl}/api/v1/auth/register`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(form),
      });

      const contentType = response.headers.get("content-type") || "";
      const isJson = contentType.includes("application/json");
      const data = isJson ? await response.json() : await response.text();

      if (!response.ok) {
        let message = "Registration failed";

        if (typeof data === "string" && data.trim()) {
          message = data;
        } else if (Array.isArray((data as any)?.detail)) {
          message = (data as any).detail.map((item: any) => item.msg).join(", ");
        } else if (typeof (data as any)?.detail === "string") {
          message = (data as any).detail;
        }

        throw new Error(message);
      }

      setSuccess("User created successfully");
      setForm({
        full_name: "",
        email: "",
        password: "",
        role: allowedRoles[0] ?? "viewer",
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setSubmitting(false);
    }
  };

  if (loadingUser) {
    return (
      <main className={pageStyles.page}>
        <div className={pageStyles.stateBox}>Loading registration page...</div>
      </main>
    );
  }

  return (
    <main className={pageStyles.page}>
      <div className={pageStyles.header}>
        <p className={pageStyles.kicker}>User Management</p>
        <h1 className={pageStyles.title}>Register User</h1>
        <p className={pageStyles.subtitle}>
          Create new users with role-based access control.
        </p>
      </div>

      {error && <div className={pageStyles.errorBox}>{error}</div>}
      {success && <div className={pageStyles.stateBox}>{success}</div>}

      <div className={pageStyles.profileCard}>
        <form onSubmit={handleSubmit} className={pageStyles.form}>
          <div className={pageStyles.formField}>
            <label className={pageStyles.formLabel}>Full name</label>
            <input
              type="text"
              value={form.full_name}
              onChange={handleChange("full_name")}
              placeholder="Enter full name"
              className={pageStyles.formInput}
            />
          </div>

          <div className={pageStyles.formField}>
            <label className={pageStyles.formLabel}>Email</label>
            <input
              type="email"
              value={form.email}
              onChange={handleChange("email")}
              placeholder="Enter email"
              className={pageStyles.formInput}
            />
          </div>

          <div className={pageStyles.formField}>
            <label className={pageStyles.formLabel}>Password</label>
            <input
              type="password"
              value={form.password}
              onChange={handleChange("password")}
              placeholder="Create password"
              className={pageStyles.formInput}
            />
          </div>

          <div className={pageStyles.formField}>
            <label className={pageStyles.formLabel}>Role</label>
            <select
              value={form.role}
              onChange={handleChange("role")}
              className={pageStyles.formSelect}
            >
              {allowedRoles.map((role) => (
                <option key={role} value={role}>
                  {role}
                </option>
              ))}
            </select>
          </div>

          <button
            type="submit"
            disabled={submitting}
            className={pageStyles.primaryButton}
          >
            {submitting ? "Creating..." : "Create user"}
          </button>
        </form>
      </div>
    </main>
  );
}