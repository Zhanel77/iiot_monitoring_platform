"use client";

import { useEffect, useState } from "react";
import pageStyles from "@/components/dashboard-pages/section.module.css";
import { fetchJson, getAccessToken } from "@/components/dashboard-pages/api";

type UserItem = {
  id: number;
  email: string;
  full_name: string;
  role: "admin" | "operator" | "viewer";
  is_active: boolean;
};

export default function UsersManagementPageContent() {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

  const [users, setUsers] = useState<UserItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const loadUsers = async () => {
    if (!apiUrl) {
      setError("NEXT_PUBLIC_API_URL is not configured");
      setLoading(false);
      return;
    }

    try {
      const data = await fetchJson<UserItem[]>(`${apiUrl}/api/v1/users`);
      setUsers(Array.isArray(data) ? data : []);
      setError("");
    } catch (err) {
      const errorObj = err as Error;
      setError(errorObj.message || "Failed to load users");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadUsers();
  }, [apiUrl]);

  const handleRoleChange = async (
    userId: number,
    newRole: "admin" | "operator" | "viewer"
  ) => {
    if (!apiUrl) return;

    const token = getAccessToken();
    if (!token) {
      setError("Authentication token not found");
      return;
    }

    try {
      setError("");
      setSuccess("");

      const response = await fetch(`${apiUrl}/api/v1/users/${userId}/role`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ role: newRole }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to update role");
      }

      setSuccess("User role updated successfully");
      await loadUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    }
  };

  const handleStatusChange = async (userId: number, isActive: boolean) => {
    if (!apiUrl) return;

    const token = getAccessToken();
    if (!token) {
      setError("Authentication token not found");
      return;
    }

    try {
      setError("");
      setSuccess("");

      const response = await fetch(`${apiUrl}/api/v1/users/${userId}/status`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ is_active: isActive }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to update status");
      }

      setSuccess("User status updated successfully");
      await loadUsers();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    }
  };

  return (
    <main className={pageStyles.page}>
      <div className={pageStyles.header}>
        <p className={pageStyles.kicker}>Administration</p>
        <h1 className={pageStyles.title}>Users Management</h1>
        <p className={pageStyles.subtitle}>
          Manage users, update roles, and control account status.
        </p>
      </div>

      {error && <div className={pageStyles.errorBox}>{error}</div>}
      {success && <div className={pageStyles.stateBox}>{success}</div>}

      {loading ? (
        <div className={pageStyles.stateBox}>Loading users...</div>
      ) : users.length === 0 ? (
        <div className={pageStyles.stateBox}>No users found.</div>
      ) : (
        <div className={pageStyles.tableWrap}>
          <table className={pageStyles.table}>
            <thead>
              <tr>
                <th>ID</th>
                <th>Full name</th>
                <th>Email</th>
                <th>Role</th>
                <th>Status</th>
                <th>Change Role</th>
                <th>Change Status</th>
              </tr>
            </thead>
            <tbody>
              {users.map((user) => (
                <tr key={user.id}>
                  <td>{user.id}</td>
                  <td>{user.full_name || "—"}</td>
                  <td>{user.email}</td>
                  <td>
                    <span
                      className={`${pageStyles.badge} ${
                        user.role === "admin"
                          ? pageStyles.badgeCritical
                          : user.role === "operator"
                          ? pageStyles.badgeWarning
                          : pageStyles.badgeNormal
                      }`}
                    >
                      {user.role}
                    </span>
                  </td>
                  <td>
                    <span
                      className={`${pageStyles.badge} ${
                        user.is_active
                          ? pageStyles.badgeNormal
                          : pageStyles.badgeCritical
                      }`}
                    >
                      {user.is_active ? "active" : "inactive"}
                    </span>
                  </td>
                  <td>
                    <select
                      className={pageStyles.formSelect}
                      value={user.role}
                      onChange={(e) =>
                        handleRoleChange(
                          user.id,
                          e.target.value as "admin" | "operator" | "viewer"
                        )
                      }
                    >
                      <option value="viewer">viewer</option>
                      <option value="operator">operator</option>
                      <option value="admin">admin</option>
                    </select>
                  </td>
                  <td>
                    <button
                      className={pageStyles.secondaryButton}
                      onClick={() => handleStatusChange(user.id, !user.is_active)}
                    >
                      {user.is_active ? "Deactivate" : "Activate"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}