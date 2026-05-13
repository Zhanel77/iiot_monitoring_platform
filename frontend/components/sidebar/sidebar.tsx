"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import styles from "./sidebar.module.css";
import { getRolePermissions } from "@/components/dashboard-pages/permissions";
import { BarChart3 } from "lucide-react";

type SidebarProps = {
  user?: {
    full_name?: string;
    email?: string;
    role?: string;
  };
};

export default function Sidebar({ user }: SidebarProps) {
  const pathname = usePathname();
  const router = useRouter();

  const permissions = getRolePermissions(user?.role);

  const menuItems = [
    permissions.canViewDashboard
      ? { label: "Control Center", href: "/dashboard", icon: "▣" }
      : null,
    permissions.canViewDashboard
      ? { label: "Weather Monitoring", href: "/dashboard/weather", icon: "☁" }
      : null,
    permissions.canViewDevices
      ? { label: "Machines", href: "/dashboard/devices", icon: "⌘" }
      : null,
    permissions.canViewPredictions
      ? { label: "Telemetry Events", href: "/dashboard/predictions", icon: "◔" }
      : null,
    permissions.canViewAlerts
      ? { label: "Alarm Center", href: "/dashboard/alerts", icon: "⚠" }
      : null,
    permissions.canRegisterUsers
      ? { label: "Register Operator", href: "/dashboard/register", icon: "✚" }
      : null,
    permissions.canManageUsers
      ? { label: "Operators", href: "/dashboard/users", icon: "☰" }
      : null,
    permissions.canViewProfile
      ? { label: "Operator Console", href: "/dashboard/profile", icon: "◉" }
      : null,

    permissions.canViewDashboard
      ? { label: "Grafana", href: "/dashboard/grafana", icon: "📊" }
      : null,
  ].filter(Boolean) as { label: string; href: string; icon: string }[];

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
    router.push("/login");
  };

  return (
    <aside className={styles.sidebar}>
      <div>
        <div className={styles.brand}>
          <div className={styles.logo}>II</div>
          <div>
            <p className={styles.brandLabel}>IIoT Monitoring</p>
            <h2 className={styles.brandTitle}>Control Center</h2>
          </div>
        </div>

        <nav className={styles.nav}>
          {menuItems.map((item) => {
            const isActive = pathname === item.href;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`${styles.navItem} ${isActive ? styles.navItemActive : ""}`}
              >
                <span className={styles.navIcon}>{item.icon}</span>
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>
      </div>

      <div className={styles.profileCard}>
        <div className={styles.avatar}>
          {(user?.full_name || "U").charAt(0).toUpperCase()}
        </div>

        <div className={styles.profileInfo}>
          <p className={styles.profileName}>{user?.full_name || "User"}</p>
          <p className={styles.profileEmail}>{user?.email || "No email"}</p>
        </div>

        <div className={styles.roleRow}>
          <span className={styles.roleLabel}>Role</span>
          <span className={styles.roleBadge}>{user?.role || "viewer"}</span>
        </div>

        <button onClick={handleLogout} className={styles.logoutButton}>
          Logout
        </button>
      </div>
    </aside>
  );
}
