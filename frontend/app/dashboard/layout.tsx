"use client";

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import Sidebar from "@/components/sidebar/sidebar";
import styles from "./dashboard-layout.module.css";
import { fetchJson, getAccessToken } from "@/components/dashboard-pages/api";
import { canAccessPath } from "@/components/dashboard-pages/permissions";

type UserInfo = {
  id?: number;
  full_name?: string;
  email?: string;
  role?: string;
  is_active?: boolean;
};

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

  const [user, setUser] = useState<UserInfo | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const run = async () => {
      const token = getAccessToken();

      if (!token) {
        router.replace("/login");
        return;
      }

      if (!apiUrl) {
        setReady(true);
        return;
      }

      try {
        const me = await fetchJson<UserInfo>(`${apiUrl}/api/v1/auth/me`);
        setUser(me);
        localStorage.setItem("user", JSON.stringify(me));

        const allowed = canAccessPath(me.role, pathname);

        if (!allowed) {
          router.replace("/dashboard");
          return;
        }
      } catch (err) {
        const error = err as Error & { status?: number };

        if (error.status === 401) {
          localStorage.removeItem("access_token");
          localStorage.removeItem("user");
          router.replace("/login");
          return;
        }

        router.replace("/login");
        return;
      } finally {
        setReady(true);
      }
    };

    run();
  }, [apiUrl, pathname, router]);

  if (!ready) {
    return <div className={styles.loading}>Loading...</div>;
  }

  return (
    <div className={styles.layout}>
      <Sidebar user={user || undefined} />
      <div className={styles.content}>{children}</div>
    </div>
  );
}