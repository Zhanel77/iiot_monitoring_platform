export type UserRole = "admin" | "operator" | "viewer";

export const rolePermissions = {
  admin: {
    canViewDashboard: true,
    canViewDevices: true,
    canViewPredictions: true,
    canViewAlerts: true,
    canViewProfile: true,
    canRegisterUsers: true,
    canManageUsers: true,
    allowedRolesToCreate: ["admin", "operator", "viewer"] as UserRole[],
  },
  operator: {
    canViewDashboard: true,
    canViewDevices: true,
    canViewPredictions: true,
    canViewAlerts: true,
    canViewProfile: true,
    canRegisterUsers: true,
    canManageUsers: false,
    allowedRolesToCreate: ["viewer"] as UserRole[],
  },
  viewer: {
    canViewDashboard: true,
    canViewDevices: false,
    canViewPredictions: false,
    canViewAlerts: false,
    canViewProfile: true,
    canRegisterUsers: false,
    canManageUsers: false,
    allowedRolesToCreate: [] as UserRole[],
  },
} as const;

export function getRolePermissions(role?: string) {
  if (role === "admin") return rolePermissions.admin;
  if (role === "operator") return rolePermissions.operator;
  return rolePermissions.viewer;
}

export function canAccessPath(role: string | undefined, pathname: string) {
  const permissions = getRolePermissions(role);

  if (pathname === "/dashboard") return permissions.canViewDashboard;
  if (pathname.startsWith("/dashboard/weather")) return permissions.canViewDashboard;
  if (pathname.startsWith("/dashboard/devices")) return permissions.canViewDevices;
  if (pathname.startsWith("/dashboard/predictions")) return permissions.canViewPredictions;
  if (pathname.startsWith("/dashboard/alerts")) return permissions.canViewAlerts;
  if (pathname.startsWith("/dashboard/profile")) return permissions.canViewProfile;
  if (pathname.startsWith("/dashboard/register")) return permissions.canRegisterUsers;
  if (pathname.startsWith("/dashboard/users")) return permissions.canManageUsers;


  return false;
}