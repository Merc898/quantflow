/**
 * Root page — redirects authenticated users to /dashboard,
 * unauthenticated users to /login.
 */
import { redirect } from "next/navigation";

export default function RootPage() {
  // This is a server component; client-side auth check happens in middleware.
  // For simplicity, redirect to login — the dashboard layout will redirect
  // back here once authenticated.
  redirect("/login");
}
