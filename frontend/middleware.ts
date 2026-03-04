/**
 * Next.js middleware for route protection.
 *
 * - Dashboard routes require authentication (JWT in localStorage is client-side,
 *   so we use a "qf_token" cookie as the server-readable indicator).
 * - Unauthenticated users hitting /dashboard/* are redirected to /login.
 * - Authenticated users hitting /login or /register are redirected to /dashboard.
 */

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const AUTH_COOKIE = "qf_auth";
const PROTECTED_PREFIXES = ["/dashboard", "/signals", "/portfolio", "/screener", "/intelligence", "/settings"];
const AUTH_PATHS = ["/login", "/register"];

export function middleware(request: NextRequest): NextResponse {
  const { pathname } = request.nextUrl;
  const isAuthenticated = request.cookies.has(AUTH_COOKIE);

  const isProtected = PROTECTED_PREFIXES.some((p) => pathname.startsWith(p));
  const isAuthPage = AUTH_PATHS.some((p) => pathname === p);

  if (isProtected && !isAuthenticated) {
    const url = request.nextUrl.clone();
    url.pathname = "/login";
    url.searchParams.set("from", pathname);
    return NextResponse.redirect(url);
  }

  if (isAuthPage && isAuthenticated) {
    const url = request.nextUrl.clone();
    url.pathname = "/dashboard";
    return NextResponse.redirect(url);
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    "/dashboard/:path*",
    "/signals/:path*",
    "/portfolio/:path*",
    "/screener/:path*",
    "/intelligence/:path*",
    "/settings/:path*",
    "/login",
    "/register",
  ],
};
