import type { Metadata } from "next";
import { Zap } from "lucide-react";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Authentication",
};

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-background px-4">
      {/* Logo */}
      <Link
        href="/"
        className="mb-8 flex items-center gap-2 text-foreground transition-opacity hover:opacity-80"
      >
        <Zap className="h-8 w-8 text-primary" />
        <span className="text-2xl font-bold">QuantFlow</span>
      </Link>

      {/* Card */}
      <div className="w-full max-w-md rounded-xl border bg-card p-8 shadow-xl">
        {children}
      </div>

      {/* Footer */}
      <p className="mt-6 text-center text-xs text-muted-foreground">
        By continuing you agree to our{" "}
        <Link href="/terms" className="underline hover:text-foreground">
          Terms of Service
        </Link>{" "}
        and{" "}
        <Link href="/privacy" className="underline hover:text-foreground">
          Privacy Policy
        </Link>
        .
      </p>
    </div>
  );
}
