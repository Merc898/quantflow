import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: {
    default: "QuantFlow — Institutional-Grade Trading Signals",
    template: "%s | QuantFlow",
  },
  description:
    "AI-powered quantitative research platform combining 50+ statistical models, " +
    "agentic market intelligence, and portfolio optimisation into a single SaaS API.",
  keywords: [
    "quantitative finance",
    "trading signals",
    "portfolio optimisation",
    "AI market intelligence",
    "algorithmic trading",
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
