import type { Metadata } from "next";
import Nav from "@/components/Nav";
import Providers from "@/components/Providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "QuantLab",
  description: "Backtesting and paper-trading dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">
        <Providers>
          <Nav />
          <main className="mx-auto max-w-6xl px-4 py-8">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
