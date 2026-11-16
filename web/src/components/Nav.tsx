"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Live" },
  { href: "/backtest", label: "Backtest" },
  { href: "/paper", label: "Paper" },
];

export default function Nav() {
  const path = usePathname();
  return (
    <header className="border-b border-zinc-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-baseline gap-8 px-4 py-4">
        <Link href="/" className="font-display text-xl tracking-tight text-zinc-900">
          QuantLab
        </Link>
        <nav className="flex gap-5 text-sm">
          {links.map((l) => {
            const active = path === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={
                  active
                    ? "font-medium text-teal-800"
                    : "text-zinc-500 hover:text-zinc-800"
                }
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
