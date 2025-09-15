

"use client";

import { usePathname } from "next/navigation";

import { ThemeProvider } from "~/components/theme-provider";

export function ThemeProviderWrapper({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <ThemeProvider
      attribute="class"
      defaultTheme={"light"}
      enableSystem={false}
      forcedTheme={"light"}
      disableTransitionOnChange
    >
      {children}
    </ThemeProvider>
  );
}
