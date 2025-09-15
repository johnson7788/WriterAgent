

import { cn } from "~/lib/utils";

import styles from "./rainbow-text.module.css";

export function RainbowText({
  animated,
  className,
  children,
}: {
  animated?: boolean;
  className?: string;
  children?: React.ReactNode;
}) {
  return (
    <span className={cn(animated && styles.animated, className)}>
      {children}
    </span>
  );
}
