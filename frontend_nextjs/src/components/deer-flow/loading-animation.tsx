

import { cn } from "~/lib/utils";

import styles from "./loading-animation.module.css";

export function LoadingAnimation({
  className,
  size = "normal",
}: {
  className?: string;
  size?: "normal" | "sm";
}) {
  return (
    <div
      className={cn(
        styles.loadingAnimation,
        size === "sm" && styles.sm,
        className,
      )}
    >
      <div></div>
      <div></div>
      <div></div>
    </div>
  );
}
