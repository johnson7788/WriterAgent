

import type { LucideIcon } from "lucide-react";
import type { FunctionComponent } from "react";

import type { SettingsState } from "~/core/store";

export type Tab = FunctionComponent<{
  settings: SettingsState;
  onChange: (changes: Partial<SettingsState>) => void;
}> & {
  displayName?: string;
  icon?: LucideIcon;
  badge?: string;
};
