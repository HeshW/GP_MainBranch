"use client";

import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { readStorage, writeStorage } from "@/lib/storage";

export type Language = "en" | "ar";
export type ThemeName = "care" | "midnight";

type PreferencesContextValue = {
  language: Language;
  theme: ThemeName;
  dir: "ltr" | "rtl";
  setLanguage: (language: Language) => void;
  toggleLanguage: () => void;
  toggleTheme: () => void;
};

const PreferencesContext = createContext<PreferencesContextValue | undefined>(undefined);

export function PreferencesProvider({ children }: { children: ReactNode }) {
  const [language, setLanguageState] = useState<Language>(() =>
    readStorage<Language>("nabda-language", "en"),
  );
  const [theme, setTheme] = useState<ThemeName>(() =>
    readStorage<ThemeName>("nabda-theme", "care"),
  );

  const dir: "ltr" | "rtl" = language === "ar" ? "rtl" : "ltr";

  useEffect(() => {
    document.documentElement.lang = language;
    document.documentElement.dir = dir;
    document.documentElement.dataset.theme = theme;
    writeStorage("nabda-language", language);
    writeStorage("nabda-theme", theme);
  }, [dir, language, theme]);

  function setLanguage(nextLanguage: Language) {
    setLanguageState(nextLanguage);
  }

  function toggleLanguage() {
    setLanguageState((current) => (current === "en" ? "ar" : "en"));
  }

  function toggleTheme() {
    setTheme((current) => (current === "care" ? "midnight" : "care"));
  }

  const value = useMemo(
    () => ({
      language,
      theme,
      dir,
      setLanguage,
      toggleLanguage,
      toggleTheme,
    }),
    [dir, language, theme],
  );

  return <PreferencesContext.Provider value={value}>{children}</PreferencesContext.Provider>;
}

export function usePreferences() {
  const context = useContext(PreferencesContext);
  if (!context) {
    throw new Error("usePreferences must be used within PreferencesProvider");
  }
  return context;
}
