import React from 'react';
import { Tab } from '@/types';

interface TabNavigationProps {
  currentTab: Tab;
  onTabChange: (tab: Tab) => void;
}

export function TabNavigation({ currentTab, onTabChange }: TabNavigationProps) {
  return (
    <nav className="tabs" aria-label="Analysis mode">
      <button
        type="button"
        className={currentTab === "labs" ? "is-active" : ""}
        onClick={() => onTabChange("labs")}
      >
        Manual labs
      </button>
      <button
        type="button"
        className={currentTab === "image" ? "is-active" : ""}
        onClick={() => onTabChange("image")}
      >
        Report image
      </button>
      <button
        type="button"
        className={currentTab === "symptoms" ? "is-active" : ""}
        onClick={() => onTabChange("symptoms")}
      >
        Symptoms text
      </button>
    </nav>
  );
}
