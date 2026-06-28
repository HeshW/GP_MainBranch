import { AnalysisTab } from "@/shared/types";

interface TabNavigationProps {
  currentTab: AnalysisTab;
  onTabChange: (tab: AnalysisTab) => void;
}

const TABS: Array<{ id: AnalysisTab; label: string }> = [
  { id: "labs", label: "Manual labs" },
  { id: "image", label: "Report image" },
  { id: "symptoms", label: "Symptoms text" },
];

export function TabNavigation({ currentTab, onTabChange }: TabNavigationProps) {
  return (
    <nav className="tabs" aria-label="Analysis mode">
      {TABS.map((tab) => (
        <button
          key={tab.id}
          type="button"
          className={currentTab === tab.id ? "is-active" : ""}
          onClick={() => onTabChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </nav>
  );
}
