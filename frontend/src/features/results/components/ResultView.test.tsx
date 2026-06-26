import { render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";

import { ResultView } from "./ResultView";

test("result view hides final diagnosis when clarification is pending", () => {
  render(
    <ResultView
      error={null}
      result={{
        report: { raw_text: "I feel thirsty and have fatigue for a week" },
        diagnosis: {
          assessment_state: "needs_clarification",
          final_diagnosis: {
            diagnosis: "Myocarditis",
            confidence: 0.53,
            source: "classifier",
          },
          clarification: {
            needed: true,
            reasons: ["Current diagnosis confidence is below the clarification threshold."],
            questions: [
              {
                question: "Do you also have chest pain or shortness of breath?",
              },
            ],
          },
          diagnostic_candidates: [
            { label: "Myocarditis", confidence: 0.53, sources: ["classifier"] },
          ],
        },
      }}
      onClarify={vi.fn().mockResolvedValue(undefined)}
    />,
  );

  expect(screen.queryByText("Final Diagnosis")).toBeNull();
  expect(screen.getByText("Assessment Pending")).toBeTruthy();
  expect(screen.getByText("Clarification Mode")).toBeTruthy();
});