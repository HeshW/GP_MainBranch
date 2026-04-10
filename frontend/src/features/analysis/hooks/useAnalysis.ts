import { useCallback, useState } from "react";
import { postImage, postLabs, postSymptoms } from "@/shared/api";
import { AnalysisResponse } from "@/shared/types";

function parseLabsJson(labsJson: string): Record<string, unknown> {
  try {
    return JSON.parse(labsJson) as Record<string, unknown>;
  } catch {
    throw new Error("Invalid JSON in lab values.");
  }
}

export function useAnalysis() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const beginRequest = () => {
    setLoading(true);
    setError(null);
    setResult(null);
  };

  const captureError = (value: unknown) => {
    setError(value instanceof Error ? value.message : String(value));
  };

  const runLabs = useCallback(async (labsJson: string, symptomsExtra: string) => {
    beginRequest();
    try {
      const data = await postLabs({
        labs: parseLabsJson(labsJson),
        symptoms: symptomsExtra.trim() || undefined,
      });
      setResult(data);
    } catch (value) {
      captureError(value);
    } finally {
      setLoading(false);
    }
  }, []);

  const runImage = useCallback(async (file: File | null) => {
    if (!file) {
      setError("Choose an image file first.");
      return;
    }

    beginRequest();
    try {
      const data = await postImage(file);
      setResult(data);
    } catch (value) {
      captureError(value);
    } finally {
      setLoading(false);
    }
  }, []);

  const runSymptoms = useCallback(async (symptomText: string, useParser: boolean) => {
    beginRequest();
    try {
      const data = await postSymptoms({
        text: symptomText,
        use_symptom_parser: useParser,
      });
      setResult(data);
    } catch (value) {
      captureError(value);
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    result,
    error,
    runLabs,
    runImage,
    runSymptoms,
  };
}
