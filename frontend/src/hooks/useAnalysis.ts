import { useState, useCallback } from 'react';
import { postLabs, postImage, postSymptoms } from '@/api/client';

export function useAnalysis() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const resetState = () => {
    setLoading(true);
    setError(null);
    setResult(null);
  };

  const throwOrSetError = (e: unknown) => {
    setError(e instanceof Error ? e.message : String(e));
  };

  const runLabs = useCallback(async (labsJson: string, symptomsExtra: string) => {
    resetState();
    try {
      let labs: Record<string, unknown>;
      try {
        labs = JSON.parse(labsJson) as Record<string, unknown>;
      } catch {
        throw new Error("Invalid JSON in lab values.");
      }
      const data = await postLabs({
        labs,
        symptoms: symptomsExtra.trim() || undefined,
      });
      setResult(data);
    } catch (e) {
      throwOrSetError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  const runImage = useCallback(async (file: File | null) => {
    if (!file) {
      setError("Choose an image file first.");
      return;
    }
    resetState();
    try {
      const data = await postImage(file);
      setResult(data);
    } catch (e) {
      throwOrSetError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  const runSymptoms = useCallback(async (symptomText: string, useParser: boolean) => {
    resetState();
    try {
      const data = await postSymptoms({
        text: symptomText,
        use_symptom_parser: useParser,
      });
      setResult(data);
    } catch (e) {
      throwOrSetError(e);
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
