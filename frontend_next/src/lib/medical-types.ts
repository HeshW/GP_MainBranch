export interface MetaInfo {
  api_version: string;
  project?: string;
  rag_enabled: boolean;
  faiss_configured?: boolean;
  finetuned_classifier_enabled?: boolean;
  finetuned_model_configured?: boolean;
  therapy_enabled?: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  kind: "text" | "analysis" | "error";
  content?: string;
  payload?: AnalysisResponse;
}

export interface AnalysisResponse {
  status?: string;
  ocr?: Record<string, unknown> | null;
  report?: Record<string, unknown>;
  parsed?: {
    raw_text?: string;
    symptoms?: Array<{ symptom: string; source?: string; confidence?: number }>;
    labs?: Record<string, unknown>;
  };
  validated?: {
    symptoms?: string[];
    warnings?: string[];
    review_required?: boolean;
  };
  follow_up?: {
    answers?: string[];
    parsed?: {
      raw_text?: string;
      symptoms?: Array<{ symptom: string; source?: string; confidence?: number }>;
      labs?: Record<string, unknown>;
    };
    validated?: {
      symptoms?: string[];
      warnings?: string[];
      review_required?: boolean;
    };
    normalized_text?: string;
    updated_report?: Record<string, unknown>;
  };
  diagnosis?: {
    findings?: Array<{
      condition: string;
      confidence?: string | number;
      evidence?: string;
      severity?: string;
      source?: string;
    }>;
    rag_response?: string;
    retrieved_cases?: Array<{
      similarity?: number;
      pathology?: string;
      patient_id?: string;
      case_text?: string;
    }>;
    classifier_prediction?: {
      predicted_label?: string;
      confidence?: number;
      top_predictions?: Array<{ label: string; confidence: number }>;
    };
    final_diagnosis?: {
      diagnosis?: string;
      confidence?: number;
      source?: string;
      mode?: string;
      reasoning?: string;
      supporting_evidence?: string[];
    };
    diagnostic_candidates?: Array<{
      label: string;
      confidence?: number;
      sources?: string[];
    }>;
    differential_diagnosis?: Array<{
      label: string;
      confidence?: number;
      urgency?: string;
      evidence_for?: string[];
      evidence_against?: string[];
      missing_evidence?: string[];
      recommended_follow_up_questions?: string[];
      sources?: string[];
    }>;
    clarification?: {
      needed?: boolean;
      mode?: string;
      reasons?: string[];
      questions?: Array<{
        question: string;
        question_ar?: string;
        type?: string;
        target_conditions?: string[];
        reason?: string;
      }>;
      candidate_diseases?: Array<{
        label: string;
        confidence?: number;
        sources?: string[];
      }>;
    };
    ai_response?: string;
    ai_response_metadata?: {
      mode?: string;
      final_diagnosis?: string;
      provider_status?: string;
      response_language?: string;
      provider_name?: string;
      model_name?: string;
    };
    gemini_response?: string;
    gemini_response_metadata?: {
      mode?: string;
      final_diagnosis?: string;
      provider_status?: string;
      response_language?: string;
      provider_name?: string;
      model_name?: string;
    };
    summary?: string;
    decision_fusion?: {
      primary_source?: string;
      supporting_sources?: string[];
      rag_used?: boolean;
      classifier_used?: boolean;
      rule_validation_status?: string;
    };
    safety?: {
      clinician_review_required?: boolean;
      emergency_attention_recommended?: boolean;
      highest_rule_severity?: string;
      reasons?: string[];
    };
  };
  therapy?: {
    therapy_plan?: string;
    metadata?: {
      mode?: string;
      findings_count?: number;
      patient_info?: string;
    };
  };
  warnings?: string[];
  elapsed_ms?: number;
  [key: string]: unknown;
}
