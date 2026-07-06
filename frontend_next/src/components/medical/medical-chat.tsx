"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChat,
  deleteChat,
  fetchChatMessages,
  fetchChats,
  postChat,
  postChatStream,
  postClarification,
  postImage,
  postLabs,
  postSymptoms,
  saveChatMessage,
  type ChatSession,
} from "@/lib/api";
import { useAuth } from "@/contexts/auth-context";
import { usePreferences } from "@/contexts/preferences-context";
import { getCopy } from "@/lib/i18n";
import type { AnalysisResponse, ChatMessage } from "@/lib/medical-types";

type SendMode = "auto" | "symptoms" | "labs" | "image" | "chat";

type ClarificationContext = {
  report: Record<string, unknown>;
  diagnosis: Record<string, unknown> | undefined;
  questions: string[];
  language: "en" | "ar";
};

type StoredAnalysisMessage = {
  text: string;
  analysis: AnalysisResponse;
};

type MedicalChatProps = {
  compact?: boolean;
  initialPrompt?: string;
  userName?: string;
};

const ANALYSIS_MESSAGE_PREFIX = "__NABDA_ANALYSIS_V1__:";

const DEFAULT_LABS_JSON = `{
  "glucose": 145,
  "hemoglobin": 11.2
}`;

const ARABIC_MEDICAL_TERMS: Record<string, string> = {
  "acute asthma exacerbation": "نوبة ربو حادة",
  "acute laryngitis": "التهاب الحنجرة الحاد",
  "acute rhinosinusitis": "التهاب الجيوب الأنفية الحاد",
  "acute viral illness": "عدوى فيروسية حادة",
  "anaphylaxis": "حساسية مفرطة",
  "anemia": "فقر الدم",
  "anaemia": "فقر الدم",
  "atrial fibrillation": "رجفان أذيني",
  "bronchitis": "التهاب الشعب الهوائية",
  "bronchospasm": "تشنج قصبي",
  "chronic rhinosinusitis": "التهاب الجيوب الأنفية المزمن",
  "dehydration": "جفاف",
  "diabetes": "السكري",
  "diabetes mellitus": "داء السكري",
  "emergency red-flag presentation": "أعراض إنذار طارئة",
  "gerd": "ارتجاع معدي مريئي",
  "gastroesophageal reflux": "ارتجاع معدي مريئي",
  "guillain barre": "متلازمة غيلان باريه",
  "guillain-barr": "متلازمة غيلان باريه",
  "heart attack": "نوبة قلبية",
  "hyperglycemia": "ارتفاع سكر الدم",
  "laryngospasm": "تشنج الحنجرة",
  "localized edema": "تورم موضعي",
  "lower respiratory infection": "عدوى بالجهاز التنفسي السفلي",
  "myasthenia gravis": "الوهن العضلي الوبيل",
  "myocardial infarction": "احتشاء عضلة القلب",
  "myocarditis": "التهاب عضلة القلب",
  "panic attack": "نوبة هلع",
  "pancreatic neoplasm": "ورم بالبنكرياس",
  "pericarditis": "التهاب غشاء القلب",
  "pneumonia": "التهاب رئوي",
  "possible acute viral illness pattern": "نمط محتمل لعدوى فيروسية حادة",
  "possible anemia-related symptom pattern": "نمط أعراض محتمل مرتبط بفقر الدم",
  "possible cardiopulmonary red-flag symptom pattern": "نمط أعراض إنذار قلبي رئوي محتمل",
  "possible gastroesophageal reflux pattern": "نمط محتمل لارتجاع معدي مريئي",
  "possible hyperglycemia / diabetes symptom pattern": "نمط أعراض محتمل لارتفاع السكر أو السكري",
  "possible lower respiratory infection pattern": "نمط محتمل لعدوى الجهاز التنفسي السفلي",
  "possible upper respiratory tract infection pattern": "نمط محتمل لعدوى الجهاز التنفسي العلوي",
  "prediabetes": "مرحلة ما قبل السكري",
  "psvt": "تسرع قلب فوق بطيني انتيابي",
  "pulmonary embolism": "انسداد رئوي",
  "pulmonary neoplasm": "ورم بالرئة",
  "reflux": "ارتجاع",
  "serious infection": "عدوى خطيرة",
  "stable angina": "ذبحة صدرية مستقرة",
  "stroke-like emergency": "حالة طارئة شبيهة بالسكتة الدماغية",
  "unstable angina": "ذبحة صدرية غير مستقرة",
  "upper respiratory tract infection": "عدوى الجهاز التنفسي العلوي",
  "urti": "عدوى الجهاز التنفسي العلوي",
  "viral pharyngitis": "التهاب بلعوم فيروسي",
};

const ARABIC_MEDICAL_PHRASES: Record<string, string> = {
  "Additional clarification is recommended before treating this as a final diagnosis.":
    "ينصح بالحصول على معلومات إضافية قبل التعامل مع هذا كتوصيف تشخيصي نهائي.",
  "Answering the follow-up questions will help refine the diagnosis.":
    "الإجابة على أسئلة المتابعة ستساعد في تحسين دقة التشخيص.",
  "Candidate diagnosis is available, but confidence or evidence agreement is insufficient for a confident final answer.":
    "يوجد تشخيص مرشح، لكن درجة الثقة أو توافق الأدلة غير كافيين لإجابة نهائية مؤكدة.",
  "Classifier candidate": "مرشح المصنف",
  "Diagnostic label canonicalized from": "تم توحيد اسم التشخيص من",
  "Follow-up questions are needed before making a stronger diagnostic claim.":
    "هناك حاجة إلى أسئلة متابعة قبل تقديم ترجيح تشخيصي أقوى.",
  "Needs immediate professional evaluation rather than app-only diagnosis.":
    "يحتاج إلى تقييم طبي فوري بدلا من الاعتماد على تشخيص التطبيق فقط.",
  "No clinically significant findings detected.": "لم يتم اكتشاف نتائج ذات دلالة سريرية مهمة.",
  "Rule findings": "نتائج القواعد",
  "Rule safety findings": "نتائج قواعد السلامة",
  "Rule safety layer flagged": "طبقة السلامة بالقواعد رصدت",
  "Therapy generation is disabled for this deployment and will be re-enabled in a later milestone. Please rely on diagnosis output and clinician review for now.":
    "إنشاء الخطة العلاجية غير مفعل في هذا الإصدار وسيتم تفعيله لاحقا. يرجى الاعتماد حاليا على نتيجة التشخيص ومراجعة الطبيب.",
  "Top retrieved case pathology": "تشخيص أقرب حالة مسترجعة",
};

const ARABIC_MEDICAL_QUESTIONS: Record<string, string> = {
  "Are symptoms mainly fluctuating eye/bulbar weakness (ptosis, speech/swallow fatigue), or more ascending limb weakness with tingling?":
    "هل الأعراض أساسا ضعف متذبذب بالعين أو البلع والكلام، أم ضعف صاعد بالأطراف مع تنميل؟",
  "Are symptoms mainly sore throat, runny nose, congestion, hoarseness, or cough?":
    "هل الأعراض أساسا ألم بالحلق أو رشح أو احتقان أو بحة صوت أو كحة؟",
  "Are symptoms mainly wheeze/chest tightness without fever or productive sputum, and do bronchodilators help?":
    "هل الأعراض أساسا صفير أو ضيق بالصدر بدون حمى أو بلغم، وهل تتحسن مع موسعات الشعب؟",
  "Are the palpitations irregular and uneven, or mostly sudden fast episodes that start and stop abruptly?":
    "هل الخفقان غير منتظم ومتفاوت، أم نوبات سريعة مفاجئة تبدأ وتنتهي فجأة؟",
  "Are you also having dizziness, shortness of breath on exertion, paleness, or unusual fatigue?":
    "هل لديك أيضا دوخة أو ضيق نفس مع المجهود أو شحوب أو تعب غير معتاد؟",
  "Did the breathing problem start suddenly?":
    "هل بدأت مشكلة التنفس فجأة؟",
  "Did the shortness of breath start suddenly, or was there recent immobility, leg swelling, or chest pain that worsens with breathing?":
    "هل بدأ ضيق التنفس فجأة؟ وهل كان هناك قلة حركة مؤخرا أو تورم بالساق أو ألم صدر يزيد مع التنفس؟",
  "Did weakness, facial droop, or speech trouble start suddenly?":
    "هل بدأ الضعف أو ميل الوجه أو صعوبة الكلام فجأة؟",
  "Do you also have fever, cough, sore throat, or nasal congestion?":
    "هل لديك أيضا حمى أو كحة أو ألم بالحلق أو احتقان بالأنف؟",
  "Do you get sudden episodes of difficulty breathing or a high-pitched sound when breathing in?":
    "هل تحدث نوبات مفاجئة من صعوبة التنفس أو صوت صفير حاد عند الشهيق؟",
  "Do you have clear infection signs (fever with productive cough), or mostly wheeze/chest tightness without infection features?":
    "هل لديك علامات عدوى واضحة مثل حمى مع كحة ببلغم، أم صفير أو ضيق صدر بدون علامات عدوى؟",
  "Do you have drooping eyelids, double vision, difficulty speaking or swallowing, or worsening weakness over the day?":
    "هل لديك تدلي بالجفن أو ازدواج بالرؤية أو صعوبة بالكلام أو البلع أو ضعف يزداد خلال اليوم؟",
  "Do you have pleuritic chest pain or higher-fever infection signs (favoring pneumonia), or mostly lingering cough after a recent cold (favoring bronchitis)?":
    "هل لديك ألم صدر يزيد مع التنفس أو علامات عدوى أشد ترجح الالتهاب الرئوي، أم كحة ممتدة بعد نزلة برد ترجح التهاب الشعب؟",
  "Do your symptoms get worse after meals or when lying down, with a sour or acidic taste in the mouth?":
    "هل تزيد الأعراض بعد الأكل أو عند الاستلقاء مع طعم حامضي أو حرقان بالفم؟",
  "Does discomfort worsen after meals or lying down, with heartburn or sour taste?":
    "هل يزيد الانزعاج بعد الأكل أو عند الاستلقاء مع حرقان أو طعم حامضي؟",
  "Does the chest discomfort appear with exertion and improve with rest?":
    "هل يظهر ألم أو ضيق الصدر مع المجهود ويتحسن بالراحة؟",
  "Has cough or weight loss been persistent or progressive over weeks to months?":
    "هل الكحة أو فقدان الوزن مستمران أو يتزايدان منذ أسابيع إلى شهور؟",
  "Has fluid intake been low, or is urine darker than usual?":
    "هل كان شرب السوائل قليلا أو البول أغمق من المعتاد؟",
  "Have you noticed increased thirst, frequent urination, weight loss, or blurred vision?":
    "هل لاحظت زيادة في العطش أو كثرة التبول أو فقدان وزن أو زغللة في النظر؟",
  "Is the chest discomfort related to exertion, deep breathing, or an irregular heartbeat/palpitations?":
    "هل ألم الصدر مرتبط بالمجهود أو التنفس العميق أو خفقان أو عدم انتظام ضربات القلب؟",
  "Is the chest pain appearing at rest or worsening recently, versus mainly with exertion and improving with rest?":
    "هل ألم الصدر يظهر أثناء الراحة أو يزداد مؤخرا، أم يحدث أساسا مع المجهود ويتحسن بالراحة؟",
  "Is there chest pain, shortness of breath, palpitations, or fainting?":
    "هل يوجد ألم بالصدر أو ضيق نفس أو خفقان أو إغماء؟",
  "Is there confusion, stiff neck, severe headache, rash, or rapidly worsening fever?":
    "هل يوجد تشوش أو تيبس بالرقبة أو صداع شديد أو طفح أو حمى تزداد بسرعة؟",
  "Is there fever with productive cough, chills, or pleuritic chest pain?":
    "هل توجد حمى مع كحة ببلغم أو رعشة أو ألم صدر يزيد مع التنفس؟",
  "Is there frequent urination, blurred vision, weight loss, or elevated glucose?":
    "هل يوجد كثرة تبول أو زغللة في النظر أو فقدان وزن أو ارتفاع في السكر؟",
  "Is there progressive upper abdominal pain, weight loss, appetite loss, or jaundice?":
    "هل يوجد ألم متزايد بأعلى البطن أو فقدان وزن أو فقدان شهية أو اصفرار؟",
  "Is there unusual fatigue with dizziness, paleness, shortness of breath, or palpitations?":
    "هل يوجد تعب غير معتاد مع دوخة أو شحوب أو ضيق نفس أو خفقان؟",
  "Is there pleuritic chest pain, leg swelling, recent immobility, or recent surgery?":
    "هل يوجد ألم صدر يزيد مع التنفس أو تورم بالساق أو قلة حركة مؤخرا أو جراحة حديثة؟",
  "Was the breathing problem sudden with pleuritic pain or leg swelling, or did it follow a gradual upper respiratory infection?":
    "هل بدأت مشكلة التنفس فجأة مع ألم يزيد بالتنفس أو تورم ساق، أم جاءت تدريجيا بعد عدوى تنفسية علوية؟",
  "Was there a recent viral illness before the chest symptoms?":
    "هل سبقت أعراض الصدر عدوى فيروسية حديثة؟",
};

const ARABIC_URGENCY_LABELS: Record<string, string> = {
  emergency: "طارئ",
  urgent: "عاجل",
  prompt: "قريب",
  routine: "روتيني",
};

const ARABIC_EVIDENCE_TERMS: Record<string, string> = {
  "abdominal pain": "ألم بالبطن",
  "appetite loss": "فقدان الشهية",
  "blurred vision": "زغللة في النظر",
  "chest discomfort": "انزعاج بالصدر",
  "chest pain": "ألم بالصدر",
  "chest pressure": "ضغط بالصدر",
  "chest tightness": "ضيق بالصدر",
  "chills": "رعشة",
  "chronic cough": "كحة مزمنة",
  "confusion": "تشوش",
  "cough": "كحة",
  "deep breathing": "تنفس عميق",
  "dizziness": "دوخة",
  "dry mouth": "جفاف الفم",
  "dyspnea": "ضيق نفس",
  "elevated glucose": "ارتفاع السكر",
  "exertion": "مجهود",
  "facial droop": "ميل بالوجه",
  "fainting": "إغماء",
  "fatigue": "تعب",
  "fever": "حمى",
  "frequent urination": "كثرة التبول",
  "heartburn": "حرقان",
  "hoarseness": "بحة صوت",
  "immobility": "قلة حركة",
  "irregular heartbeat": "عدم انتظام ضربات القلب",
  "jaundice": "اصفرار",
  "leg swelling": "تورم بالساق",
  "myalgia": "آلام عضلية",
  "nasal congestion": "احتقان بالأنف",
  "palpitations": "خفقان",
  "paleness": "شحوب",
  "pleuritic": "يزيد مع التنفس",
  "productive cough": "كحة ببلغم",
  "rash": "طفح",
  "recent surgery": "جراحة حديثة",
  "recent viral illness": "عدوى فيروسية حديثة",
  "reduced intake": "قلة تناول السوائل أو الطعام",
  "shortness of breath": "ضيق نفس",
  "sore throat": "ألم بالحلق",
  "sour taste": "طعم حامضي",
  "sputum": "بلغم",
  "sudden": "مفاجئ",
  "thirst": "عطش",
  "viral prodrome": "أعراض فيروسية سابقة",
  "weakness": "ضعف",
  "weight loss": "فقدان وزن",
  "wheezing": "صفير",
};

function createSessionId() {
  return `session-${Math.random().toString(36).slice(2, 10)}`;
}

function createMessageId(prefix: string) {
  return `${prefix}-${Math.random().toString(36).slice(2, 12)}`;
}

function parseLabsJson(labsJson: string): Record<string, unknown> {
  try {
    return JSON.parse(labsJson) as Record<string, unknown>;
  } catch {
    throw new Error("Invalid JSON in lab values.");
  }
}

function symptomHeuristic(text: string): boolean {
  return /pain|fever|cough|fatigue|dizziness|thirst|nausea|vomit|headache|chest|breath|rash|sore|حمى|ألم|الم|وجع|كحة|سعال|صداع|دوخة|غثيان|صدر|تنفس/i.test(
    text,
  );
}

function normalizeLookupKey(value: string): string {
  return value.trim().toLowerCase().replace(/[._-]+/g, " ").replace(/\s+/g, " ");
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function localizeExact(value: string | undefined, dictionary: Record<string, string>): string | undefined {
  if (!value) return value;
  return dictionary[normalizeLookupKey(value)] ?? dictionary[value.trim()] ?? value;
}

function replaceKnownTerms(value: string, dictionary: Record<string, string>): string {
  return Object.entries(dictionary)
    .sort(([left], [right]) => right.length - left.length)
    .reduce((current, [english, arabic]) => {
      const pattern = new RegExp(`(^|[^\\p{L}\\p{N}])(${escapeRegExp(english)})(?=$|[^\\p{L}\\p{N}])`, "giu");
      return current.replace(pattern, `$1${arabic}`);
    }, value);
}

function localizeMedicalText(value: string | undefined, isArabic: boolean): string {
  if (!value) return "";
  if (!isArabic) return value;

  const exactQuestion = ARABIC_MEDICAL_QUESTIONS[value.trim()];
  if (exactQuestion) return exactQuestion;

  const exactTerm = localizeExact(value, ARABIC_MEDICAL_TERMS);
  if (exactTerm && exactTerm !== value) return exactTerm;

  let translated = value;
  translated = replaceKnownTerms(translated, ARABIC_MEDICAL_PHRASES);
  translated = replaceKnownTerms(translated, ARABIC_MEDICAL_TERMS);
  translated = replaceKnownTerms(translated, ARABIC_EVIDENCE_TERMS);
  return translated;
}

function localizeList(values: string[] | undefined, isArabic: boolean): string {
  return (values ?? [])
    .map((item) => localizeMedicalText(item, isArabic))
    .filter(Boolean)
    .join(isArabic ? "، " : ", ");
}

function getAnalysisLanguage(result: AnalysisResponse): "en" | "ar" {
  const diagnosis = result.diagnosis as
    | (AnalysisResponse["diagnosis"] & { response_language?: string })
    | undefined;
  const responseLanguage =
    diagnosis?.ai_response_metadata?.response_language ??
    diagnosis?.gemini_response_metadata?.response_language ??
    diagnosis?.response_language;

  if (String(responseLanguage ?? "").toLowerCase().startsWith("ar")) return "ar";

  const sourceText = JSON.stringify({
    report: result.report,
    parsed: result.parsed,
    follow_up: result.follow_up,
    response: getResponseText(result),
  });
  return /[\u0600-\u06FF]/.test(sourceText) ? "ar" : "en";
}

function summarizeAnalysis(result: AnalysisResponse): string {
  const isArabic = getAnalysisLanguage(result) === "ar";
  const topDifferential = result.diagnosis?.differential_diagnosis?.[0]?.label;
  const diagnosis =
    result.diagnosis?.final_diagnosis?.diagnosis ??
    topDifferential ??
    (isArabic ? "لا يوجد تشخيص نهائي" : "No final diagnosis");
  const localizedDiagnosis = localizeMedicalText(diagnosis, isArabic);
  const confidence = result.diagnosis?.final_diagnosis?.confidence;
  const confidenceText = confidence === undefined ? "n/a" : String(confidence);

  if (isArabic) {
    return result.diagnosis?.final_diagnosis
      ? `اكتمل التحليل. الحالة المرجحة: ${localizedDiagnosis}. درجة الثقة: ${confidenceText}.`
      : `اكتمل التحليل. لا يوجد تشخيص نهائي بعد. التشخيص التفريقي الأبرز: ${localizedDiagnosis}.`;
  }

  return result.diagnosis?.final_diagnosis
    ? `Analysis completed. Likely condition: ${diagnosis}. Confidence: ${confidenceText}.`
    : `Analysis completed. No final diagnosis yet. Leading differential: ${diagnosis}.`;
}

function serializeAnalysisMessage(analysis: AnalysisResponse): string {
  return `${ANALYSIS_MESSAGE_PREFIX}${JSON.stringify({
    text: summarizeAnalysis(analysis),
    analysis,
  } satisfies StoredAnalysisMessage)}`;
}

function parseStoredAnalysisMessage(content: string): StoredAnalysisMessage | null {
  if (!content.startsWith(ANALYSIS_MESSAGE_PREFIX)) return null;

  try {
    const parsed = JSON.parse(content.slice(ANALYSIS_MESSAGE_PREFIX.length)) as Partial<StoredAnalysisMessage>;
    if (!parsed.analysis || typeof parsed.analysis !== "object") return null;
    return {
      text: typeof parsed.text === "string" && parsed.text.trim() ? parsed.text : summarizeAnalysis(parsed.analysis),
      analysis: parsed.analysis,
    };
  } catch {
    return null;
  }
}

function normalizeForCompare(value: string | undefined): string {
  return (value ?? "").trim().toLowerCase();
}

function getResponseText(analysis: AnalysisResponse): string | undefined {
  return (
    analysis.diagnosis?.ai_response?.trim() ||
    analysis.diagnosis?.gemini_response?.trim() ||
    analysis.diagnosis?.summary?.trim()
  );
}

function AnalysisCard({ analysis }: { analysis: AnalysisResponse }) {
  const isArabic = getAnalysisLanguage(analysis) === "ar";
  const diagnosis = analysis.diagnosis?.final_diagnosis;
  const differential = analysis.diagnosis?.differential_diagnosis ?? [];
  const responseText = getResponseText(analysis);
  const therapy = analysis.therapy?.therapy_plan;
  const safetyReasons = analysis.diagnosis?.safety?.reasons ?? [];
  const clarification = analysis.diagnosis?.clarification;
  const normalizedResponse = normalizeForCompare(responseText);
  const visibleSafetyReasons = safetyReasons.filter((item) => {
    const normalizedItem = normalizeForCompare(item);
    return normalizedItem && !normalizedResponse.includes(normalizedItem);
  });
  const localizedDiagnosis = localizeMedicalText(diagnosis?.diagnosis, isArabic);
  const localizedResponseText = localizeMedicalText(responseText, isArabic);
  const localizedTherapy = localizeMedicalText(therapy, isArabic);

  return (
    <div className="rounded-3xl border border-[var(--brand-border)] bg-[var(--brand-surface)] p-4 text-[var(--brand-text)] shadow-[var(--brand-shadow)]">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-[var(--brand-border)] pb-3">
        <div>
          <p className="text-xs font-semibold uppercase text-[var(--brand-primary)]">Nabda analysis</p>
          <h3 className="mt-1 text-lg font-semibold text-[var(--brand-heading)]">
            {localizedDiagnosis || (isArabic ? "التشخيص التفريقي قيد الانتظار" : "Differential diagnosis pending")}
          </h3>
        </div>
        {diagnosis?.confidence !== undefined ? (
          <span className="rounded-2xl bg-[var(--brand-soft)] px-3 py-1 text-xs font-semibold text-[var(--brand-primary)]">
            {isArabic ? "الثقة" : "Confidence"} {String(diagnosis.confidence)}
          </span>
        ) : null}
      </div>

      {localizedResponseText ? <p className="mt-4 whitespace-pre-wrap text-sm leading-6">{localizedResponseText}</p> : null}
      {differential.length ? (
        <div className="mt-4 rounded-2xl border border-[var(--brand-border)] bg-white/70 p-3">
          <p className="text-xs font-semibold uppercase text-[var(--brand-primary)]">
            {isArabic ? "التشخيص التفريقي" : "Differential diagnosis"}
          </p>
          <div className="mt-2 space-y-2">
            {differential.slice(0, 4).map((item) => (
              <div key={item.label} className="rounded-2xl bg-[var(--brand-soft)] px-3 py-2">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <p className="text-sm font-semibold text-[var(--brand-heading)]">
                    {localizeMedicalText(item.label, isArabic)}
                  </p>
                  <span className="text-xs font-semibold uppercase text-[var(--brand-primary)]">
                    {isArabic ? localizeExact(item.urgency ?? "routine", ARABIC_URGENCY_LABELS) : item.urgency ?? "routine"} | {item.confidence ?? "n/a"}
                  </span>
                </div>
                {item.missing_evidence?.length ? (
                  <p className="mt-1 text-xs leading-5 text-[var(--brand-muted)]">
                    {isArabic ? "البيانات الناقصة: " : "Missing: "}
                    {localizeList(item.missing_evidence.slice(0, 3), isArabic)}
                  </p>
                ) : null}
              </div>
            ))}
          </div>
        </div>
      ) : null}
      {therapy ? (
        <p className="mt-3 rounded-2xl bg-[var(--brand-soft)] px-3 py-2 text-sm leading-6 text-[var(--brand-text)]">
          {isArabic ? "الخطة العلاجية: " : "Therapy: "}
          {localizedTherapy}
        </p>
      ) : null}

      {visibleSafetyReasons.length ? (
        <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 p-3">
          <p className="text-xs font-semibold uppercase text-amber-900">
            {isArabic ? "ملاحظات السلامة" : "Safety notes"}
          </p>
          <ul className="mt-2 space-y-1 text-sm leading-6 text-amber-950">
            {visibleSafetyReasons.map((item) => (
              <li key={item}>{localizeMedicalText(item, isArabic)}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {clarification?.needed ? (
        <div className="mt-4 rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] p-3">
          <p className="text-xs font-semibold uppercase text-[var(--brand-primary)]">
            {isArabic ? "أسئلة متابعة" : "Follow-up questions"}
          </p>
          <ul className="mt-2 space-y-1 text-sm leading-6 text-[var(--brand-text)]">
            {(clarification.questions ?? []).map((item) => (
              <li key={item.question}>
                {localizeMedicalText(item.question_ar ?? item.question, isArabic)}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </div>
  );
}

export function MedicalChat({ compact = false, initialPrompt, userName = "there" }: MedicalChatProps) {
  const { token, logout } = useAuth();
  const { language } = usePreferences();
  const t = getCopy(language);
  const [sessionId] = useState(createSessionId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [chats, setChats] = useState<ChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<number | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [composerText, setComposerText] = useState("");
  const [sendMode, setSendMode] = useState<SendMode>("auto");
  const [actionPanelOpen, setActionPanelOpen] = useState(false);
  const [useParser, setUseParser] = useState(true);
  const [labsJson, setLabsJson] = useState(DEFAULT_LABS_JSON);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [clarificationContext, setClarificationContext] = useState<ClarificationContext | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  useEffect(() => {
    if (!token) return;
    let active = true;

    async function loadChats() {
      setHistoryLoading(true);
      try {
        const items = await fetchChats(token as string);
        if (!active) return;
        setChats(items);
        setActiveChatId((current) => current ?? items[0]?.id ?? null);
      } catch (error) {
        if (!active) return;
        setHistoryError(error instanceof Error ? error.message : "Unable to load chats.");
      } finally {
        if (active) setHistoryLoading(false);
      }
    }

    void loadChats();

    return () => {
      active = false;
    };
  }, [token]);

  useEffect(() => {
    if (!token || !activeChatId) {
      queueMicrotask(() => setMessages([]));
      return;
    }

    let active = true;

    async function loadMessages() {
      setHistoryLoading(true);
      try {
        const items = await fetchChatMessages(token as string, activeChatId as number);
        if (!active) return;
        setMessages(
          items.map((item) => {
            const storedAnalysis = item.role === "assistant" ? parseStoredAnalysisMessage(item.content) : null;
            if (storedAnalysis) {
              return {
                id: `stored-${item.id}`,
                role: item.role,
                kind: "analysis",
                content: storedAnalysis.text,
                payload: storedAnalysis.analysis,
              };
            }

            return {
              id: `stored-${item.id}`,
              role: item.role,
              kind: "text",
              content: item.content,
            };
          }),
        );
      } catch (error) {
        if (!active) return;
        setHistoryError(error instanceof Error ? error.message : "Unable to load chat messages.");
      } finally {
        if (active) setHistoryLoading(false);
      }
    }

    void loadMessages();

    return () => {
      active = false;
    };
  }, [activeChatId, token]);

  const hasAnalysis = useMemo(() => messages.some((item) => item.kind === "analysis"), [messages]);
  const hasStarted = messages.length > 0 || loading;

  const sendDisabled = useMemo(() => {
    if (loading) return true;
    if (sendMode === "image") return !imageFile;
    if (sendMode === "labs") return !labsJson.trim();
    return !composerText.trim();
  }, [composerText, imageFile, labsJson, loading, sendMode]);

  const addMessage = (role: ChatMessage["role"], content: string, kind: ChatMessage["kind"] = "text") => {
    setMessages((current) => [
      ...current,
      {
        id: createMessageId(role),
        role,
        kind,
        content,
      },
    ]);
  };

  const refreshChats = async () => {
    if (!token) return;
    const items = await fetchChats(token);
    setChats(items);
  };

  const ensureChatSession = async () => {
    if (!token) return null;
    if (activeChatId) return activeChatId;
    const chat = await createChat(token);
    setChats((current) => [chat, ...current]);
    return chat.id;
  };

  const persistTurn = async (userContent: string, assistantContent: string) => {
    if (!token || !userContent.trim() || !assistantContent.trim()) return;
    try {
      const chatId = await ensureChatSession();
      if (!chatId) return;
      await saveChatMessage(token, chatId, { role: "user", content: userContent });
      await saveChatMessage(token, chatId, { role: "assistant", content: assistantContent });
      if (!activeChatId) setActiveChatId(chatId);
      await refreshChats();
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Unable to save chat.");
    }
  };

  const addAnalysis = (analysis: AnalysisResponse) => {
    const summary = summarizeAnalysis(analysis);
    const isArabic = getAnalysisLanguage(analysis) === "ar";
    setMessages((current) => [
      ...current,
      {
        id: createMessageId("assistant"),
        role: "assistant",
        kind: "analysis",
        content: summary,
        payload: analysis,
      },
    ]);

    const clarification = analysis.diagnosis?.clarification;
    if (clarification?.needed && analysis.report) {
      setClarificationContext({
        report: analysis.report,
        diagnosis: analysis.diagnosis as Record<string, unknown> | undefined,
        questions: (clarification.questions ?? [])
          .map((item) => localizeMedicalText(item.question_ar ?? item.question, isArabic))
          .filter((item) => item.trim().length > 0),
        language: isArabic ? "ar" : "en",
      });
    } else {
      setClarificationContext(null);
    }

    return summary;
  };

  const setLastAssistantText = (content: string) => {
    setMessages((current) => {
      const next = [...current];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === "assistant" && next[index].kind === "text") {
          next[index] = { ...next[index], content };
          return next;
        }
      }
      return [...next, { id: createMessageId("assistant"), role: "assistant", kind: "text", content }];
    });
  };

  const appendChunkToLastAssistant = (chunk: string) => {
    setMessages((current) => {
      const next = [...current];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === "assistant" && next[index].kind === "text") {
          next[index] = {
            ...next[index],
            content: `${next[index].content ?? ""}${chunk}`,
          };
          return next;
        }
      }
      return [
        ...next,
        { id: createMessageId("assistant"), role: "assistant", kind: "text", content: chunk },
      ];
    });
  };

  const routeAutoAction = (text: string): SendMode => {
    if (clarificationContext) return "symptoms";
    if (imageFile) return "image";
    if (sendMode === "labs") return "labs";
    if (!hasAnalysis || symptomHeuristic(text)) return "symptoms";
    return "chat";
  };

  const chooseMode = (mode: SendMode) => {
    setSendMode(mode);
    setActionPanelOpen(false);
  };

  const runChatTurn = async (text: string) => {
    addMessage("assistant", "");
    let hasStreamedChunk = false;
    let streamedContent = "";

    try {
      const streamedText = await postChatStream({ session_id: sessionId, message: text }, (chunk) => {
        hasStreamedChunk = true;
        streamedContent += chunk;
        appendChunkToLastAssistant(chunk);
      });

      if (!hasStreamedChunk && !streamedText) {
        const fallback = await postChat({ session_id: sessionId, message: text });
        setLastAssistantText(fallback.response);
        return fallback.response;
      }
      return streamedText || streamedContent;
    } catch {
      if (!hasStreamedChunk) {
        try {
          const fallback = await postChat({ session_id: sessionId, message: text });
          setLastAssistantText(fallback.response);
          return fallback.response;
        } catch (error) {
          const content = error instanceof Error ? error.message : String(error);
          setLastAssistantText(`Chat failed: ${content}`);
          return `Chat failed: ${content}`;
        }
      }
      return streamedContent;
    }
  };

  const handleSend = async () => {
    const trimmed = composerText.trim();
    const action = sendMode === "auto" ? routeAutoAction(trimmed) : sendMode;

    if (action !== "image" && action !== "labs" && !trimmed) return;

    let userSummary = trimmed;
    if (!userSummary && action === "image" && imageFile) userSummary = `Uploaded image: ${imageFile.name}`;
    if (!userSummary && action === "labs") userSummary = "Submitted lab values.";

    addMessage("user", userSummary || "Submitted");
    setComposerText("");
    setLoading(true);

    try {
      if (clarificationContext && trimmed && action !== "chat") {
        const answers = trimmed
          .split("\n")
          .map((item) => item.trim())
          .filter(Boolean);
        const analysis = await postClarification({
          report: clarificationContext.report,
          diagnosis: clarificationContext.diagnosis,
          answers: answers.length ? answers : [trimmed],
        });
        addAnalysis(analysis);
        await persistTurn(userSummary || "Submitted", serializeAnalysisMessage(analysis));
      } else if (action === "image" && imageFile) {
        const analysis = await postImage(imageFile);
        addAnalysis(analysis);
        await persistTurn(userSummary || "Submitted", serializeAnalysisMessage(analysis));
        setImageFile(null);
      } else if (action === "labs") {
        const analysis = await postLabs({
          labs: parseLabsJson(labsJson),
          symptoms: trimmed || undefined,
        });
        addAnalysis(analysis);
        await persistTurn(userSummary || "Submitted", serializeAnalysisMessage(analysis));
      } else if (action === "symptoms") {
        const analysis = await postSymptoms({
          text: trimmed,
          use_symptom_parser: useParser,
        });
        addAnalysis(analysis);
        await persistTurn(userSummary || "Submitted", serializeAnalysisMessage(analysis));
      } else {
        const assistantText = await runChatTurn(trimmed);
        await persistTurn(userSummary || "Submitted", assistantText);
      }
    } catch (error) {
      const content = error instanceof Error ? error.message : String(error);
      addMessage("assistant", content, "error");
    } finally {
      setLoading(false);
      setSendMode("auto");
      setActionPanelOpen(false);
    }
  };

  const modeLabel = {
    auto: "Instant",
    symptoms: t.chat.symptoms,
    labs: t.chat.labs,
    image: t.chat.image,
    chat: t.chat.chat,
  }[sendMode];

  const composer = (
    <div className="relative mx-auto w-full max-w-3xl">
      {actionPanelOpen ? (
        <div className="absolute bottom-full left-0 z-10 mb-3 w-full rounded-3xl border border-[var(--brand-border)] bg-[var(--brand-surface)] p-3 shadow-[var(--brand-shadow)]">
          <div className="grid gap-2 sm:grid-cols-4">
            {[
              ["symptoms", t.chat.symptoms],
              ["chat", t.chat.chat],
              ["labs", t.chat.labs],
              ["image", t.chat.image],
            ].map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                className="rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-soft)] px-3 py-2 text-sm font-semibold text-[var(--brand-primary)] transition hover:bg-[var(--brand-surface)] hover:shadow-sm"
                onClick={() => chooseMode(mode as SendMode)}
              >
                {label}
              </button>
            ))}
          </div>
          <label className="mt-3 flex items-center gap-2 text-sm text-[var(--brand-muted)]">
            <input
              type="checkbox"
              checked={useParser}
              onChange={(event) => setUseParser(event.target.checked)}
            />
            {t.chat.parser}
          </label>
        </div>
      ) : null}

      {sendMode === "labs" ? (
        <label className="mb-3 block text-sm font-medium text-[var(--brand-text)]">
          {t.chat.labLabel}
          <textarea
            className="mt-2 min-h-28 w-full resize-y rounded-2xl border border-[var(--brand-border)] bg-[var(--brand-surface)] p-3 font-mono text-sm text-[var(--brand-text)] shadow-sm outline-none transition focus:border-[var(--brand-primary)] focus:ring-4 focus:ring-blue-500/10"
            value={labsJson}
            onChange={(event) => setLabsJson(event.target.value)}
            spellCheck={false}
          />
        </label>
      ) : null}

      {sendMode === "image" ? (
        <label className="mb-3 flex cursor-pointer items-center justify-center rounded-3xl border border-dashed border-[var(--brand-border-strong)] bg-[var(--brand-soft)] px-4 py-5 text-center text-sm font-medium text-[var(--brand-text)] transition hover:border-[var(--brand-primary)]">
          <input
            className="sr-only"
            type="file"
            accept="image/png,image/jpeg,image/webp,image/bmp"
            onChange={(event) => setImageFile(event.target.files?.[0] ?? null)}
          />
          {imageFile ? `Attached: ${imageFile.name}` : t.chat.imageLabel}
        </label>
      ) : null}

      <div className="flex items-center gap-2 rounded-[2rem] border border-[var(--brand-border)] bg-[var(--brand-surface-glass)] px-3 py-2 shadow-[0_18px_60px_rgba(15,73,128,0.14)] backdrop-blur-xl">
        <button
          type="button"
          className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl text-2xl leading-none text-[var(--brand-primary)] transition hover:bg-[var(--brand-soft)]"
          onClick={() => setActionPanelOpen((current) => !current)}
          aria-label={t.chat.attach}
        >
          +
        </button>
        <textarea
          className="max-h-32 min-h-10 flex-1 resize-none bg-transparent px-1 py-2 text-sm text-[var(--brand-text)] outline-none placeholder:text-slate-400"
          value={composerText}
          onChange={(event) => setComposerText(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              void handleSend();
            }
          }}
          placeholder={
            clarificationContext
              ? clarificationContext.language === "ar"
                ? "اكتب إجابات أسئلة المتابعة هنا"
                : "Answer the follow-up questions here"
              : t.chat.placeholder
          }
          disabled={loading}
        />
        <span className="hidden rounded-2xl bg-[var(--brand-soft)] px-3 py-2 text-xs font-semibold text-[var(--brand-primary)] sm:inline-flex">
          {modeLabel}
        </span>
        <button
          type="button"
          className="flex h-10 min-w-10 items-center justify-center rounded-2xl bg-[var(--brand-primary)] px-4 text-sm font-semibold text-white transition hover:bg-[var(--brand-primary-strong)] disabled:cursor-not-allowed disabled:opacity-60"
          disabled={sendDisabled}
          onClick={() => void handleSend()}
        >
          {loading ? t.chat.sending : t.chat.send}
        </button>
      </div>
    </div>
  );

  const startNewChat = () => {
    setActiveChatId(null);
    setMessages([]);
    setClarificationContext(null);
    setComposerText("");
    setHistoryError(null);
  };

  const removeChat = async (chatId: number) => {
    if (!token) return;
    try {
      await deleteChat(token, chatId);
      const nextChats = chats.filter((chat) => chat.id !== chatId);
      setChats(nextChats);
      if (activeChatId === chatId) {
        setActiveChatId(nextChats[0]?.id ?? null);
        if (!nextChats.length) setMessages([]);
      }
    } catch (error) {
      setHistoryError(error instanceof Error ? error.message : "Unable to delete chat.");
    }
  };

  const historySidebar = !compact ? (
    <aside className="flex min-h-0 w-full flex-col border-b border-[var(--brand-border)] bg-[var(--brand-surface)] p-3 md:w-72 md:border-b-0 md:border-r">
      <div className="flex items-center justify-between gap-2">
        <p className="text-sm font-semibold text-[var(--brand-heading)]">Chats</p>
        <button
          type="button"
          className="rounded-2xl bg-[var(--brand-primary)] px-3 py-2 text-xs font-semibold text-white transition hover:bg-[var(--brand-primary-strong)]"
          onClick={startNewChat}
        >
          New Chat
        </button>
      </div>
      {historyError ? (
        <p className="mt-3 rounded-2xl bg-rose-50 px-3 py-2 text-xs font-medium text-rose-700">{historyError}</p>
      ) : null}
      <div className="mt-3 flex-1 space-y-2 overflow-y-auto">
        {historyLoading && !chats.length ? (
          <p className="px-2 py-3 text-sm text-[var(--brand-muted)]">Loading chats...</p>
        ) : null}
        {chats.map((chat) => (
          <div key={chat.id} className="group flex items-center gap-2">
            <button
              type="button"
              className={[
                "min-w-0 flex-1 truncate rounded-2xl px-3 py-2 text-left text-sm font-medium transition",
                activeChatId === chat.id
                  ? "bg-[var(--brand-soft)] text-[var(--brand-primary)]"
                  : "text-[var(--brand-text)] hover:bg-[var(--brand-soft)]",
              ].join(" ")}
              onClick={() => setActiveChatId(chat.id)}
            >
              {chat.title}
            </button>
            <button
              type="button"
              className="rounded-xl px-2 py-1 text-xs font-semibold text-[var(--brand-muted)] transition hover:bg-rose-50 hover:text-rose-700"
              onClick={() => void removeChat(chat.id)}
              aria-label={`Delete ${chat.title}`}
            >
              Delete
            </button>
          </div>
        ))}
        {!historyLoading && !chats.length ? (
          <p className="px-2 py-3 text-sm text-[var(--brand-muted)]">No saved chats yet.</p>
        ) : null}
      </div>
      <button
        type="button"
        className="mt-3 rounded-2xl border border-[var(--brand-border)] px-3 py-2 text-sm font-semibold text-[var(--brand-primary)] transition hover:bg-[var(--brand-soft)]"
        onClick={logout}
      >
        Logout
      </button>
    </aside>
  ) : null;

  return (
    <section
      className={[
        "flex h-full min-h-0 overflow-hidden bg-[var(--brand-bg)]",
        compact ? "flex-col" : "flex-col md:flex-row",
        compact ? "max-h-[78vh] rounded-3xl border border-[var(--brand-border)]" : "min-h-[calc(100svh-64px)]",
      ].join(" ")}
    >
      {historySidebar}
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
      {!hasStarted ? (
        <div className="flex flex-1 flex-col items-center justify-end px-4 pb-10 pt-16 text-center sm:pb-14">
          <div className="mb-8 max-w-3xl">
            <p className="text-3xl font-semibold text-[var(--brand-heading)] sm:text-5xl">
              {initialPrompt ?? `Hey, ${userName} 👋`}
            </p>
            <p className="mt-4 text-2xl font-semibold text-[var(--brand-heading)] sm:text-4xl">
              How can Nabda help you today?
            </p>
          </div>
          {composer}
          <div className="mt-5 flex flex-wrap justify-center gap-2">
            {[t.chat.symptoms, t.chat.image, t.chat.labs].map((label) => (
              <button
                key={label}
                type="button"
                className="rounded-full border border-[var(--brand-border)] bg-[var(--brand-surface)] px-4 py-2 text-sm font-medium text-[var(--brand-text)] shadow-sm transition hover:bg-[var(--brand-soft)]"
                onClick={() => {
                  const nextMode =
                    label === t.chat.image ? "image" : label === t.chat.labs ? "labs" : "symptoms";
                  chooseMode(nextMode);
                }}
              >
                {label}
              </button>
            ))}
            <Link
              href="/doctors"
              className="rounded-full border border-[var(--brand-border)] bg-[var(--brand-surface)] px-4 py-2 text-sm font-medium text-[var(--brand-text)] shadow-sm transition hover:bg-[var(--brand-soft)]"
            >
              Find a doctor
            </Link>
          </div>
          <p className="mt-5 text-xs leading-5 text-[var(--brand-muted)]">{t.chat.disclaimer}</p>
        </div>
      ) : (
        <>
          <div className="flex-1 overflow-y-auto px-4 py-6">
            <div className="mx-auto flex max-w-4xl flex-col gap-4">
              {messages.map((message) => (
                <article
                  key={message.id}
                  className={[
                    "max-w-[88%] rounded-2xl px-4 py-3 text-sm leading-6 shadow-sm",
                    message.role === "user"
                      ? "ml-auto bg-[var(--brand-primary)] text-white"
                    : message.kind === "error"
                        ? "border border-rose-200 bg-rose-50 text-rose-900"
                        : "mr-auto bg-[var(--brand-surface)] text-[var(--brand-text)]",
                    message.kind === "analysis" ? "w-full max-w-full bg-transparent p-0 shadow-none" : "",
                  ].join(" ")}
                >
                  {message.kind === "analysis" && message.payload ? (
                    <AnalysisCard analysis={message.payload} />
                  ) : (
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  )}
                </article>
              ))}
              {loading ? (
                <div className="mr-auto rounded-2xl bg-[var(--brand-surface)] px-4 py-2 text-sm text-[var(--brand-muted)] shadow-sm">
                  {t.chat.working}
                </div>
              ) : null}
              <div ref={endRef} />
            </div>
          </div>

          {clarificationContext?.questions.length ? (
            <div className="border-t border-[var(--brand-border)] bg-[var(--brand-soft)] px-5 py-3 text-sm text-[var(--brand-text)]">
              <p className="font-semibold">{clarificationContext.language === "ar" ? "متابعة" : "Follow-up"}</p>
              <ul className="mt-2 space-y-1">
                {clarificationContext.questions.map((question) => (
                  <li key={question}>{question}</li>
                ))}
              </ul>
            </div>
          ) : null}

          <div className="border-t border-[var(--brand-border)] bg-[var(--brand-surface-glass)] p-4 backdrop-blur">
            {composer}
            <p className="mx-auto mt-3 max-w-3xl text-xs leading-5 text-[var(--brand-muted)]">
              {t.chat.disclaimer}
            </p>
          </div>
        </>
      )}
      </div>
    </section>
  );
}
