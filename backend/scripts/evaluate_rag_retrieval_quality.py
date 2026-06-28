"""Evaluate FAISS-backed RAG retrieval quality on built-in smoke cases.

This script intentionally avoids LLM calls and does not print secrets from
``backend/.env``. It reads the same RAG asset paths used by the backend settings,
loads the local ClinicalBERT embedder and FAISS metadata, writes a data coverage
report, then evaluates a small set of known clinical retrieval cases.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from models.diagnosis.rag import ClinicalBERTEmbedder, MedicalCaseSearcher, MedicalRAGAssistant
from models.diagnosis.text import build_combined_text


DEFAULT_OUTPUT_DIR = Path("data/evaluation/rag_diagnostics")
DIABETES_TERMS = ("diabetes", "hyperglycemia", "diabetic", "prediabetes")
DEFAULT_THRESHOLDS = {
    "top_5_accuracy": 0.85,
    "mrr": 0.70,
    "out_of_scope_low_confidence_rate": 0.90,
}


@dataclass(frozen=True)
class SmokeCase:
    case_id: str
    clinical_text: str
    symptoms: list[str]
    labs: dict[str, Any]
    expected_family: str
    acceptable_aliases: tuple[str, ...]
    scope: str = "in_scope"
    age: int | None = None
    sex: str | None = None


SMOKE_CASES = [
    SmokeCase(
        case_id="stable_angina",
        clinical_text=(
            "Exertional chest pressure radiating to the left arm, improves with rest, "
            "with cardiovascular risk factors."
        ),
        symptoms=["chest pain", "chest pressure", "exertion"],
        labs={},
        expected_family="stable angina",
        acceptable_aliases=("stable angina",),
        age=62,
        sex="M",
    ),
    SmokeCase(
        case_id="asthma_bronchospasm",
        clinical_text=(
            "Shortness of breath with wheezing and chest tightness, worse after dust exposure, "
            "without fever or productive cough."
        ),
        symptoms=["shortness of breath", "wheezing", "chest tightness"],
        labs={},
        expected_family="asthma/bronchospasm",
        acceptable_aliases=("bronchospasm", "asthma", "acute asthma exacerbation"),
        age=24,
        sex="F",
    ),
    SmokeCase(
        case_id="pneumonia_fever_cough",
        clinical_text=(
            "Fever with productive cough, chills, pleuritic chest discomfort, and shortness of breath."
        ),
        symptoms=["fever", "productive cough", "chills", "shortness of breath"],
        labs={},
        expected_family="pneumonia",
        acceptable_aliases=("pneumonia",),
        age=45,
        sex="M",
    ),
    SmokeCase(
        case_id="urti_sore_throat_rhinitis",
        clinical_text=(
            "Sore throat with mild fever, painful swallowing, runny nose, and no shortness of breath."
        ),
        symptoms=["sore throat", "fever", "painful swallowing", "runny nose"],
        labs={},
        expected_family="upper respiratory tract infection",
        acceptable_aliases=("urti", "upper respiratory tract infection", "viral pharyngitis"),
        age=19,
        sex="F",
    ),
    SmokeCase(
        case_id="cluster_headache",
        clinical_text=(
            "Severe unilateral pain around one eye with tearing, nasal congestion, and recurrent short attacks."
        ),
        symptoms=["headache", "eye pain", "tearing", "nasal congestion"],
        labs={},
        expected_family="cluster headache",
        acceptable_aliases=("cluster headache",),
        age=34,
        sex="M",
    ),
    SmokeCase(
        case_id="diabetes_hyperglycemia_oos",
        clinical_text=(
            "Patient has two weeks of fatigue, increased thirst, frequent urination, "
            "mild weight loss, and fasting glucose is elevated."
        ),
        symptoms=["fatigue", "thirst", "polyuria", "weight loss"],
        labs={"glucose": {"value": 230, "unit": "mg/dL"}},
        expected_family="diabetes/hyperglycemia",
        acceptable_aliases=("diabetes", "hyperglycemia", "prediabetes"),
        scope="out_of_scope",
        age=50,
        sex="F",
    ),
]


EXPANDED_CASES = [
    *SMOKE_CASES[:-1],
    SmokeCase(
        case_id="copd_exacerbation_sputum",
        clinical_text="Known COPD and cigarette smoking with worse shortness of breath and more colored sputum than usual.",
        symptoms=["shortness of breath", "cough", "productive cough"],
        labs={},
        expected_family="acute copd exacerbation",
        acceptable_aliases=("acute copd exacerbation", "copd"),
        age=68,
        sex="M",
    ),
    SmokeCase(
        case_id="bronchitis_post_cold_cough",
        clinical_text="Persistent cough and burning chest discomfort after a recent cold, mild sputum, no leg swelling.",
        symptoms=["cough", "chest discomfort", "recent cold"],
        labs={},
        expected_family="bronchitis",
        acceptable_aliases=("bronchitis",),
        age=34,
        sex="F",
    ),
    SmokeCase(
        case_id="bronchiolitis_infant_wheeze",
        clinical_text="Infant with nasal congestion, cough, fever, and significant difficulty breathing with brief pauses during sleep.",
        symptoms=["nasal congestion", "cough", "fever", "shortness of breath"],
        labs={},
        expected_family="bronchiolitis",
        acceptable_aliases=("bronchiolitis",),
        age=1,
        sex="M",
    ),
    SmokeCase(
        case_id="bronchiectasis_cf_sputum",
        clinical_text="History of cystic fibrosis with chronic productive cough, abundant colored sputum, and recurrent pneumonia.",
        symptoms=["cough", "productive cough", "shortness of breath"],
        labs={},
        expected_family="bronchiectasis",
        acceptable_aliases=("bronchiectasis",),
        age=22,
        sex="F",
    ),
    SmokeCase(
        case_id="pulmonary_embolism_sudden_pleuritic",
        clinical_text="Sudden shortness of breath with pleuritic chest pain after prolonged immobility and unilateral leg swelling.",
        symptoms=["shortness of breath", "chest pain", "leg swelling"],
        labs={},
        expected_family="pulmonary embolism",
        acceptable_aliases=("pulmonary embolism",),
        age=45,
        sex="F",
    ),
    SmokeCase(
        case_id="spontaneous_pneumothorax_unilateral",
        clinical_text="Abrupt one-sided sharp chest pain at rest with acute shortness of breath, no fever or productive cough.",
        symptoms=["chest pain", "shortness of breath", "one sided"],
        labs={},
        expected_family="spontaneous pneumothorax",
        acceptable_aliases=("spontaneous pneumothorax", "pneumothorax"),
        age=21,
        sex="M",
    ),
    SmokeCase(
        case_id="pulmonary_edema_orthopnea_sweating",
        clinical_text="Severe breathlessness worse lying flat with sweating, ankle swelling, and prior fluid in the lungs.",
        symptoms=["shortness of breath", "sweating", "ankle swelling"],
        labs={},
        expected_family="acute pulmonary edema",
        acceptable_aliases=("acute pulmonary edema", "pulmonary edema"),
        age=63,
        sex="M",
    ),
    SmokeCase(
        case_id="pulmonary_neoplasm_chronic_weight_loss",
        clinical_text="Chronic cough in a smoker with coughing blood, progressive shortness of breath, and involuntary weight loss.",
        symptoms=["cough", "hemoptysis", "shortness of breath", "weight loss"],
        labs={},
        expected_family="pulmonary neoplasm",
        acceptable_aliases=("pulmonary neoplasm", "lung cancer", "neoplasm"),
        age=70,
        sex="M",
    ),
    SmokeCase(
        case_id="tuberculosis_weight_loss_hemoptysis",
        clinical_text="Chronic cough with coughing blood, night sweats, low BMI, and involuntary weight loss over three months.",
        symptoms=["cough", "hemoptysis", "weight loss", "night sweats"],
        labs={},
        expected_family="tuberculosis",
        acceptable_aliases=("tuberculosis",),
        age=31,
        sex="M",
    ),
    SmokeCase(
        case_id="sarcoidosis_dry_cough_nodes",
        clinical_text="Gradual dry cough with shortness of breath, painful swollen lymph nodes, and joint pains in the fingers.",
        symptoms=["dry cough", "shortness of breath", "swollen lymph nodes"],
        labs={},
        expected_family="sarcoidosis",
        acceptable_aliases=("sarcoidosis",),
        age=29,
        sex="F",
    ),
    SmokeCase(
        case_id="viral_pharyngitis_daycare_sore_throat",
        clinical_text="Sore throat after daycare exposure with burning tonsil pain, mild fever, and contact with similar symptoms.",
        symptoms=["sore throat", "fever", "tonsil pain"],
        labs={},
        expected_family="viral pharyngitis",
        acceptable_aliases=("viral pharyngitis",),
        age=26,
        sex="F",
    ),
    SmokeCase(
        case_id="acute_laryngitis_hoarse_voice",
        clinical_text="Hoarse voice and throat pain after recent upper respiratory infection, worse with voice use.",
        symptoms=["hoarseness", "sore throat", "recent cold"],
        labs={},
        expected_family="acute laryngitis",
        acceptable_aliases=("acute laryngitis", "laryngitis"),
        age=31,
        sex="F",
    ),
    SmokeCase(
        case_id="acute_otitis_media_child_ear_pain",
        clinical_text="Young child with sharp ear pain, recent oral antibiotic for ear infection, fever, and pulling at the ear.",
        symptoms=["ear pain", "fever"],
        labs={},
        expected_family="acute otitis media",
        acceptable_aliases=("acute otitis media", "otitis media", "ear infection"),
        age=4,
        sex="F",
    ),
    SmokeCase(
        case_id="acute_rhinosinusitis_facial_pain",
        clinical_text="Acute forehead and cheek pain with nasal congestion, thick nasal discharge, and symptoms for one week.",
        symptoms=["nasal congestion", "facial pain", "forehead pain"],
        labs={},
        expected_family="acute rhinosinusitis",
        acceptable_aliases=("acute rhinosinusitis", "rhinosinusitis", "sinusitis"),
        age=42,
        sex="F",
    ),
    SmokeCase(
        case_id="chronic_rhinosinusitis_long_duration",
        clinical_text="Months of recurrent nasal congestion with cheek and forehead pressure, reduced smell, and chronic sinus symptoms.",
        symptoms=["nasal congestion", "facial pain", "chronic"],
        labs={},
        expected_family="chronic rhinosinusitis",
        acceptable_aliases=("chronic rhinosinusitis",),
        age=38,
        sex="M",
    ),
    SmokeCase(
        case_id="allergic_sinusitis_itchy_eyes",
        clinical_text="Itchy nose and throat with severe itchy eyes, hay fever history, and dry cough after pollen exposure.",
        symptoms=["nasal congestion", "itchy eyes", "cough"],
        labs={},
        expected_family="allergic sinusitis",
        acceptable_aliases=("allergic sinusitis", "allergic rhinitis", "hay fever"),
        age=23,
        sex="F",
    ),
    SmokeCase(
        case_id="croup_barking_stridor_child",
        clinical_text="Toddler with nasal congestion, barking whooping cough, high-pitched sound when breathing in, and drooling.",
        symptoms=["cough", "stridor", "nasal congestion"],
        labs={},
        expected_family="croup",
        acceptable_aliases=("croup",),
        age=2,
        sex="M",
    ),
    SmokeCase(
        case_id="epiglottitis_severe_throat_drooling",
        clinical_text="Severe throat and neck pain with drooling, high fever, difficulty swallowing, and acute distress.",
        symptoms=["sore throat", "drooling", "difficulty swallowing", "fever"],
        labs={},
        expected_family="epiglottitis",
        acceptable_aliases=("epiglottitis",),
        age=18,
        sex="M",
    ),
    SmokeCase(
        case_id="whooping_cough_post_tussive_vomit",
        clinical_text="Intense coughing fits after pertussis exposure with vomiting after coughing and no travel.",
        symptoms=["cough", "vomiting"],
        labs={},
        expected_family="whooping cough",
        acceptable_aliases=("whooping cough", "pertussis"),
        age=12,
        sex="F",
    ),
    SmokeCase(
        case_id="influenza_fever_myalgias",
        clinical_text="Abrupt fever with sweating, exhausting body aches, headache, cough, and recent contact with similar symptoms.",
        symptoms=["fever", "sweating", "headache", "cough"],
        labs={},
        expected_family="influenza",
        acceptable_aliases=("influenza", "flu"),
        age=35,
        sex="M",
    ),
    SmokeCase(
        case_id="hiv_initial_lymph_nodes",
        clinical_text="Fever, sore throat, swollen painful lymph nodes, diarrhea, night sweats, and recent STI risk.",
        symptoms=["fever", "sore throat", "swollen lymph nodes", "diarrhea"],
        labs={},
        expected_family="hiv initial infection",
        acceptable_aliases=("hiv", "initial infection"),
        age=28,
        sex="M",
    ),
    SmokeCase(
        case_id="ebola_travel_contact_diarrhea",
        clinical_text="Recent West Africa travel and Ebola contact with fever, confusion, sore throat, cough, diarrhea, and fatigue.",
        symptoms=["fever", "confusion", "sore throat", "diarrhea", "fatigue"],
        labs={},
        expected_family="ebola",
        acceptable_aliases=("ebola",),
        age=40,
        sex="F",
    ),
    SmokeCase(
        case_id="anaphylaxis_food_allergy",
        clinical_text="Known severe food allergy with sudden flushing, breathing difficulty, dizziness, and rash after eating allergen.",
        symptoms=["shortness of breath", "dizziness", "rash"],
        labs={},
        expected_family="anaphylaxis",
        acceptable_aliases=("anaphylaxis",),
        age=26,
        sex="F",
    ),
    SmokeCase(
        case_id="scombroid_fish_flushing",
        clinical_text="Soon after eating fish, cheeks turned red with flushing, dizziness, diarrhea, rash, and mild breathing difficulty.",
        symptoms=["flushing", "diarrhea", "rash", "shortness of breath"],
        labs={},
        expected_family="scombroid food poisoning",
        acceptable_aliases=("scombroid", "food poisoning"),
        age=30,
        sex="M",
    ),
    SmokeCase(
        case_id="atrial_fibrillation_irregular",
        clinical_text="Irregular heartbeat with palpitations, dizziness, shortness of breath, high blood pressure, and hyperthyroidism history.",
        symptoms=["palpitations", "dizziness", "shortness of breath"],
        labs={},
        expected_family="atrial fibrillation",
        acceptable_aliases=("atrial fibrillation",),
        age=67,
        sex="F",
    ),
    SmokeCase(
        case_id="psvt_sudden_rapid_attacks",
        clinical_text="Sudden rapid heartbeat episodes that start and stop abruptly after caffeine and energy drinks, with anxiety.",
        symptoms=["palpitations", "anxiety"],
        labs={},
        expected_family="psvt",
        acceptable_aliases=("psvt",),
        age=28,
        sex="F",
    ),
    SmokeCase(
        case_id="unstable_angina_rest_worse",
        clinical_text="Chest pain at rest with sweating and nausea, worsening over two weeks with less effort needed to trigger pain.",
        symptoms=["chest pain", "sweating", "nausea"],
        labs={},
        expected_family="unstable angina",
        acceptable_aliases=("unstable angina",),
        age=58,
        sex="M",
    ),
    SmokeCase(
        case_id="possible_stemi_heavy_chest",
        clinical_text="Severe heavy scary upper chest pain radiating to the left arm with sweating and nausea.",
        symptoms=["chest pain", "sweating", "nausea"],
        labs={},
        expected_family="possible nstemi stemi",
        acceptable_aliases=("possible nstemi", "stemi", "nstemi"),
        age=61,
        sex="M",
    ),
    SmokeCase(
        case_id="pericarditis_positional_pleuritic",
        clinical_text="Sharp pleuritic chest pain radiating to the back, worse lying down and better sitting forward after viral illness.",
        symptoms=["chest pain", "pleuritic", "viral infection"],
        labs={},
        expected_family="pericarditis",
        acceptable_aliases=("pericarditis",),
        age=42,
        sex="M",
    ),
    SmokeCase(
        case_id="myocarditis_viral_chest_pain",
        clinical_text="Recent viral infection followed by chest pain, shortness of breath, fatigue, and palpitations.",
        symptoms=["viral infection", "chest pain", "shortness of breath", "fatigue"],
        labs={},
        expected_family="myocarditis",
        acceptable_aliases=("myocarditis",),
        age=24,
        sex="M",
    ),
    SmokeCase(
        case_id="gerd_burning_after_meals",
        clinical_text="Burning lower chest and epigastric discomfort after meals, sour taste, reflux, worse lying down.",
        symptoms=["reflux", "chest discomfort", "abdominal pain"],
        labs={},
        expected_family="gerd",
        acceptable_aliases=("gerd", "gastroesophageal reflux"),
        age=37,
        sex="M",
    ),
    SmokeCase(
        case_id="boerhaave_after_vomiting",
        clinical_text="Violent chest and upper abdominal pain after repeated forceful vomiting, feels like a tearing knife stroke.",
        symptoms=["chest pain", "vomiting", "abdominal pain"],
        labs={},
        expected_family="boerhaave",
        acceptable_aliases=("boerhaave",),
        age=52,
        sex="M",
    ),
    SmokeCase(
        case_id="pancreatic_neoplasm_weight_loss",
        clinical_text="Progressive epigastric pain radiating to the back with poor appetite, chronic pancreatitis, diarrhea, and weight loss.",
        symptoms=["abdominal pain", "diarrhea", "weight loss"],
        labs={},
        expected_family="pancreatic neoplasm",
        acceptable_aliases=("pancreatic neoplasm",),
        age=64,
        sex="F",
    ),
    SmokeCase(
        case_id="inguinal_hernia_testicular_groin",
        clinical_text="Groin and testicular pain with abdominal bloating and a bulge that is worse with lifting.",
        symptoms=["abdominal pain", "groin pain", "testicular pain"],
        labs={},
        expected_family="inguinal hernia",
        acceptable_aliases=("inguinal hernia",),
        age=36,
        sex="M",
    ),
    SmokeCase(
        case_id="guillain_barre_ascending_weakness",
        clinical_text="Recent viral infection followed by ascending weakness in both legs and arms with tingling and numbness.",
        symptoms=["weakness", "tingling", "viral infection"],
        labs={},
        expected_family="guillain-barre syndrome",
        acceptable_aliases=("guillain", "guillain barre", "guillain-barr"),
        age=43,
        sex="M",
    ),
    SmokeCase(
        case_id="myasthenia_ptosis_diplopia",
        clinical_text="Drooping eyelids, double vision, difficulty speaking and swallowing, fatigable weakness worse later in the day.",
        symptoms=["weakness", "double vision", "difficulty swallowing"],
        labs={},
        expected_family="myasthenia gravis",
        acceptable_aliases=("myasthenia gravis",),
        age=65,
        sex="F",
    ),
    SmokeCase(
        case_id="acute_dystonic_antipsychotic",
        clinical_text="Started antipsychotic medication within seven days and now has neck spasm, jaw stiffness, and trouble opening eyelids.",
        symptoms=["neck spasm", "jaw stiffness", "eyelid problem"],
        labs={},
        expected_family="acute dystonic reaction",
        acceptable_aliases=("acute dystonic", "dystonic reaction"),
        age=19,
        sex="M",
    ),
    SmokeCase(
        case_id="anemia_fatigue_pallor",
        clinical_text="Poor diet with known anemia, unusual fatigue, dizziness, pale appearance, and shortness of breath on exertion.",
        symptoms=["fatigue", "dizziness", "shortness of breath"],
        labs={"hemoglobin": {"value": 8.5, "unit": "g/dL"}},
        expected_family="anemia",
        acceptable_aliases=("anemia",),
        age=30,
        sex="F",
    ),
    SmokeCase(
        case_id="sLE_joint_pericarditis",
        clinical_text="History of pericarditis with wrist and shoulder joint pain, photosensitive rash, fatigue, and autoimmune symptoms.",
        symptoms=["joint pain", "rash", "fatigue"],
        labs={},
        expected_family="sle",
        acceptable_aliases=("sle", "systemic lupus"),
        age=35,
        sex="F",
    ),
    SmokeCase(
        case_id="diabetes_hyperglycemia_oos",
        clinical_text=(
            "Patient has two weeks of fatigue, increased thirst, frequent urination, "
            "mild weight loss, and fasting glucose is elevated."
        ),
        symptoms=["fatigue", "thirst", "polyuria", "weight loss"],
        labs={"glucose": {"value": 230, "unit": "mg/dL"}},
        expected_family="diabetes/hyperglycemia",
        acceptable_aliases=("diabetes", "hyperglycemia", "prediabetes"),
        scope="out_of_scope",
        age=50,
        sex="F",
    ),
    SmokeCase(
        case_id="uti_cystitis_oos",
        clinical_text="Burning urination with urinary frequency, urgency, suprapubic discomfort, and no flank pain.",
        symptoms=["dysuria", "urinary frequency", "urgency", "suprapubic pain"],
        labs={},
        expected_family="urinary tract infection",
        acceptable_aliases=("urinary tract infection", "uti", "cystitis"),
        scope="out_of_scope",
        age=29,
        sex="F",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval smoke quality.")
    parser.add_argument("--faiss-index-dir", type=Path, default=None)
    parser.add_argument("--clinicalbert-model-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--case-set", choices=("smoke", "expanded"), default="smoke")
    parser.add_argument("--top5-threshold", type=float, default=DEFAULT_THRESHOLDS["top_5_accuracy"])
    parser.add_argument("--mrr-threshold", type=float, default=DEFAULT_THRESHOLDS["mrr"])
    parser.add_argument(
        "--out-of-scope-threshold",
        type=float,
        default=DEFAULT_THRESHOLDS["out_of_scope_low_confidence_rate"],
    )
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit non-zero if configured metric thresholds are not met.",
    )
    parser.add_argument(
        "--run-label",
        default="current",
        help="Optional label copied into the output summary, e.g. baseline or improved.",
    )
    return parser.parse_args()


def select_cases(case_set: str) -> list[SmokeCase]:
    if case_set == "expanded":
        return EXPANDED_CASES
    return SMOKE_CASES


def output_prefix_for_case_set(case_set: str) -> str:
    return "expanded_retrieval_eval" if case_set == "expanded" else "retrieval_eval"


def normalize_label(value: Any) -> str:
    return " ".join(str(value or "").lower().replace("-", " ").replace("_", " ").split())


def label_matches(pathology: str, aliases: tuple[str, ...]) -> bool:
    normalized = normalize_label(pathology)
    for alias in aliases:
        normalized_alias = normalize_label(alias)
        if normalized_alias == "stable angina":
            if normalized == normalized_alias:
                return True
            continue
        if normalized_alias == normalized or normalized_alias in normalized:
            return True
    return False


def build_report_for_case(case: SmokeCase) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    if case.age is not None and case.sex:
        fields["sex_age"] = f"{case.age} year old {case.sex}"
    return {
        "raw_text": case.clinical_text,
        "symptoms": case.symptoms,
        "labs": case.labs,
        "fields": fields,
    }


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(score) or math.isinf(score):
        return default
    return score


def disease_distribution(metadata: dict[str, Any]) -> Counter[str]:
    return Counter(str(item).strip() or "Unknown" for item in metadata.get("pathologies", []) or [])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def copy_labeled_output(path: Path, run_label: str) -> None:
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_label)
    if not safe_label or safe_label == "current":
        return
    labeled = path.with_name(f"{path.stem}_{safe_label}{path.suffix}")
    labeled.write_bytes(path.read_bytes())


def write_coverage_report(
    searcher: MedicalCaseSearcher,
    output_dir: Path,
    faiss_dir: Path,
    model_dir: Path,
    cases: list[SmokeCase],
) -> dict[str, Any]:
    metadata = searcher.metadata
    distribution = disease_distribution(metadata)
    total_cases = len(metadata.get("patient_ids", []) or [])
    unique_pathologies = len(distribution)
    top_distribution = distribution.most_common()
    diabetes_distribution = {
        label: count
        for label, count in top_distribution
        if any(term in normalize_label(label) for term in DIABETES_TERMS)
    }
    max_count = max(distribution.values(), default=0)
    min_count = min(distribution.values(), default=0)
    median_count = 0
    if distribution:
        counts = sorted(distribution.values())
        median_count = counts[len(counts) // 2]

    report = {
        "faiss_index_dir": str(faiss_dir),
        "clinicalbert_model_dir": str(model_dir),
        "index_vectors": int(searcher.index.ntotal),
        "metadata_cases": total_cases,
        "metadata_has_natural_text": searcher.metadata_has_natural_text(),
        "unique_pathologies": unique_pathologies,
        "distribution_summary": {
            "max_count": max_count,
            "median_count": median_count,
            "min_count": min_count,
            "max_to_median_ratio": round(max_count / median_count, 3) if median_count else None,
        },
        "diabetes_hyperglycemia_related": diabetes_distribution,
        "smoke_case_label_coverage": {
            case.case_id: {
                "scope": case.scope,
                "expected_family": case.expected_family,
                "matching_labels": {
                    label: count
                    for label, count in top_distribution
                    if label_matches(label, case.acceptable_aliases)
                },
            }
            for case in cases
        },
        "disease_distribution": [
            {"pathology": label, "count": count}
            for label, count in top_distribution
        ],
        "assessment": {
            "small_dataset": total_cases < 5000,
            "imbalanced": bool(median_count and max_count / median_count >= 5),
            "diabetes_missing": not bool(diabetes_distribution),
        },
    }
    write_json(output_dir / "data_coverage_report.json", report)

    lines = [
        "# RAG Data Coverage Report",
        "",
        f"- FAISS index dir: `{faiss_dir}`",
        f"- ClinicalBERT model dir: `{model_dir}`",
        f"- Index vectors: {searcher.index.ntotal}",
        f"- Metadata cases: {total_cases}",
        f"- Unique pathologies: {unique_pathologies}",
        f"- Metadata has natural text: {report['metadata_has_natural_text']}",
        f"- Dataset considered small: {report['assessment']['small_dataset']}",
        f"- Dataset considered imbalanced: {report['assessment']['imbalanced']}",
        f"- Diabetes/hyperglycemia labels missing: {report['assessment']['diabetes_missing']}",
        "",
        "## Diabetes / Hyperglycemia Coverage",
    ]
    if diabetes_distribution:
        lines.extend(f"- {label}: {count}" for label, count in diabetes_distribution.items())
    else:
        lines.append("- No diabetes/hyperglycemia-related labels found.")
    lines.extend(["", "## Disease Distribution"])
    lines.extend(f"- {label}: {count}" for label, count in top_distribution)
    (output_dir / "data_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def case_failure_reason(case: SmokeCase, matches: list[int], coverage: dict[str, Any], retrieved: list[dict[str, Any]]) -> str:
    if matches:
        return "passed"
    case_coverage = coverage.get("smoke_case_label_coverage", {}).get(case.case_id, {})
    if not case_coverage.get("matching_labels"):
        return f"missing data: no searchable metadata label for {case.expected_family}"
    expected_terms = {normalize_label(alias) for alias in case.acceptable_aliases}
    retrieved_labels = {normalize_label(item.get("pathology")) for item in retrieved}
    if expected_terms.intersection(retrieved_labels):
        return "label mismatch / alias normalization issue"
    top = retrieved[0] if retrieved else {}
    top_overlap = safe_float(top.get("symptom_overlap"))
    top_lexical = safe_float(top.get("lexical_overlap"))
    top_rerank = safe_float(top.get("rerank_score"), safe_float(top.get("similarity")))
    if top_overlap < 0.20 and top_lexical < 0.08:
        return "query construction or lexical/symptom alignment issue"
    if top_rerank < 0.25:
        return "embedding similarity or reranking score too weak"
    return "embedding/reranking issue: expected disease not retrieved in top-k despite non-trivial score"


async def evaluate_cases(
    searcher: MedicalCaseSearcher,
    embedder: ClinicalBERTEmbedder,
    output_dir: Path,
    coverage: dict[str, Any],
    cases: list[SmokeCase],
    top_k: int,
    run_label: str,
    output_prefix: str = "retrieval_eval",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    reciprocal_ranks: list[float] = []
    in_scope_cases = [case for case in cases if case.scope == "in_scope"]
    out_of_scope_cases = [case for case in cases if case.scope != "in_scope"]
    out_of_scope_low_confidence = 0

    for case in cases:
        report = build_report_for_case(case)
        query_text = build_combined_text(report)
        embedding = embedder.encode_text(query_text)
        retrieved = searcher.search(
            embedding,
            k=top_k,
            query_text=query_text,
            query_symptoms=case.symptoms,
        )
        detected_out_of_scope_signals = MedicalCaseSearcher.detect_out_of_scope_signals(
            query_text,
            case.symptoms,
        )
        rag_confidence = MedicalRAGAssistant._build_confidence_metadata(
            retrieved,
            detected_out_of_scope_signals=detected_out_of_scope_signals,
        )
        ranks = [
            idx + 1
            for idx, item in enumerate(retrieved)
            if label_matches(str(item.get("pathology", "")), case.acceptable_aliases)
        ]
        rank = ranks[0] if ranks else None
        if case.scope == "in_scope":
            top1_hits += int(rank == 1)
            top3_hits += int(rank is not None and rank <= 3)
            top5_hits += int(rank is not None and rank <= 5)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            failure_reason = case_failure_reason(case, ranks, coverage, retrieved)
        else:
            out_of_scope_low_confidence += int(not rag_confidence.get("usable_for_fusion", True))
            failure_reason = (
                "out_of_scope safety case; expected disease is absent from DDXPlus metadata; "
                f"rag confidence={rag_confidence.get('level')}"
            )

        rows.append(
            {
                "run_label": run_label,
                "case_id": case.case_id,
                "scope": case.scope,
                "expected_family": case.expected_family,
                "acceptable_aliases": "; ".join(case.acceptable_aliases),
                "query_text": query_text,
                "retrieved_pathologies": "; ".join(str(item.get("pathology", "")) for item in retrieved),
                "retrieval_scores": "; ".join(f"{safe_float(item.get('similarity')):.4f}" for item in retrieved),
                "rerank_scores": "; ".join(f"{safe_float(item.get('rerank_score'), safe_float(item.get('similarity'))):.4f}" for item in retrieved),
                "symptom_overlap_scores": "; ".join(f"{safe_float(item.get('symptom_overlap')):.4f}" for item in retrieved),
                "lexical_overlap_scores": "; ".join(f"{safe_float(item.get('lexical_overlap')):.4f}" for item in retrieved),
                "rank": rank or "",
                "top1_hit": rank == 1,
                "top3_hit": rank is not None and rank <= 3,
                "top5_hit": rank is not None and rank <= 5,
                "expected_anywhere_in_top_k": bool(rank),
                "rag_confidence_level": rag_confidence.get("level", ""),
                "rag_usable_for_fusion": rag_confidence.get("usable_for_fusion", ""),
                "rag_top_rerank_score": rag_confidence.get("top_rerank_score", ""),
                "rag_explicit_signal": rag_confidence.get("explicit_signal", ""),
                "rag_max_penalty": rag_confidence.get("max_penalty", ""),
                "rag_scope_status": rag_confidence.get("scope_status", ""),
                "detected_out_of_scope_signals": "; ".join(detected_out_of_scope_signals),
                "failure_reason": failure_reason,
            }
        )

    total = len(in_scope_cases)
    summary = {
        "run_label": run_label,
        "case_set": "expanded" if output_prefix.startswith("expanded") else "smoke",
        "num_cases": total,
        "num_total_cases_including_out_of_scope": len(cases),
        "num_out_of_scope_safety_cases": len(out_of_scope_cases),
        "top_1_accuracy": round(top1_hits / total, 4) if total else 0.0,
        "top_3_accuracy": round(top3_hits / total, 4) if total else 0.0,
        "top_5_accuracy": round(top5_hits / total, 4) if total else 0.0,
        "mrr": round(sum(reciprocal_ranks) / total, 4) if total else 0.0,
        "out_of_scope_low_confidence_rate": (
            round(out_of_scope_low_confidence / len(out_of_scope_cases), 4)
            if out_of_scope_cases
            else None
        ),
        "case_ids": [case.case_id for case in in_scope_cases],
        "out_of_scope_case_ids": [case.case_id for case in out_of_scope_cases],
    }

    summary_path = output_dir / f"{output_prefix}_summary.json"
    cases_path = output_dir / f"{output_prefix}_cases.csv"
    report_path = output_dir / f"{output_prefix}_report.md"
    write_json(summary_path, summary)
    with cases_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# RAG Retrieval Evaluation Report",
        "",
        f"- Run label: `{run_label}`",
        f"- In-scope cases used for Top-k/MRR: {total}",
        f"- Out-of-scope safety cases: {len(out_of_scope_cases)}",
        f"- Top-1 accuracy: {summary['top_1_accuracy']}",
        f"- Top-3 accuracy: {summary['top_3_accuracy']}",
        f"- Top-5 accuracy: {summary['top_5_accuracy']}",
        f"- MRR: {summary['mrr']}",
        f"- Out-of-scope low-confidence rate: {summary['out_of_scope_low_confidence_rate']}",
        "",
        "## Case Results",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['case_id']}",
                f"- Scope: {row['scope']}",
                f"- Expected: {row['expected_family']}",
                f"- Rank: {row['rank'] or 'not found'}",
                f"- Retrieved: {row['retrieved_pathologies']}",
                f"- Rerank scores: {row['rerank_scores']}",
                f"- Symptom overlaps: {row['symptom_overlap_scores']}",
                f"- Lexical overlaps: {row['lexical_overlap_scores']}",
                f"- RAG confidence: {row['rag_confidence_level']} (usable_for_fusion={row['rag_usable_for_fusion']})",
                f"- RAG scope status: {row['rag_scope_status']}",
                f"- Detected out-of-scope signals: {row['detected_out_of_scope_signals'] or 'none'}",
                f"- Failure reason: {row['failure_reason']}",
                "",
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    copy_labeled_output(summary_path, run_label)
    copy_labeled_output(cases_path, run_label)
    copy_labeled_output(report_path, run_label)
    return summary


def evaluate_thresholds(
    summary: dict[str, Any],
    *,
    top5_threshold: float = DEFAULT_THRESHOLDS["top_5_accuracy"],
    mrr_threshold: float = DEFAULT_THRESHOLDS["mrr"],
    out_of_scope_threshold: float = DEFAULT_THRESHOLDS["out_of_scope_low_confidence_rate"],
) -> dict[str, Any]:
    checks = {
        "top_5_accuracy": {
            "actual": safe_float(summary.get("top_5_accuracy")),
            "threshold": float(top5_threshold),
            "passed": safe_float(summary.get("top_5_accuracy")) >= float(top5_threshold),
        },
        "mrr": {
            "actual": safe_float(summary.get("mrr")),
            "threshold": float(mrr_threshold),
            "passed": safe_float(summary.get("mrr")) >= float(mrr_threshold),
        },
    }
    out_of_scope_rate = summary.get("out_of_scope_low_confidence_rate")
    if out_of_scope_rate is not None:
        checks["out_of_scope_low_confidence_rate"] = {
            "actual": safe_float(out_of_scope_rate),
            "threshold": float(out_of_scope_threshold),
            "passed": safe_float(out_of_scope_rate) >= float(out_of_scope_threshold),
        }
    return {
        "passed": all(item["passed"] for item in checks.values()),
        "checks": checks,
    }


def append_threshold_report(report_path: Path, threshold_report: dict[str, Any]) -> None:
    lines = ["", "## Threshold Checks", "", f"- Overall passed: `{threshold_report['passed']}`"]
    for metric, payload in threshold_report["checks"].items():
        lines.append(
            f"- {metric}: actual `{payload['actual']}`, threshold `{payload['threshold']}`, "
            f"passed `{payload['passed']}`"
        )
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


async def main_async() -> None:
    args = parse_args()
    settings = get_settings()
    faiss_dir = args.faiss_index_dir or Path(settings.faiss_index_dir or "")
    model_dir = args.clinicalbert_model_dir or Path(settings.clinicalbert_model_dir or "")
    if not faiss_dir:
        raise ValueError("FAISS index directory is not configured.")
    if not model_dir:
        raise ValueError("ClinicalBERT model directory is not configured.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(args.case_set)
    output_prefix = output_prefix_for_case_set(args.case_set)

    searcher = MedicalCaseSearcher(faiss_dir)
    coverage = write_coverage_report(searcher, output_dir, faiss_dir, model_dir, cases)
    embedder = ClinicalBERTEmbedder(model_dir=model_dir)
    summary = await evaluate_cases(
        searcher=searcher,
        embedder=embedder,
        output_dir=output_dir,
        coverage=coverage,
        cases=cases,
        top_k=args.top_k,
        run_label=args.run_label,
        output_prefix=output_prefix,
    )
    threshold_report = evaluate_thresholds(
        summary,
        top5_threshold=args.top5_threshold,
        mrr_threshold=args.mrr_threshold,
        out_of_scope_threshold=args.out_of_scope_threshold,
    )
    summary["thresholds"] = threshold_report
    write_json(output_dir / f"{output_prefix}_summary.json", summary)
    append_threshold_report(output_dir / f"{output_prefix}_report.md", threshold_report)
    if args.run_label and args.run_label != "current":
        copy_labeled_output(output_dir / f"{output_prefix}_summary.json", args.run_label)
        copy_labeled_output(output_dir / f"{output_prefix}_report.md", args.run_label)
    print(
        json.dumps(
            {
                "status": "ok" if threshold_report["passed"] else "threshold_failed",
                "output_dir": str(output_dir),
                "summary": summary,
            },
            indent=2,
        )
    )
    if args.fail_on_threshold and not threshold_report["passed"]:
        raise SystemExit(2)


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
