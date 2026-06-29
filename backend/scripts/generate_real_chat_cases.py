"""Generate realistic free-text cases for end-to-end chat-style evaluation.

The saved classifier/RAG artifacts were built from DDX-style symptom text. This
fixture creates patient-like phrasing without using an LLM, so it is stable,
cheap, and safe to commit. Use it with evaluate_pipeline_end_to_end.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REAL_CHAT_CASES: list[dict[str, Any]] = [
    {
        "id": "real_chat_anemia_001",
        "raw_text": "I feel wiped out most days, dizzy when I stand, and I get winded walking upstairs. My family says I look pale.",
        "expected_conditions": ["Anemia"],
    },
    {
        "id": "real_chat_gerd_001",
        "raw_text": "I keep getting burning in my chest after eating, sour burps, and it is worse when lying down. I do not have shortness of breath.",
        "expected_conditions": ["GERD"],
    },
    {
        "id": "real_chat_afib_001",
        "raw_text": "My heart is racing in an irregular way and I feel dizzy and short of breath. It does not feel like a regular fast beat.",
        "expected_conditions": ["Atrial fibrillation"],
    },
    {
        "id": "real_chat_psvt_001",
        "raw_text": "I get sudden attacks where my heart races very fast, then it stops abruptly. Between attacks I feel mostly okay.",
        "expected_conditions": ["PSVT"],
    },
    {
        "id": "real_chat_bronchospasm_001",
        "raw_text": "My chest feels tight and I hear whistling in my chest. I am hard to breathe but I do not have fever or productive cough.",
        "expected_conditions": ["Bronchospasm / acute asthma exacerbation"],
    },
    {
        "id": "real_chat_pneumonia_001",
        "raw_text": "I have fever, chills, a productive cough, and sharp pain in my chest that is worse when I breathe.",
        "expected_conditions": ["Pneumonia"],
    },
    {
        "id": "real_chat_pe_001",
        "raw_text": "The breathlessness came out of nowhere after a long trip. I have sharp chest pain with a deep breath and one calf is swollen.",
        "expected_conditions": ["Pulmonary embolism"],
    },
    {
        "id": "real_chat_stable_angina_001",
        "raw_text": "I get chest pressure when walking upstairs or with exercise, and it gets better with rest after a few minutes.",
        "expected_conditions": ["Stable angina"],
    },
    {
        "id": "real_chat_unstable_angina_001",
        "raw_text": "I have chest pressure even at rest, it is getting worse this week, and I feel nauseated and sweaty.",
        "expected_conditions": ["Unstable angina"],
    },
    {
        "id": "real_chat_viral_pharyngitis_001",
        "raw_text": "My throat is scratchy and sore, I have a mild fever, runny nose, and my voice is hoarse since yesterday.",
        "expected_conditions": ["Viral pharyngitis"],
    },
    {
        "id": "real_chat_acute_laryngitis_001",
        "raw_text": "I lost my voice after a cold. My throat hurts a bit but the main issue is hoarseness and voice strain.",
        "expected_conditions": ["Acute laryngitis"],
    },
    {
        "id": "real_chat_acute_sinusitis_001",
        "raw_text": "For the past week I have a blocked nose, pressure around my face, fever, and thick yellow nasal discharge.",
        "expected_conditions": ["Acute rhinosinusitis"],
    },
    {
        "id": "real_chat_chronic_sinusitis_001",
        "raw_text": "My nose has been stuffy for months with pressure in my face and reduced smell. It keeps coming back.",
        "expected_conditions": ["Chronic rhinosinusitis"],
    },
    {
        "id": "real_chat_panic_001",
        "raw_text": "I suddenly felt like I was choking, my heart was pounding, my hands were numb, and I was scared I might die.",
        "expected_conditions": ["Panic attack"],
    },
    {
        "id": "real_chat_anaphylaxis_001",
        "raw_text": "Right after eating peanuts I became flushed, started wheezing, felt my throat tighten, and got dizzy.",
        "expected_conditions": ["Anaphylaxis"],
    },
    {
        "id": "real_chat_scombroid_001",
        "raw_text": "Soon after eating fish my face became flushed, I felt sick to my stomach, threw up, and my heart was racing.",
        "expected_conditions": ["Scombroid food poisoning"],
    },
    {
        "id": "real_chat_localized_edema_001",
        "raw_text": "Both ankles are swollen, especially by the end of the day. I do not have chest pain or fever.",
        "expected_conditions": ["Localized edema"],
    },
    {
        "id": "real_chat_pulmonary_neoplasm_001",
        "raw_text": "I have a chronic cough that is getting worse, I lost weight without trying, and I feel short of breath gradually.",
        "expected_conditions": ["Pulmonary neoplasm"],
    },
    {
        "id": "real_chat_pancreatic_neoplasm_001",
        "raw_text": "I have upper stomach pain, poor appetite, and I have been losing weight over the past months.",
        "expected_conditions": ["Pancreatic neoplasm"],
    },
    {
        "id": "real_chat_myasthenia_001",
        "raw_text": "My eyelids droop later in the day, I get double vision, and my speech and swallowing get weaker with use.",
        "expected_conditions": ["Myasthenia gravis"],
    },
    {
        "id": "real_chat_gbs_001",
        "raw_text": "I have tingling in my feet and my legs feel weak. The weakness is moving upward and I am having trouble walking.",
        "expected_conditions": ["Guillain-Barr\u00e9 syndrome"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write realistic chat-style evaluation cases.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/archive/real_chat/real_chat_cases.json"),
        help="Where to write the generated JSON case list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(REAL_CHAT_CASES, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "num_cases": len(REAL_CHAT_CASES)}, indent=2))


if __name__ == "__main__":
    main()
