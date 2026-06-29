"""Lazy LoRA-backed mental-health support assistant with deterministic safety gates."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from threading import RLock
from typing import Any

from app.config import get_settings

MODEL_ID = "llama-3.2-3b-qlora-mental-support"
DISCLAIMER = "This is supportive guidance, not medical diagnosis or therapy."
UNAVAILABLE_MESSAGE = "Mental support model is currently unavailable. Safety guidance is still available."
REQUESTED_DEFAULT_MODEL_DIR = Path("backend/artifacts/artifacts/mental_health/complaint_model_final")
DISCOVERED_MODEL_DIR = Path("backend/artifacts/artifacts/mental_health/complaint_model_final")
REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "README.md",
)

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LOAD_LOCK = RLock()
_STATE: dict[str, Any] = {
    "model": None,
    "tokenizer": None,
    "loaded": False,
    "load_error": None,
    "model_dir": None,
    "base_model": None,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(value: str | Path | None) -> Path:
    if not value:
        return _repo_root() / REQUESTED_DEFAULT_MODEL_DIR
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def resolve_model_dir(configured_dir: str | Path | None = None) -> Path:
    """Return the configured adapter dir, falling back to the extracted artifact layout."""

    settings = get_settings()
    configured = configured_dir or getattr(settings, "mental_health_model_dir", None)
    requested = _resolve_path(configured)
    if requested.is_dir():
        return requested

    fallback = _repo_root() / DISCOVERED_MODEL_DIR
    if fallback.is_dir():
        return fallback
    return requested


def read_adapter_config(model_dir: str | Path | None = None) -> dict[str, Any]:
    path = resolve_model_dir(model_dir) / "adapter_config.json"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def inspect_mental_model_assets(model_dir: str | Path | None = None) -> dict[str, Any]:
    settings = get_settings()
    configured_value = str(model_dir or getattr(settings, "mental_health_model_dir", REQUESTED_DEFAULT_MODEL_DIR))
    configured_path = _resolve_path(configured_value)
    resolved_path = resolve_model_dir(model_dir)
    adapter_config = read_adapter_config(resolved_path)
    missing = [name for name in REQUIRED_FILES if not (resolved_path / name).is_file()]
    peft_type = str(adapter_config.get("peft_type") or "").upper()
    is_lora_adapter = peft_type == "LORA" and (resolved_path / "adapter_model.safetensors").is_file()
    looks_merged = (resolved_path / "config.json").is_file() and any(
        (resolved_path / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    )

    warnings: list[str] = []
    if configured_path != resolved_path:
        warnings.append(
            "Configured mental-health artifact path was not found; using discovered extracted adapter path."
        )
    if looks_merged:
        warnings.append("Directory also contains merged-model-looking files; verify deployment target.")

    status = "ok" if resolved_path.is_dir() and not missing and is_lora_adapter else "blocked"
    return {
        "status": status,
        "configured_model_dir": str(configured_path),
        "resolved_model_dir": str(resolved_path),
        "model_dir_exists": resolved_path.is_dir(),
        "required_files": list(REQUIRED_FILES),
        "missing_files": missing,
        "adapter_config_exists": (resolved_path / "adapter_config.json").is_file(),
        "adapter_weights_exists": (resolved_path / "adapter_model.safetensors").is_file(),
        "tokenizer_files_exist": all((resolved_path / name).is_file() for name in ("tokenizer.json", "tokenizer_config.json")),
        "base_model_name_or_path": adapter_config.get("base_model_name_or_path"),
        "peft_type": adapter_config.get("peft_type"),
        "task_type": adapter_config.get("task_type"),
        "is_lora_adapter": is_lora_adapter,
        "is_merged_model": bool(looks_merged and not is_lora_adapter),
        "warnings": warnings,
    }


def detect_language(message: str, language: str | None = None) -> str:
    requested = (language or "").strip().lower()
    if requested in {"ar", "en"}:
        return requested
    return "ar" if _ARABIC_RE.search(message or "") else "en"


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    normalized = f" {text.lower()} "
    return any(phrase in normalized for phrase in phrases)


CRISIS_TERMS = (
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "wanna die",
    "don't want to live",
    "do not want to live",
    "hurt myself",
    "harm myself",
    "self harm",
    "self-harm",
    "cut myself",
    "overdose",
    "jump off",
    "hang myself",
    "kill someone",
    "hurt someone",
    "harm others",
    "harming others",
    "abuse emergency",
    "violence emergency",
    "beaten",
    "assaulted",
    "انتحار",
    "اقتل نفسي",
    "اموت",
    "أموت",
    "عايز اموت",
    "أريد أن أموت",
    "إيذاء نفسي",
    "اذي نفسي",
    "جرعة زائدة",
    "اقتل شخص",
)
MEDICATION_TERMS = (
    "prescribe",
    "prescription",
    "what medication",
    "which medication",
    "what meds",
    "dosage",
    "dose",
    "xanax",
    "prozac",
    "sertraline",
    "fluoxetine",
    "antidepressant",
    "ssri",
    "benzodiazepine",
    "give me medicine",
    "وصفة",
    "دواء",
    "جرعة",
    "مضاد اكتئاب",
)
DIAGNOSIS_TERMS = (
    "diagnose me",
    "formal diagnosis",
    "psychiatric diagnosis",
    "do i have depression",
    "am i depressed",
    "am i bipolar",
    "do i have anxiety disorder",
    "mental illness diagnosis",
    "تشخيص",
    "شخصني",
    "هل عندي اكتئاب",
    "هل لدي اكتئاب",
)
DANGEROUS_TERMS = (
    "how to overdose",
    "how can i overdose",
    "how to cut",
    "hide self harm",
    "hurt someone without",
    "make a weapon",
    "كيفية الانتحار",
    "ازاي انتحر",
)


def _crisis_reply(language: str) -> str:
    if language == "ar":
        return (
            "أنا آسف إنك تمر بهذا الألم. إذا كنت في خطر فوري أو قد تؤذي نفسك أو شخصا آخر، "
            "اتصل بخدمات الطوارئ المحلية الآن أو اذهب لأقرب قسم طوارئ. حاول البقاء مع شخص تثق به، "
            "وأبعد أي وسيلة قد تؤذيك. إذا كان بإمكانك، تواصل مع خط دعم أزمات محلي أو فرد من العائلة الآن."
        )
    return (
        "I'm really sorry you're carrying this. If you might hurt yourself or someone else, "
        "please contact local emergency services now or go to the nearest emergency department. "
        "Stay near a trusted person if you can, move away from anything you could use to harm yourself, "
        "and contact a local crisis hotline or someone you trust right now."
    )


def _medication_reply(language: str) -> str:
    if language == "ar":
        return (
            "لا أستطيع وصف أدوية أو جرعات. يمكنني مساعدتك في ترتيب ما تشعر به، وكتابة أسئلة آمنة "
            "لطبيب أو طبيب نفسي، والتفكير في خطوات تهدئة مؤقتة مثل التنفس البطيء والتواصل مع شخص موثوق."
        )
    return (
        "I can't prescribe medication or dosing. I can help you organize what you're feeling, "
        "prepare questions for a clinician, and think through safer short-term coping steps like paced breathing, "
        "grounding, and reaching out to someone you trust."
    )


def _diagnosis_reply(language: str) -> str:
    if language == "ar":
        return (
            "لا أستطيع تقديم تشخيص نفسي رسمي. أستطيع أن أساعدك في وصف الأعراض والأنماط التي تلاحظها "
            "والتحضير لمحادثة مع مختص مرخص."
        )
    return (
        "I can't provide a formal psychiatric diagnosis. I can help you describe your symptoms, "
        "notice patterns, and prepare for a conversation with a licensed professional."
    )


def _dangerous_reply(language: str) -> str:
    if language == "ar":
        return "لا أستطيع المساعدة في تعليمات قد تسبب الأذى. أستطيع البقاء معك ومساعدتك على الوصول إلى دعم آمن الآن."
    return "I can't help with instructions that could cause harm. I can stay with you and help you get safer support right now."


def apply_mental_health_guardrails(message: str, language: str | None = None) -> dict[str, Any]:
    detected_language = detect_language(message, language)
    text = message or ""
    if _contains_any(text, CRISIS_TERMS):
        return {"blocked": True, "safety_status": "crisis", "reply": _crisis_reply(detected_language), "detected_language": detected_language}
    if _contains_any(text, DANGEROUS_TERMS):
        return {"blocked": True, "safety_status": "dangerous_refusal", "reply": _dangerous_reply(detected_language), "detected_language": detected_language}
    if _contains_any(text, MEDICATION_TERMS):
        return {"blocked": True, "safety_status": "medication_refusal", "reply": _medication_reply(detected_language), "detected_language": detected_language}
    if _contains_any(text, DIAGNOSIS_TERMS):
        return {"blocked": True, "safety_status": "diagnosis_refusal", "reply": _diagnosis_reply(detected_language), "detected_language": detected_language}
    return {"blocked": False, "safety_status": "safe", "reply": None, "detected_language": detected_language}


def _fallback_support_reply(message: str, language: str) -> str:
    if language == "ar":
        return (
            "أسمعك. خذ لحظة بطيئة الآن: تنفس بهدوء، وسم ما تشعر به في جملة قصيرة، "
            "ثم اختر خطوة صغيرة قابلة للتنفيذ مثل شرب ماء، المشي لدقيقتين، أو مراسلة شخص تثق به. "
            "إذا أردت، أخبرني بما حدث اليوم وسنرتبه معا."
        )
    return (
        "I hear you. Take one slower breath with me, name the feeling in one plain sentence, "
        "then choose one small next step: drink water, step outside for two minutes, or message someone you trust. "
        "If you want to share what happened today, we can sort it out together."
    )


def _build_prompt(message: str, language: str) -> list[dict[str, str]]:
    language_instruction = "Reply in Arabic." if language == "ar" else "Reply in English."
    return [
        {
            "role": "system",
            "content": (
                "You are a warm emotional-support assistant. You are not a licensed therapist. "
                "Do not diagnose, prescribe medication, or provide dangerous instructions. "
                "Encourage professional or emergency support when appropriate. "
                f"{language_instruction}"
            ),
        },
        {"role": "user", "content": message},
    ]


def preload_model() -> dict[str, Any]:
    """Load tokenizer, base model, and LoRA adapter once; return a small status payload."""

    with _LOAD_LOCK:
        if _STATE["loaded"]:
            return {"model_loaded": True, "error": None, "base_model": _STATE.get("base_model")}
        if _STATE["load_error"]:
            return {"model_loaded": False, "error": _STATE["load_error"], "base_model": _STATE.get("base_model")}

        assets = inspect_mental_model_assets()
        if assets["status"] != "ok":
            _STATE["load_error"] = f"mental-health model assets are not ready: missing={assets['missing_files']}"
            return {"model_loaded": False, "error": _STATE["load_error"], "base_model": assets.get("base_model_name_or_path")}

        model_dir = Path(assets["resolved_model_dir"])
        base_model = str(assets.get("base_model_name_or_path") or "").strip()
        _STATE["model_dir"] = str(model_dir)
        _STATE["base_model"] = base_model

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
            try:
                from transformers import BitsAndBytesConfig
            except Exception:
                BitsAndBytesConfig = None

            settings = get_settings()
            cuda_available = bool(torch.cuda.is_available())
            load_in_4bit = bool(getattr(settings, "mental_health_load_in_4bit", True))
            device_pref = str(getattr(settings, "mental_health_device", "auto") or "auto").lower()
            if load_in_4bit and not cuda_available and device_pref == "auto":
                _STATE["load_error"] = "CUDA is unavailable for the configured 4-bit mental-health model load."
                return {"model_loaded": False, "error": _STATE["load_error"], "base_model": base_model}

            quantization_config = None
            if load_in_4bit and cuda_available and BitsAndBytesConfig is not None:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            kwargs: dict[str, Any] = {"trust_remote_code": True}
            if quantization_config is not None:
                kwargs["quantization_config"] = quantization_config
                kwargs["device_map"] = "auto"
            elif device_pref == "auto":
                kwargs["device_map"] = "auto" if cuda_available else None
                kwargs["torch_dtype"] = torch.float16 if cuda_available else torch.float32
            else:
                kwargs["device_map"] = {"": device_pref}

            kwargs = {key: value for key, value in kwargs.items() if value is not None}
            base = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
            model = PeftModel.from_pretrained(base, model_dir)
            model.eval()
            _STATE["tokenizer"] = tokenizer
            _STATE["model"] = model
            _STATE["loaded"] = True
            return {"model_loaded": True, "error": None, "base_model": base_model}
        except Exception as exc:
            _STATE["load_error"] = f"{type(exc).__name__}: {exc}"
            return {"model_loaded": False, "error": _STATE["load_error"], "base_model": base_model}


def generate_mental_support_reply(
    message: str,
    language: str | None = None,
    max_new_tokens: int | None = None,
    allow_model_load: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    guardrail = apply_mental_health_guardrails(message, language)
    detected_language = str(guardrail["detected_language"])
    if guardrail["blocked"]:
        return {
            "reply": guardrail["reply"],
            "detected_language": detected_language,
            "safety_status": guardrail["safety_status"],
            "model_loaded": bool(_STATE["loaded"]),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    settings = get_settings()
    if not bool(getattr(settings, "mental_health_enabled", True)):
        return {
            "reply": UNAVAILABLE_MESSAGE + " " + _fallback_support_reply(message, detected_language),
            "detected_language": detected_language,
            "safety_status": "unavailable",
            "model_loaded": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    if not allow_model_load:
        return {
            "reply": UNAVAILABLE_MESSAGE + " " + _fallback_support_reply(message, detected_language),
            "detected_language": detected_language,
            "safety_status": "unavailable",
            "model_loaded": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }

    load_status = preload_model()
    if not load_status["model_loaded"]:
        return {
            "reply": UNAVAILABLE_MESSAGE + " " + _fallback_support_reply(message, detected_language),
            "detected_language": detected_language,
            "safety_status": "unavailable",
            "model_loaded": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error": load_status.get("error"),
        }

    try:
        import torch

        tokenizer = _STATE["tokenizer"]
        model = _STATE["model"]
        messages = _build_prompt(message, detected_language)
        if hasattr(tokenizer, "apply_chat_template"):
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        else:
            prompt = f"{messages[0]['content']}\nUser: {message}\nAssistant:"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        target_device = getattr(model, "device", None)
        if target_device is not None:
            input_ids = input_ids.to(target_device)

        limit = int(max_new_tokens or getattr(settings, "mental_health_max_new_tokens", 400))
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=limit,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=getattr(tokenizer, "eos_token_id", None),
            )
        generated = output_ids[0][input_ids.shape[-1] :]
        reply = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if not reply:
            reply = _fallback_support_reply(message, detected_language)
        return {
            "reply": reply,
            "detected_language": detected_language,
            "safety_status": "safe",
            "model_loaded": True,
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {
            "reply": UNAVAILABLE_MESSAGE + " " + _fallback_support_reply(message, detected_language),
            "detected_language": detected_language,
            "safety_status": "unavailable",
            "model_loaded": False,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
        }
