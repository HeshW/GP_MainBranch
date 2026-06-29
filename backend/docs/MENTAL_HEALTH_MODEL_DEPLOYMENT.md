# Mental Health Model Deployment

## Purpose

The mental-health model is a separate emotional-support chatbot feature. It is not part of the medical diagnosis pipeline and must not replace or modify RAG retrieval, the fine-tuned classifier, rules, diagnosis synthesis, or therapy-plan generation.

## Model

- Feature name: Mental Support
- Model id: `llama-3.2-3b-qlora-mental-support`
- Base model from adapter config: `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`
- Original requested base model: `unsloth/Llama-3.2-3B-Instruct`
- Fine-tuning method: QLoRA / LoRA adapter
- Adapter path: `backend/artifacts/artifacts/mental_health/complaint_model_final/`

The service reads `base_model_name_or_path` from `adapter_config.json` and loads the adapter with PEFT. The checked-in code keeps the feature optional and lazy-loaded; importing the API does not load the model.

## Configuration

`.env.example` includes:

```text
MENTAL_HEALTH_MODEL_DIR=backend/artifacts/artifacts/mental_health/complaint_model_final
MENTAL_HEALTH_ENABLED=true
MENTAL_HEALTH_LOAD_IN_4BIT=true
MENTAL_HEALTH_MAX_NEW_TOKENS=400
MENTAL_HEALTH_DEVICE=auto
```

Artifact/config validation has passed for this path. Full Llama generation is pending GPU validation.

## Hardware

The preferred deployment target is a CUDA GPU with enough VRAM for the 3B base model plus LoRA adapter in 4-bit quantization. CPU loading may be slow or unavailable depending on installed `torch`, `transformers`, `peft`, `accelerate`, and quantization support. The default health check does not require a GPU. Full generation was not run locally and is marked as pending GPU validation.

Optional AI dependencies are listed in `backend/requirements-ai.txt`.

## Endpoint

```http
POST /api/v1/mental-health/chat
```

Request:

```json
{
  "message": "I feel overwhelmed and anxious",
  "language": "en"
}
```

Response:

```json
{
  "reply": "...",
  "safety_status": "safe",
  "detected_language": "en",
  "model": "llama-3.2-3b-qlora-mental-support",
  "disclaimer": "This is supportive guidance, not medical diagnosis or therapy.",
  "model_loaded": true,
  "latency_ms": 1234
}
```

If the model cannot be loaded, the endpoint returns HTTP 200 with `safety_status: "unavailable"` and a clear fallback message:

```text
Mental support model is currently unavailable. Safety guidance is still available.
```

## Safety Guardrails

Guardrails run before model generation and do not require the model to be loaded.

The service returns deterministic safe responses for:

- suicide or self-harm crisis
- wanting to die
- overdose
- harming others
- severe crisis
- abuse or violence emergency
- medication prescription or dosage requests
- formal psychiatric diagnosis requests
- dangerous instructions

Crisis responses encourage emergency services, trusted-person support, and local crisis resources. The assistant must not claim to be a licensed therapist, prescribe medication, or provide formal diagnosis.

## Language Support

The service uses `ar` for Arabic and `en` for English. Arabic is detected by Arabic Unicode characters; otherwise English is used. Translation is optional and not required for the endpoint. Deterministic guardrail and fallback responses are same-language for Arabic and English. Generated same-language quality depends on the loaded model.

## Commands

Default health check without model loading:

```powershell
python backend/scripts/mental_model_health_check.py --pretty
```

Optional model load:

```powershell
python backend/scripts/mental_model_health_check.py --load-model --pretty
```

Optional generation smoke test:

```powershell
python backend/scripts/mental_model_health_check.py --load-model --generate --pretty
```

Run the generation smoke test later on a GPU environment such as Colab T4 or the deployment GPU server.

Local non-GPU evaluation:

```powershell
python backend/scripts/evaluate_mental_health_model.py --pretty
```

This local evaluation uses guardrails and fallback responses only. Full generation is `Pending GPU validation`.

Outputs are written under:

```text
data/evaluation/mental_model_diagnostics/
```

## Limitations

- This is emotional support only, not therapy, diagnosis, or emergency care.
- Guardrails are keyword based and should be reviewed with clinical/safety stakeholders before production use.
- Model loading requires compatible local AI dependencies and may need a CUDA GPU for practical latency.
- Full live inference is pending validation on a GPU host such as Colab T4 or the deployment GPU server.

## Future Work

- Add externally maintained crisis hotline localization.
- Add conversation memory scoped to the mental-support feature only.
- Add multilingual safety test cases beyond Arabic and English.
- Add monitoring for refusal rates, latency, load errors, and crisis escalations without storing sensitive message content.
