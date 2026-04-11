# Hybrid NLP Medical Assistant

## Slide 1: Title

**Hybrid NLP Medical Assistant**  
From One-Shot Diagnosis to Interactive Clinical Clarification

What to say:
- Our project is a hybrid medical decision-support prototype.
- The main challenge was the gap between strong isolated components and weaker end-to-end behavior on natural user input.
- We redesigned the system toward interactive clarification instead of forcing a weak one-shot diagnosis.

---

## Slide 2: Problem Statement

**Why End-to-End Was Hard**

- Strong submodule results did not automatically produce strong final pipeline results.
- Real user input is noisy, incomplete, multilingual, and different from the training format.
- Medical ambiguity often requires follow-up questions.

What to say:
- The bottleneck was not weak models alone.
- The bottleneck was train-inference mismatch plus insufficient clarification.

---

## Slide 3: Project Features

**Main Features**

- OCR-based medical report understanding with structured lab extraction.
- Symptom parsing from manual text input.
- Hybrid diagnosis with rule engine, fine-tuned ClinicalBERT classifier, and RAG retrieval.
- Safety-oriented metadata and clinician-review style signaling.
- Therapy / recommendation generation.
- Chat assistant with short session memory.
- Clarification mode with targeted follow-up questions.
- Arabic, English, and mixed-input support.

Suggested note:
- This is not an LLM-only system. It is a modular hybrid pipeline.

---

## Slide 4: System Architecture

Use the Mermaid diagram from:

- [ARCHITECTURE_DIAGRAM.md](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/backend/docs/ARCHITECTURE_DIAGRAM.md:1)

Main flow:

`User -> OCR / Symptom Normalization -> Hybrid Diagnosis -> Decision Fusion -> Uncertainty Check -> Follow-Up Questions -> Re-Ranked Diagnosis -> Therapy / Chat Response`

What to say:
- The system combines deterministic rules, a fine-tuned classifier, retrieval, and LLM-assisted explanation.
- The clarification loop is now part of the architecture, not just an optional UI trick.

---

## Slide 5: Fine-Tuned Model Results

**Fine-Tuned ClinicalBERT**

Source:
- [summary.json](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/backend/artifacts/clinicalbert_classifier_natural/summary.json:1)

Metrics:
- Test Accuracy: `98.9%`
- Test Macro-F1: `97.9%`
- Best Validation Macro-F1: `97.36%`
- Dataset split:
  - Train: `10,000`
  - Validation: `2,000`
  - Test: `2,000`

What to say:
- The supervised classifier is very strong inside the supported label space.
- This is why we moved the architecture toward classifier-first support inside scope.

---

## Slide 6: RAG Results

**RAG Retrieval Performance**

Source:
- [rag_metrics_summary.json](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/data/evaluation/rag_metrics_summary.json:1)

Metrics:
- Top-1 Retrieval Accuracy: `98.27%`
- Top-3 Retrieval Accuracy: `99.20%`
- Top-5 Retrieval Accuracy: `99.40%`
- Macro-F1: `95.80%`
- Number of evaluated cases: `3,000`

What to say:
- The retrieval layer is strong at finding clinically similar cases.
- We use RAG mainly for support, explanation, and evidence retrieval, not as the sole final decision source.

---

## Slide 7: Why The Gap Appeared

**Strong Components, Weaker End-to-End**

- Classifier and RAG are both strong in isolation.
- But natural user free text is harder than structured DDX-style training input.
- The biggest loss happened before and during fusion:
  - symptom normalization
  - ambiguity
  - missing discriminative details
  - forced one-shot ranking

What to say:
- The main issue was not that the project lacked good AI components.
- The issue was that real-world input needed clarification and better representation.

---

## Slide 8: Main Improvements We Implemented

**From One-Shot to Interactive Triage**

- Clarification mode when confidence is low or sources disagree.
- Follow-up questions based on suspected diseases.
- Answer-aware reranking after the user replies.
- Stronger normalization for symptoms, negation, and context.
- Better fusion with safer fallback behavior.
- Practical multilingual handling for Arabic, English, and mixed phrasing.

What to say:
- We did not only add more prompts.
- We changed the architecture so follow-up answers actually influence ranking.

---

## Slide 9: One-Shot Benchmark

**Difficult Natural Free-Text Evaluation**

Source:
- [pipeline_end_to_end_current_summary.json](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/data/evaluation/pipeline_end_to_end_current_summary.json:1)

Metrics:
- Top-1 Accuracy: `5.3%`
- Top-3 Accuracy: `26.3%`
- Clinical Top-1 Accuracy: `15.8%`
- Clinical Top-3 Accuracy: `36.8%`

What to say:
- One-shot diagnosis on difficult natural free-text cases remained limited.
- This result motivated the clarification redesign.

---

## Slide 10: Interactive Clarification Results

**After Follow-Up Questions**

Source:
- [pipeline_end_to_end_clarified_v2_summary.json](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/data/evaluation/pipeline_end_to_end_clarified_v2_summary.json:1)

Metrics:
- Post-Clarification Top-1 Accuracy: `15.8%`
- Post-Clarification Top-3 Accuracy: `31.6%`
- Post-Clarification Clinical Top-1 Accuracy: `26.3%`
- Post-Clarification Clinical Top-3 Accuracy: `42.1%`
- Top-1 Gain After Clarification: `+10.5 percentage points`
- Clarification Rate: `100%`
- Average Follow-Up Questions: `3`

What to say:
- Interactive clarification improved performance on the difficult benchmark.
- This validates the shift from one-shot diagnosis to iterative refinement.

---

## Slide 11: Normalization Metrics

**Input Understanding Evaluation**

Source:
- [normalization_summary.json](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/data/evaluation/normalization_summary.json:1)

Metrics:
- Symptom Precision: `1.00`
- Symptom Recall: `0.90`
- Symptom F1: `0.947`
- Normalization Exact Match Rate: `0.75`
- Negation Exact Match Rate: `1.00`
- Context Recall: `0.75`

What to say:
- We evaluated the preprocessing layer itself, not just the final diagnosis.
- This is important because errors in normalization can break the whole pipeline.

---

## Slide 12: Multilingual Support

**Arabic + English + Mixed Input**

- Arabic and English symptom parsing.
- Arabic and English negation handling.
- Arabic and English context extraction.
- Arabic follow-up questions when the user input is Arabic.
- Mixed free text can still be normalized into a training-like report.

What to say:
- Multilingual support is practical and integrated into the diagnosis workflow, not only the chat layer.

---

## Slide 13: Error Analysis

**Where Failures Still Happen**

Source:
- [pipeline_failure_analysis.json](/C:/Users/10/Downloads/New%20folder%20(5)/GP_MainBranch-master/data/evaluation/pipeline_failure_analysis.json:1)

Failure distribution:
- Rule pattern too generic: `7`
- RAG misfire: `3`
- Classifier misfire: `3`
- Candidate list contains truth but rank failed: `2`
- Clinically close but not exact: `2`
- Other: `1`

What to say:
- We did not stop at reporting accuracy.
- We analyzed the remaining failure modes to understand what still needs improvement.

---

## Slide 14: Strengths

**Why The Project Is Strong**

- Hybrid architecture instead of LLM-only reasoning.
- Measurable evaluation at multiple levels.
- Clarification-based refinement instead of forced low-confidence diagnosis.
- Multilingual input handling.
- Modular backend and frontend structure.
- Safer and more explainable than a pure black-box response pipeline.

---

## Slide 15: Limitations

**Current Limitations**

- One-shot Top-1 remains limited on difficult natural free-text cases.
- The clarification benchmark is still relatively small and curated.
- The system is not a replacement for clinical diagnosis.
- More external validation and broader natural-text data are still needed.

What to say:
- This is a decision-support educational prototype, not a clinical deployment system.

---

## Slide 16: Honest Final Claim

**What We Can Claim**

- We identified the real bottleneck between strong components and weaker final pipeline behavior.
- We redesigned the system around progressive clarification.
- Interactive follow-up improved difficult natural-text performance.
- The project is a strong hybrid clinical decision-support prototype with a clear roadmap.

Best exact sentence:

> Our main improvement was not forcing a weak one-shot diagnosis. We shifted to progressive clarification, and this improved post-follow-up performance on difficult natural-text cases.

---

## Slide 17: What Not To Say

Avoid saying:

- The system diagnoses diseases accurately in general.
- The final product is clinically reliable for deployment.
- The assistant is a replacement for doctors.
- The one-shot model is strong.

Say instead:

- The interactive version performs better than the one-shot version.
- The project is a hybrid decision-support prototype.
- Clarification reduced forced low-confidence predictions.
- The architecture is stronger as an iterative triage assistant than as a one-shot diagnosis engine.

---

## Slide 18: Notes About Sources

- All metrics in this deck come from the current workspace files and generated evaluation artifacts.
- I cannot inspect separate Codex desktop threads directly from here, so the listed improvements are the ones implemented and verified in this project workspace.

