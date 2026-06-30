from __future__ import annotations

import html
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


EMU = 914400
W = 12192000
H = 6858000

BG = "061A2F"
BG2 = "0B2447"
PANEL = "0E2A4D"
PANEL2 = "123A63"
TEXT = "EAF6FF"
MUTED = "A9C2DA"
ACCENT = "38BDF8"
ACCENT2 = "22C55E"
AMBER = "FBBF24"
ROSE = "FB7185"
WHITE = "FFFFFF"


@dataclass
class ShapeWriter:
    parts: list[str]
    next_id: int = 10

    def _id(self) -> int:
        value = self.next_id
        self.next_id += 1
        return value

    @staticmethod
    def _emu(value: float) -> int:
        return int(value * EMU)

    @staticmethod
    def _safe(value: str) -> str:
        return html.escape(str(value), quote=False)

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        *,
        fill: str = PANEL,
        line: str | None = None,
        radius: bool = True,
        alpha: int | None = None,
    ) -> None:
        sid = self._id()
        line_xml = (
            f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
            if line
            else '<a:ln><a:noFill/></a:ln>'
        )
        alpha_xml = f'<a:alpha val="{alpha}"/>' if alpha is not None else ""
        geom = "roundRect" if radius else "rect"
        self.parts.append(
            f"""
            <p:sp>
              <p:nvSpPr><p:cNvPr id="{sid}" name="Shape {sid}"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>
              <p:spPr>
                <a:xfrm><a:off x="{self._emu(x)}" y="{self._emu(y)}"/><a:ext cx="{self._emu(w)}" cy="{self._emu(h)}"/></a:xfrm>
                <a:prstGeom prst="{geom}"><a:avLst/></a:prstGeom>
                <a:solidFill><a:srgbClr val="{fill}">{alpha_xml}</a:srgbClr></a:solidFill>
                {line_xml}
              </p:spPr>
              <p:txBody><a:bodyPr/><a:lstStyle/><a:p/></p:txBody>
            </p:sp>
            """
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: str = ACCENT, width: int = 18000) -> None:
        sid = self._id()
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1) or 0.01
        h = abs(y2 - y1) or 0.01
        flip_h = ' flipH="1"' if x2 < x1 else ""
        flip_v = ' flipV="1"' if y2 < y1 else ""
        self.parts.append(
            f"""
            <p:cxnSp>
              <p:nvCxnSpPr><p:cNvPr id="{sid}" name="Connector {sid}"/><p:cNvCxnSpPr/><p:nvPr/></p:nvCxnSpPr>
              <p:spPr>
                <a:xfrm{flip_h}{flip_v}><a:off x="{self._emu(x)}" y="{self._emu(y)}"/><a:ext cx="{self._emu(w)}" cy="{self._emu(h)}"/></a:xfrm>
                <a:prstGeom prst="line"><a:avLst/></a:prstGeom>
                <a:ln w="{width}"><a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:ln>
              </p:spPr>
            </p:cxnSp>
            """
        )

    def text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        text: str | list[str],
        *,
        size: int = 24,
        color: str = TEXT,
        bold: bool = False,
        align: str = "l",
        fill: str | None = None,
        line: str | None = None,
        margin: int = 91440,
        name: str = "Text",
    ) -> None:
        if fill or line:
            self.rect(
                x,
                y,
                w,
                h,
                fill=fill or BG,
                line=line,
                radius=True,
            )
            fill = None
            line = None
        sid = self._id()
        text_box_attr = "" if (fill or line) else ' txBox="1"'
        if isinstance(text, str):
            paras = text.split("\n")
        else:
            paras = text
        align_value = {"c": "ctr", "center": "ctr", "l": "l", "left": "l", "r": "r", "right": "r"}.get(align, align)
        fill_xml = (
            f'<a:solidFill><a:srgbClr val="{fill}"/></a:solidFill>'
            if fill
            else '<a:noFill/>'
        )
        line_xml = (
            f'<a:ln w="12700"><a:solidFill><a:srgbClr val="{line}"/></a:solidFill></a:ln>'
            if line
            else '<a:ln><a:noFill/></a:ln>'
        )
        p_xml = []
        for para in paras:
            p_xml.append(
                f"""
                <a:p>
                  <a:pPr algn="{align_value}"/>
                  <a:r>
                    <a:rPr lang="en-US" sz="{size * 100}" b="{1 if bold else 0}">
                      <a:solidFill><a:srgbClr val="{color}"/></a:solidFill>
                      <a:latin typeface="Aptos"/>
                    </a:rPr>
                    <a:t>{self._safe(para)}</a:t>
                  </a:r>
                </a:p>
                """
            )
        self.parts.append(
            f"""
            <p:sp>
              <p:nvSpPr><p:cNvPr id="{sid}" name="{self._safe(name)} {sid}"/><p:cNvSpPr{text_box_attr}/><p:nvPr/></p:nvSpPr>
              <p:spPr>
                <a:xfrm><a:off x="{self._emu(x)}" y="{self._emu(y)}"/><a:ext cx="{self._emu(w)}" cy="{self._emu(h)}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                {fill_xml}
                {line_xml}
              </p:spPr>
              <p:txBody>
                <a:bodyPr wrap="square" lIns="{margin}" tIns="{margin}" rIns="{margin}" bIns="{margin}"/>
                <a:lstStyle/>
                {''.join(p_xml)}
              </p:txBody>
            </p:sp>
            """
        )


def slide_xml(parts: list[str]) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:bg><p:bgPr><a:solidFill><a:srgbClr val="{BG}"/></a:solidFill><a:effectLst/></p:bgPr></p:bg>
    <p:spTree>
      <p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
      <p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
      {''.join(parts)}
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>
"""


def base(sw: ShapeWriter, footer: str = "Nabda - Multilingual NLP Medical Assistant") -> None:
    sw.rect(0, 0, 13.333, 7.5, fill=BG, radius=False)
    sw.rect(0, 0, 13.333, 0.18, fill=ACCENT, radius=False)
    sw.rect(11.0, 0.18, 2.333, 7.32, fill=BG2, radius=False, alpha=65000)
    sw.rect(0.45, 6.95, 12.45, 0.02, fill=PANEL2, radius=False)
    sw.text(0.55, 7.0, 8.5, 0.28, footer, size=8, color=MUTED)


def title_slide(title: str, subtitle: str, chips: list[str]) -> str:
    sw = ShapeWriter([])
    base(sw, "FastAPI + Next.js + ClinicalBERT + FAISS")
    sw.text(0.65, 0.55, 3.0, 0.35, "Nabda Project", size=17, color=ACCENT, bold=True)
    sw.text(0.65, 1.55, 9.1, 1.35, title, size=42, color=WHITE, bold=True)
    sw.text(0.68, 3.0, 7.7, 0.78, subtitle, size=20, color=MUTED)
    x = 0.72
    for chip in chips:
        sw.text(x, 4.08, 2.15, 0.42, chip, size=11, color=TEXT, bold=True, align="c", fill=PANEL, line=ACCENT)
        x += 2.35
    sw.rect(8.8, 1.2, 3.7, 4.9, fill=PANEL, line=PANEL2)
    sw.text(9.1, 1.55, 3.1, 0.5, "Project Scope", size=18, color=ACCENT, bold=True)
    sw.text(9.1, 2.2, 3.05, 2.9, [
        "Symptom triage and follow-up",
        "Medical report OCR and lab parsing",
        "AI diagnosis fusion with safety checks",
        "Mental support as a separate guarded feature",
    ], size=15, color=TEXT)
    return slide_xml(sw.parts)


def section_slide(name: str, number: str) -> str:
    sw = ShapeWriter([])
    base(sw)
    sw.text(0.78, 2.25, 8.6, 0.9, name, size=38, color=WHITE, bold=True)
    sw.text(0.82, 3.15, 4.0, 0.6, number, size=26, color=ACCENT, bold=True)
    sw.rect(9.55, 1.05, 2.35, 4.95, fill=PANEL, line=ACCENT)
    sw.text(9.78, 2.05, 1.85, 1.6, number, size=58, color=ACCENT, bold=True, align="c")
    return slide_xml(sw.parts)


def content_slide(title: str, bullets: list[str], *, source: str = "", tag: str = "") -> str:
    sw = ShapeWriter([])
    base(sw, source or "Nabda codebase")
    sw.text(0.62, 0.55, 9.8, 0.62, title, size=28, color=WHITE, bold=True)
    if tag:
        sw.text(10.5, 0.58, 2.05, 0.42, tag, size=12, color=ACCENT, bold=True, align="c", fill=PANEL, line=ACCENT)
    y = 1.55
    for i, bullet in enumerate(bullets, 1):
        sw.rect(0.75, y + 0.07, 0.42, 0.42, fill=ACCENT if i % 2 else ACCENT2, radius=True)
        sw.text(0.77, y + 0.1, 0.38, 0.24, str(i), size=10, color=BG, bold=True, align="c", margin=0)
        sw.text(1.3, y, 10.65, 0.68, bullet, size=18, color=TEXT)
        y += 0.88
    return slide_xml(sw.parts)


def agenda_slide() -> str:
    items = [
        "Introduction",
        "Problem Definition and Motivation",
        "Main Objectives",
        "System Architecture",
        "Phase Description",
        "Artifacts and Case Sets",
        "Experimental Results",
        "Demo",
        "Conclusion and Future Work",
        "Tools and References",
    ]
    sw = ShapeWriter([])
    base(sw)
    sw.text(0.62, 0.58, 5.5, 0.65, "Our Agenda", size=32, color=WHITE, bold=True)
    for idx, item in enumerate(items):
        col = idx % 2
        row = idx // 2
        x = 0.78 + col * 6.0
        y = 1.55 + row * 0.95
        sw.rect(x, y, 5.15, 0.62, fill=PANEL, line=PANEL2)
        sw.text(x + 0.15, y + 0.11, 0.65, 0.24, f"{idx + 1:02}", size=11, color=ACCENT, bold=True)
        sw.text(x + 0.78, y + 0.06, 4.05, 0.36, item, size=16, color=TEXT, bold=True)
    return slide_xml(sw.parts)


def two_column(title: str, left_title: str, left: list[str], right_title: str, right: list[str], *, source: str = "") -> str:
    sw = ShapeWriter([])
    base(sw, source or "Nabda codebase")
    sw.text(0.62, 0.55, 10.5, 0.62, title, size=28, color=WHITE, bold=True)
    for x, heading, bullets, color in [(0.8, left_title, left, ACCENT), (6.75, right_title, right, ACCENT2)]:
        sw.rect(x, 1.45, 5.35, 4.9, fill=PANEL, line=color)
        sw.text(x + 0.25, 1.75, 4.75, 0.45, heading, size=19, color=color, bold=True)
        y = 2.45
        for bullet in bullets:
            sw.text(x + 0.35, y, 4.65, 0.54, "- " + bullet, size=15, color=TEXT)
            y += 0.72
    return slide_xml(sw.parts)


def table_slide(title: str, headers: list[str], rows: list[list[str]], *, source: str = "") -> str:
    sw = ShapeWriter([])
    base(sw, source or "Nabda codebase")
    sw.text(0.62, 0.55, 10.5, 0.62, title, size=28, color=WHITE, bold=True)
    x0, y0, tw = 0.7, 1.45, 11.9
    col_w = tw / len(headers)
    sw.rect(x0, y0, tw, 0.55, fill=PANEL2, line=ACCENT, radius=False)
    for c, header in enumerate(headers):
        sw.text(x0 + c * col_w + 0.08, y0 + 0.1, col_w - 0.16, 0.28, header, size=12, color=WHITE, bold=True, align="c", margin=0)
    y = y0 + 0.62
    for r, row in enumerate(rows):
        fill = "0A213D" if r % 2 else PANEL
        sw.rect(x0, y, tw, 0.62, fill=fill, line=PANEL2, radius=False)
        for c, cell in enumerate(row):
            sw.text(x0 + c * col_w + 0.08, y + 0.1, col_w - 0.16, 0.32, cell, size=11, color=TEXT, align="c", margin=0)
        y += 0.66
    return slide_xml(sw.parts)


def cards_slide(title: str, cards: list[tuple[str, str, str]], *, source: str = "") -> str:
    sw = ShapeWriter([])
    base(sw, source or "Nabda codebase")
    sw.text(0.62, 0.55, 10.5, 0.62, title, size=28, color=WHITE, bold=True)
    for idx, (metric, label, note) in enumerate(cards):
        col = idx % 3
        row = idx // 3
        x = 0.72 + col * 4.05
        y = 1.55 + row * 2.3
        sw.rect(x, y, 3.55, 1.8, fill=PANEL, line=[ACCENT, ACCENT2, AMBER][idx % 3])
        sw.text(x + 0.2, y + 0.25, 3.1, 0.42, metric, size=24, color=WHITE, bold=True, align="c")
        sw.text(x + 0.2, y + 0.82, 3.1, 0.28, label, size=12, color=ACCENT, bold=True, align="c", margin=0)
        sw.text(x + 0.25, y + 1.18, 3.0, 0.32, note, size=10, color=MUTED, align="c", margin=0)
    return slide_xml(sw.parts)


def architecture_slide() -> str:
    sw = ShapeWriter([])
    base(sw, "Source: README.md, backend/app/main.py, frontend_next/README.md")
    sw.text(0.62, 0.55, 6.2, 0.62, "System Architecture", size=30, color=WHITE, bold=True)
    layers = [
        ("Presentation Layer", ["Next.js 16.2.9", "Bilingual UI", "Assistant, doctors, services"]),
        ("API Layer", ["FastAPI", "Auth and chat history", "Pipeline, chat, mental support routers"]),
        ("Orchestration Layer", ["ChatManager", "Session store", "Timeout and degraded startup guardrails"]),
        ("AI and Rules Layer", ["Symptom parser", "Clinical rules", "ClinicalBERT classifier", "FAISS RAG", "LLM synthesis"]),
        ("Data and Artifacts", ["SQLite chat DB", "12,025 FAISS vectors", "49-label classifier", "OCR patterns"]),
    ]
    y = 1.28
    for i, (name, items) in enumerate(layers):
        color = [ACCENT, ACCENT2, AMBER, ACCENT, ACCENT2][i]
        sw.rect(0.88, y, 11.55, 0.8, fill=PANEL if i % 2 == 0 else "0A213D", line=color)
        sw.text(1.1, y + 0.18, 2.45, 0.28, name, size=14, color=color, bold=True)
        sw.text(3.75, y + 0.08, 8.1, 0.4, " | ".join(items), size=12, color=TEXT)
        if i < len(layers) - 1:
            sw.line(6.55, y + 0.82, 6.55, y + 1.02, color=color)
        y += 1.02
    return slide_xml(sw.parts)


def phase_overview() -> str:
    phases = [
        "1- Symptom NLP Parsing",
        "2- OCR and Lab Extraction",
        "3- Diagnosis Fusion",
        "4- RAG and Classifier Evidence",
        "5- Safety, Clarification, and Support",
    ]
    sw = ShapeWriter([])
    base(sw)
    sw.text(0.62, 0.55, 7.2, 0.62, "Phase Description", size=30, color=WHITE, bold=True)
    for idx, phase in enumerate(phases):
        x = 0.85 + (idx % 3) * 4.0
        y = 1.65 + (idx // 3) * 2.1
        sw.rect(x, y, 3.45, 1.35, fill=PANEL, line=[ACCENT, ACCENT2, AMBER][idx % 3])
        sw.text(x + 0.25, y + 0.35, 2.95, 0.42, phase, size=17, color=TEXT, bold=True, align="c")
    return slide_xml(sw.parts)


def pipeline_flow() -> str:
    sw = ShapeWriter([])
    base(sw, "Source: backend/docs/PIPELINE_EVALUATION.md and manager/chat_manager.py")
    sw.text(0.62, 0.55, 8.5, 0.62, "Diagnosis Pipeline Flow", size=30, color=WHITE, bold=True)
    steps = [
        ("Input", "Symptoms, labs, or report image"),
        ("Parser", "Extract symptoms, labs, age, sex, context"),
        ("Normalizer", "Aliases, typo fixes, Arabic cue expansion"),
        ("Evidence", "Rules + classifier + RAG retrieval"),
        ("Fusion", "Confidence calibration and source agreement"),
        ("Output", "Diagnosis, safety, clarification, summary"),
    ]
    y = 2.0
    for idx, (name, detail) in enumerate(steps):
        x = 0.55 + idx * 2.08
        sw.rect(x, y, 1.72, 1.12, fill=PANEL, line=[ACCENT, ACCENT2, AMBER][idx % 3])
        sw.text(x + 0.1, y + 0.18, 1.52, 0.24, name, size=13, color=WHITE, bold=True, align="c", margin=0)
        sw.text(x + 0.1, y + 0.52, 1.52, 0.32, detail, size=8, color=MUTED, align="c", margin=0)
        if idx < len(steps) - 1:
            sw.line(x + 1.76, y + 0.55, x + 2.02, y + 0.55, color=ACCENT)
    sw.text(1.2, 4.15, 10.8, 0.8, "Evaluation path: user text -> parser -> validator -> normalizer -> rules -> classifier -> RAG -> fusion -> final diagnosis -> safety metadata", size=18, color=TEXT, align="c", fill="0A213D", line=PANEL2)
    return slide_xml(sw.parts)


def safety_architecture() -> str:
    sw = ShapeWriter([])
    base(sw, "Source: backend/models/diagnosis/diagnosisengine.py and backend/docs/MENTAL_HEALTH_MODEL_DEPLOYMENT.md")
    sw.text(0.62, 0.55, 8.5, 0.62, "Safety and Scope Architecture", size=30, color=WHITE, bold=True)
    branches = [
        ("Diagnosis Safety", ["Unsupported emergency scope signals", "Low-confidence and out-of-scope gating", "Rule validation status", "Clinician review metadata"]),
        ("Clarification", ["Up to 3 follow-up questions", "Candidate disease comparison", "Re-run diagnosis with answers", "Follow-up scoring override controls"]),
        ("Mental Support", ["Separate /mental-health/chat endpoint", "Crisis and self-harm guardrails", "Medication and diagnosis refusals", "Model unavailable fallback"]),
    ]
    for idx, (heading, bullets) in enumerate(branches):
        x = 0.75 + idx * 4.15
        sw.rect(x, 1.55, 3.65, 4.65, fill=PANEL, line=[ACCENT, AMBER, ACCENT2][idx])
        sw.text(x + 0.25, 1.9, 3.15, 0.35, heading, size=18, color=[ACCENT, AMBER, ACCENT2][idx], bold=True, align="c")
        y = 2.6
        for bullet in bullets:
            sw.text(x + 0.3, y, 3.0, 0.42, "- " + bullet, size=12, color=TEXT)
            y += 0.7
    return slide_xml(sw.parts)


def demo_slide() -> str:
    sw = ShapeWriter([])
    base(sw, "Source: frontend_next/src/components/medical/medical-chat.tsx and frontend_next/src/lib/api.ts")
    sw.text(0.62, 0.55, 5.5, 0.62, "Demo Flow", size=30, color=WHITE, bold=True)
    flows = [
        ("Symptom analysis", "User enters natural text; auto mode routes to /pipeline/symptoms with advanced parser enabled."),
        ("Lab analysis", "User submits lab JSON; optional symptom text is merged into /pipeline/labs."),
        ("Report image", "User uploads PNG/JPEG/WebP/BMP; backend OCR extracts report fields then runs diagnosis."),
        ("Clarification loop", "If diagnosis is uncertain, follow-up answers are sent to /pipeline/diagnosis/clarify."),
        ("Streaming chat", "After analysis, follow-up questions stream from /chat/stream with session memory."),
        ("Doctor handoff", "The doctors page uses browser geolocation and Google Maps search/directions."),
    ]
    y = 1.45
    for i, (name, detail) in enumerate(flows, 1):
        sw.rect(0.75, y, 11.6, 0.68, fill=PANEL if i % 2 else "0A213D", line=PANEL2)
        sw.text(0.95, y + 0.16, 0.55, 0.22, f"{i}", size=11, color=ACCENT, bold=True, align="c", margin=0)
        sw.text(1.55, y + 0.12, 2.25, 0.24, name, size=13, color=WHITE, bold=True, margin=0)
        sw.text(3.85, y + 0.12, 7.95, 0.28, detail, size=11, color=MUTED, margin=0)
        y += 0.78
    return slide_xml(sw.parts)


def tools_slide() -> str:
    groups = [
        ("Frontend", "Next.js 16.2.9, React 19.2.4, TypeScript, Tailwind CSS"),
        ("Backend", "FastAPI 0.135.3, Uvicorn, Pydantic Settings, SQLAlchemy, JWT"),
        ("OCR", "PaddleOCR, PaddlePaddle, OpenCV, NumPy, Pillow"),
        ("NLP and AI", "ClinicalBERT, transformers, torch, FAISS, Gemini/OpenRouter provider abstraction"),
        ("Mental Support", "Llama 3.2 3B QLoRA/LoRA adapter, PEFT, accelerate, safetensors"),
        ("Evaluation", "pytest, classifier/RAG/pipeline health checks, diagnostic report scripts"),
    ]
    return two_column(
        "Tools",
        "Runtime Stack",
        [groups[0][1], groups[1][1], groups[2][1]],
        "AI and Evaluation Stack",
        [groups[3][1], groups[4][1], groups[5][1]],
        source="Source: backend/requirements*.txt and frontend_next/package.json",
    )


def references_slide() -> str:
    rows = [
        ["Project overview", "README.md"],
        ["Backend API and endpoints", "backend/README.md, backend/app/main.py, backend/app/routers/*.py"],
        ["Frontend flows", "frontend_next/README.md, frontend_next/src/lib/api.ts, medical-chat.tsx"],
        ["Classifier metrics", "backend/docs/CLASSIFIER_EVALUATION.md, classifier summary.json"],
        ["RAG scope and metrics", "backend/docs/RAG_SCOPE_AND_LIMITATIONS.md, faiss index_info.json"],
        ["Pipeline evaluation", "backend/docs/PIPELINE_EVALUATION.md"],
        ["Mental support", "backend/docs/MENTAL_HEALTH_MODEL_DEPLOYMENT.md"],
    ]
    return table_slide("Codebase Sources", ["Topic", "Source Files"], rows, source="All content sourced from local repository files")


def build_slides() -> list[str]:
    return [
        title_slide(
            "Nabda - Multilingual NLP Medical Assistant",
            "A bilingual medical assistant prototype for symptom triage, report OCR, lab analysis, AI-assisted diagnosis, mental support, and doctor handoff.",
            ["English", "Arabic-ready", "Medical NLP", "Decision Support"],
        ),
        cards_slide(
            "Project Coverage Map",
            [
                ("Frontend", "Next.js active app", "Assistant, doctors, services, contact, mental support"),
                ("Backend", "FastAPI API", "Pipeline, auth, chat history, health, chat, mental support"),
                ("NLP", "Clinical pipeline", "Parser, validator, normalizer, classifier, RAG, rules"),
                ("Artifacts", "Saved model bundles", "49 labels, 12,025 FAISS vectors, test predictions"),
                ("Safety", "Guardrails", "Scope gating, clarification, crisis and medication safeguards"),
                ("Evaluation", "Reproducible scripts", "Classifier, RAG, pipeline, mental model diagnostics"),
            ],
            source="Source: README.md and backend/docs/PROJECT_ARTIFACTS_STRUCTURE.md",
        ),
        agenda_slide(),
        section_slide("Introduction", "01"),
        content_slide(
            "What is Nabda?",
            [
                "A bilingual medical platform for symptom triage, report OCR, lab analysis, streaming chat, and doctor search through Google Maps.",
                "The backend exposes FastAPI endpoints for symptoms, labs, report images, OCR-only extraction, diagnosis-only runs, and clarification.",
                "The diagnosis engine combines deterministic clinical rules, a fine-tuned ClinicalBERT classifier, FAISS RAG retrieval, and optional LLM synthesis.",
                "Mental Support is a separate emotional-support chatbot with its own endpoint, model adapter, and safety guardrails.",
            ],
            source="Source: frontend_next/src/lib/i18n.ts, backend/README.md, backend/docs/MENTAL_HEALTH_MODEL_DEPLOYMENT.md",
            tag="Introduction",
        ),
        section_slide("Problem Definition and Motivation", "02"),
        content_slide(
            "The Patient Interaction Gap",
            [
                "Users can arrive with free-text symptoms, structured lab values, or a medical report image; a single workflow must support all three input styles.",
                "Natural symptom descriptions can be noisy, typo-heavy, or Arabic; the parser includes deterministic normalization and multilingual cue expansion.",
                "Low-confidence first-pass diagnosis needs follow-up questions rather than a forced answer.",
                "AI output must remain educational and route urgent symptoms to real clinical care.",
            ],
            source="Source: backend/docs/PIPELINE_EVALUATION.md and frontend_next/src/components/medical/medical-chat.tsx",
        ),
        content_slide(
            "The Clinical Decision-Support Challenge",
            [
                "Diagnosis quality depends on combining parser output, lab rules, symptom rules, classifier predictions, RAG retrieval, and safety metadata.",
                "The active AI label universe is fixed at 49 DDXPlus-derived pathologies; unsupported diseases must not be presented as confident in-scope results.",
                "Out-of-scope conditions can retrieve similar indexed cases, so RAG confidence gating is required before fusion.",
                "The system must degrade gracefully when optional RAG or classifier artifacts are unavailable at startup.",
            ],
            source="Source: backend/docs/RAG_SCOPE_AND_LIMITATIONS.md and backend/app/main.py",
        ),
        content_slide(
            "The Safety and Trust Gap",
            [
                "The project is AI-assisted decision support, not clinical validation or a replacement for professional care.",
                "Unsupported emergency signals such as stroke-like symptoms or pregnancy-related emergencies are explicitly detected and escalated.",
                "Mental health support cannot provide formal diagnosis, medication prescriptions, or emergency care.",
                "Every pipeline response carries safety metadata, disclaimer text, and clinician-review signals when needed.",
            ],
            source="Source: backend/models/diagnosis/diagnosisengine.py and backend/docs/MENTAL_HEALTH_MODEL_DEPLOYMENT.md",
        ),
        two_column(
            "Motivation: Why We Built Nabda",
            "For Users",
            [
                "Start with symptoms, labs, or image uploads in one assistant.",
                "Continue uncertain cases with follow-up questions.",
                "Move from AI guidance to doctor search and directions.",
            ],
            "For the System",
            [
                "Normalize multilingual and noisy medical text before diagnosis.",
                "Fuse rules, classifier, and RAG instead of relying on one source.",
                "Expose safety, confidence, and source metadata in responses.",
            ],
            source="Source: frontend_next/README.md and backend/docs/PIPELINE_EVALUATION.md",
        ),
        two_column(
            "Motivation: Unifying Medical Workflows",
            "Supported Inputs",
            [
                "Free-text symptoms through advanced symptom parser.",
                "Structured lab JSON with optional symptom context.",
                "Report images processed through OCR and lab extraction.",
            ],
            "Supported Outputs",
            [
                "Final diagnosis, confidence, and supporting evidence.",
                "Safety notes and professional-care reminders.",
                "Clarification questions when the first pass is uncertain.",
            ],
            source="Source: backend/app/routers/pipeline.py and frontend_next/src/components/medical/medical-chat.tsx",
        ),
        table_slide(
            "Workflow Coverage Implemented in Nabda",
            ["Capability", "Backend", "Frontend", "Evidence"],
            [
                ["Symptom triage", "/pipeline/symptoms", "Auto/symptom mode", "Parser + validation + fusion"],
                ["Lab analysis", "/pipeline/labs", "Labs JSON panel", "Clinical rules + AI context"],
                ["Report OCR", "/pipeline/image and /pipeline/ocr", "Report image upload", "PaddleOCR + regex parser"],
                ["Clarification", "/pipeline/diagnosis/clarify", "Follow-up answer loop", "Re-run with merged answers"],
                ["Streaming chat", "/chat/stream", "Assistant conversation", "Session memory"],
                ["Mental support", "/mental-health/chat", "Mental Support page", "Guardrails + LoRA adapter"],
            ],
            source="Source: backend/app/routers and frontend_next/src/lib/api.ts",
        ),
        section_slide("Main Objectives", "03"),
        content_slide(
            "Main Objectives",
            [
                "Build an end-to-end medical assistant that accepts symptoms, labs, and report images through a single user-facing workflow.",
                "Use deterministic symptom parsing, normalization, and validation to prepare robust clinical input for the diagnosis pipeline.",
                "Integrate ClinicalBERT classification and FAISS RAG retrieval with rule-based safety checks and calibrated fusion.",
                "Provide bilingual UI copy, Arabic-aware parsing/translation options, and same-language safety fallbacks.",
                "Separate emotional-support chat from medical diagnosis, with explicit crisis and medication guardrails.",
            ],
            source="Source: README.md, frontend_next/README.md, backend/docs/PROJECT_ARTIFACTS_STRUCTURE.md",
        ),
        section_slide("System Architecture", "04"),
        architecture_slide(),
        section_slide("Phase Description", "05"),
        phase_overview(),
        content_slide(
            "1- Symptom NLP Parsing",
            [
                "parse_symptoms extracts labs, symptoms, and context from raw natural language.",
                "The parser normalizes noisy and multilingual text before validation.",
                "validate_parsed canonicalizes lab names, units, and symptom names while marking low-confidence review cases.",
                "build_normalized_symptom_text creates a normalized text payload for downstream classifier and RAG evidence.",
            ],
            source="Source: backend/manager/symptom_parser.py, symptom_validator.py, symptom_normalizer.py",
        ),
        content_slide(
            "2- OCR and Lab Extraction",
            [
                "OCREngine uses PaddleOCR with optional angle classification and OpenCV preprocessing.",
                "Lab extraction supports multi-match parsing, duplicate warnings, full-text, line-by-line, and cross-line fallback passes.",
                "Supported lab keys include glucose, hemoglobin, iron, WBC, RBC, platelets, hematocrit, cholesterol, creatinine, urea, sodium, potassium, and calcium.",
                "Image endpoints restrict uploads to common report image types and enforce a configured maximum upload size.",
            ],
            source="Source: backend/models/ocr/README.md and backend/app/routers/pipeline.py",
        ),
        content_slide(
            "3- Diagnosis Fusion",
            [
                "DiagnosisEngine builds a combined clinical text representation from report fields, labs, symptoms, sections, and follow-up answers.",
                "Clinical rules detect lab and symptom findings before AI fusion.",
                "Classifier and RAG outputs are collected as candidates, then calibrated with rule alignment, confidence margins, and scope signals.",
                "The final response includes diagnosis, confidence, source, supporting evidence, decision_fusion metadata, safety metadata, and summary text.",
            ],
            source="Source: backend/models/diagnosis/diagnosisengine.py and backend/models/diagnosis/text.py",
        ),
        content_slide(
            "4- RAG and Classifier Evidence",
            [
                "The active classifier bundle is clinicalbert_classifier_targeted with 49 labels and saved test predictions.",
                "The active FAISS bundle is faiss_data_targeted with 12,025 vectors and 768-dimensional embeddings.",
                "RAG reranking combines embedding similarity with symptom overlap, lexical overlap, feature alignment, lab match, age/sex alignment, disease-family hints, and penalties.",
                "Confidence gating prevents low-confidence or out-of-scope RAG from overpowering rules and classifier evidence.",
            ],
            source="Source: backend/docs/CLASSIFIER_EVALUATION.md and backend/docs/RAG_SCOPE_AND_LIMITATIONS.md",
        ),
        pipeline_flow(),
        content_slide(
            "5- Mental Support and Safety",
            [
                "Mental Support uses the llama-3.2-3b-qlora-mental-support adapter and is explicitly separate from diagnosis, RAG, classifier, rules, and therapy generation.",
                "The endpoint returns reply, safety_status, detected_language, model, disclaimer, model_loaded, and latency_ms.",
                "Guardrails run before generation and handle crisis, self-harm, overdose, harm-to-others, medication dosage, formal diagnosis, and dangerous-instruction requests.",
                "Full Llama generation is pending GPU validation; fallback and guardrail behavior can run without model loading.",
            ],
            source="Source: backend/docs/MENTAL_HEALTH_MODEL_DEPLOYMENT.md and backend/app/routers/mental_health.py",
        ),
        safety_architecture(),
        section_slide("Training and Evaluation Artifacts", "06"),
        table_slide(
            "Training and Evaluation Artifacts",
            ["Artifact", "Active Path", "Key Contents", "Status"],
            [
                ["Classifier", "clinicalbert_classifier_targeted", "49 labels, tokenizer, weights, test_predictions.csv", "Active"],
                ["RAG / FAISS", "faiss_data_targeted", "medical_cases.index, metadata, index_info", "Active"],
                ["Mental Support", "mental_health/complaint_model_final", "LoRA adapter, tokenizer, chat template", "Optional"],
                ["Pipeline cases", "data/evaluation/pipeline_diagnostics/cases", "In-scope, Arabic, noisy, ambiguous, safety", "Evaluation"],
                ["OCR rules", "backend/models/ocr", "Patterns, fields, synonyms, parsing utilities", "Runtime"],
            ],
            source="Source: backend/docs/PROJECT_ARTIFACTS_STRUCTURE.md",
        ),
        section_slide("Experimental Results", "07"),
        cards_slide(
            "ClinicalBERT Classifier Results",
            [
                ("49", "Label universe", "Consistent across classifier, RAG, FAISS metadata, and maps"),
                ("2,006", "Saved test rows", "Metrics recomputed from saved test predictions"),
                ("0.9915", "Saved-test accuracy", "summary.json and classifier evaluation docs"),
                ("0.9882", "Saved-test macro F1", "Targeted classifier bundle"),
                ("0.9796", "Smoke top-1", "Top-3 and Top-5 both 1.0000"),
                ("0.0358", "Smoke ECE", "Expected calibration error"),
            ],
            source="Source: backend/docs/CLASSIFIER_EVALUATION.md and backend/artifacts/clinicalbert_classifier_targeted/summary.json",
        ),
        cards_slide(
            "RAG Retrieval Results",
            [
                ("12,025", "FAISS vectors", "Active targeted index"),
                ("768", "Embedding dimension", "FAISS index_info.json"),
                ("49", "Unique pathologies", "DDXPlus-derived scope"),
                ("0.8636", "Expanded Top-1", "44 in-scope cases"),
                ("0.8864", "Expanded Top-5", "Above default 0.85 threshold"),
                ("1.0000", "OOS low confidence", "Out-of-scope safety cases"),
            ],
            source="Source: backend/docs/RAG_SCOPE_AND_LIMITATIONS.md and backend/artifacts/faiss_data_targeted/index_info.json",
        ),
        cards_slide(
            "End-to-End Pipeline Evaluation",
            [
                ("0.8125", "In-scope Top-1", "Latest safety/parser fix run"),
                ("1.0000", "In-scope Top-3", "Latest pipeline evaluation"),
                ("1.0000", "Expected label/family hit", "Final diagnosis family coverage"),
                ("0.8974", "Parser success overall", "All pipeline case sets"),
                ("0.8750", "Natural/Arabic/noisy parser success", "Combined parser stress cases"),
                ("0", "Failed cases", "Latest run reported no failed cases"),
            ],
            source="Source: backend/docs/PIPELINE_EVALUATION.md",
        ),
        cards_slide(
            "Safety and Scope Results",
            [
                ("1.0000", "Out-of-scope safe handling", "Pipeline safety/parser fix run"),
                ("0.0000", "Unsafe confident diagnosis rate", "Latest pipeline evaluation"),
                ("120 s", "Backend pipeline timeout", "PIPELINE_TIMEOUT_SECONDS default"),
                ("150 s", "Frontend request timeout", "REQUEST_TIMEOUT_MS"),
                ("10 MB", "Default max upload", "MAX_UPLOAD_BYTES"),
                ("3", "Max clarification questions", "DiagnosisEngine limit"),
            ],
            source="Source: backend/docs/PIPELINE_EVALUATION.md, backend/app/config.py, frontend_next/src/lib/api.ts",
        ),
        content_slide(
            "OCR and Runtime Reliability",
            [
                "OCR parsing stores exact source_match text for each extracted lab value, improving auditability of report extraction.",
                "PaddleOCR API compatibility is handled by retrying without cls when a version rejects the cls argument.",
                "Startup disables RAG or classifier automatically when required artifacts are missing, then continues in degraded mode.",
                "Pipeline endpoints return JSON 504 responses on timeout rather than leaving the frontend with a dropped upstream socket.",
            ],
            source="Source: backend/models/ocr/README.md, backend/app/main.py, backend/app/routers/pipeline.py",
        ),
        section_slide("Demo", "08"),
        demo_slide(),
        content_slide(
            "Demo Screens to Review",
            [
                "Home page: bilingual Nabda entry point with assistant and doctor handoff actions.",
                "Assistant page: full chat workspace with symptoms, labs JSON, report image upload, clarification, and streaming chat.",
                "Doctors page: browser geolocation with Google Maps doctor search and directions.",
                "Mental Support page: separate support chat showing guardrail status, model-loaded state, and latency when available.",
            ],
            source="Source: frontend_next/src/app/page.tsx, frontend_next/src/components/medical/medical-chat.tsx, frontend_next/src/components/mental/mental-support-chat.tsx",
        ),
        section_slide("Conclusion and Future Work", "09"),
        content_slide(
            "Conclusion",
            [
                "Nabda implements a full medical-assistant prototype across Next.js and FastAPI, not only an isolated model demo.",
                "The diagnosis path is evidence-fused: parser, validator, normalizer, rules, classifier, RAG, calibration, clarification, and safety metadata.",
                "The active classifier shows 0.9915 saved-test accuracy and 0.9882 macro F1 over the targeted 49-label universe.",
                "The latest pipeline run reports 0.8125 in-scope Top-1, 1.0000 Top-3, 1.0000 out-of-scope safe handling, and 0 unsafe confident diagnoses.",
                "Mental Support is intentionally separated from medical diagnosis and protected with deterministic safety guardrails.",
            ],
            source="Source: README.md and backend/docs/*.md",
        ),
        two_column(
            "Future Work and Scalability",
            "Model and Evaluation",
            [
                "Broaden the indexed medical corpus before expanding beyond the 49-label scope.",
                "Retrain classifier and rebuild FAISS if the supported label universe changes.",
                "Add more cases per pathology and more out-of-scope safety cases.",
                "Validate full mental-support generation on a GPU host.",
            ],
            "Deployment and Product",
            [
                "Add CI thresholds once evaluation sets are stable enough for release gating.",
                "Monitor latency, load errors, refusals, crisis escalations, and confidence calibration.",
                "Localize crisis resources and extend multilingual safety tests.",
                "Harden production API-key, auth, and artifact-management workflows.",
            ],
            source="Source: backend/docs/RAG_SCOPE_AND_LIMITATIONS.md and backend/docs/MENTAL_HEALTH_MODEL_DEPLOYMENT.md",
        ),
        section_slide("Tools and References", "10"),
        tools_slide(),
        references_slide(),
        title_slide(
            "Thank You",
            "Questions time",
            ["Diagnosis Pipeline", "RAG + Classifier", "Safety Guardrails", "Frontend Demo"],
        ),
    ]


def patch_docprops(deck_path: Path) -> None:
    # The reference package may contain stale document metadata. Rewriting
    # these XML files keeps the file properties aligned with the generated deck.
    replacements = {
        "docProps/core.xml": f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Nabda - Multilingual NLP Medical Assistant</dc:title>
  <dc:subject>Graduation project presentation</dc:subject>
  <dc:creator>Codex</dc:creator>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">2026-06-30T00:00:00Z</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">2026-06-30T00:00:00Z</dcterms:modified>
</cp:coreProperties>
""",
        "docProps/app.xml": """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft PowerPoint</Application>
  <PresentationFormat>On-screen Show (16:9)</PresentationFormat>
  <Slides>43</Slides>
  <Notes>0</Notes>
  <HiddenSlides>0</HiddenSlides>
  <MMClips>0</MMClips>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Theme</vt:lpstr></vt:variant><vt:variant><vt:i4>1</vt:i4></vt:variant></vt:vector></HeadingPairs>
  <TitlesOfParts><vt:vector size="1" baseType="lpstr"><vt:lpstr>Nabda Presentation</vt:lpstr></vt:vector></TitlesOfParts>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
""",
    }
    tmp_path = deck_path.with_suffix(".tmp.pptx")
    with zipfile.ZipFile(deck_path, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            if item.filename in replacements:
                zout.writestr(item, replacements[item.filename])
            else:
                zout.writestr(item, zin.read(item.filename))
    tmp_path.replace(deck_path)


def build(reference: Path, output: Path) -> None:
    slides = build_slides()
    if len(slides) != 43:
        raise RuntimeError(f"Expected 43 slides, got {len(slides)}")

    shutil.copyfile(reference, output)
    patch_docprops(output)

    tmp_path = output.with_suffix(".tmp.pptx")
    slide_pattern = re.compile(r"ppt/slides/slide(\d+)\.xml$")
    slide_rels_pattern = re.compile(r"ppt/slides/_rels/slide(\d+)\.xml\.rels$")
    with zipfile.ZipFile(output, "r") as zin, zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            match = slide_pattern.match(item.filename)
            if match:
                idx = int(match.group(1))
                if 1 <= idx <= len(slides):
                    zout.writestr(item, slides[idx - 1])
                    continue
            rels_match = slide_rels_pattern.match(item.filename)
            if rels_match:
                existing_rels = zin.read(item.filename).decode("utf-8", "ignore")
                layout_match = re.search(
                    r'<Relationship[^>]+Type="http://schemas\.openxmlformats\.org/officeDocument/2006/relationships/slideLayout"[^>]+Target="([^"]+)"[^>]*/>',
                    existing_rels,
                )
                layout_target = layout_match.group(1) if layout_match else "../slideLayouts/slideLayout7.xml"
                clean_rels = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="{layout_target}"/></Relationships>"""
                zout.writestr(item, clean_rels)
                continue
            if item.filename.startswith("ppt/media/"):
                continue
            zout.writestr(item, zin.read(item.filename))
    tmp_path.replace(output)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    reference = root / "docs" / "Presentation.pptx"
    output = root / "docs" / "Nabda_Multilingual_NLP_Medical_Assistant_Draft.pptx"
    build(reference, output)
    print(output)


if __name__ == "__main__":
    main()
