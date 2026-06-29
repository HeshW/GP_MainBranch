"""Pipeline endpoints wrapping :class:`~manager.chat_manager.ChatManager`."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.config import get_settings
from app.deps import get_chat_manager, require_service_api_key
from app.schemas.pipeline import (
    ClarificationRequest,
    DiagnosisFromSymptomsRequest,
    DiagnosisOnlyRequest,
    LabsPipelineRequest,
    SymptomsPipelineRequest,
)
from manager.chat_manager import ChatManager

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["pipeline"],
    dependencies=[Depends(require_service_api_key)],
)

_ALLOWED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
_UPLOAD_CHUNK_SIZE = 1024 * 1024


async def _run_pipeline_with_timeout(operation, *, label: str) -> Dict[str, Any]:
    timeout_seconds = get_settings().pipeline_timeout_seconds
    try:
        async with asyncio.timeout(timeout_seconds):
            return await operation
    except TimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=f"{label} timed out after {timeout_seconds:.0f} seconds.",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("%s failed", label)
        raise HTTPException(
            status_code=500,
            detail=f"{label} failed. Check backend logs for details.",
        ) from exc


async def _save_upload_to_temp(
    file: UploadFile,
    *,
    suffix: str,
    max_upload_bytes: int,
) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    total = 0

    try:
        while True:
            chunk = await file.read(_UPLOAD_CHUNK_SIZE)
            if not chunk:
                break

            total += len(chunk)
            if total > max_upload_bytes:
                limit_mb = max_upload_bytes / (1024 * 1024)
                raise HTTPException(
                    status_code=413,
                    detail=f"File too large. Max upload size is {limit_mb:.1f} MB.",
                )

            tmp.write(chunk)

        if total == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        tmp.flush()
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


@router.post("/pipeline/labs")
async def pipeline_labs(
    body: LabsPipelineRequest,
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    manual_input = None
    if body.symptoms:
        manual_input = {"symptoms": body.symptoms, "labs": body.labs}
        result = await _run_pipeline_with_timeout(
            manager.run_pipeline(manual_input=manual_input),
            label="Labs pipeline",
        )
    else:
        result = await _run_pipeline_with_timeout(
            manager.run_pipeline(labs=body.labs),
            label="Labs pipeline",
        )
    return result


@router.post("/pipeline/image")
async def pipeline_image(
    file: UploadFile = File(..., description="Medical report image (PNG, JPEG, …)."),
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = os.path.splitext(file.filename)[1].lower() or ".png"
    if suffix not in _ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use PNG, JPEG, or similar.",
        )

    max_upload_bytes = get_settings().max_upload_bytes
    tmp_path = await _save_upload_to_temp(
        file,
        suffix=suffix,
        max_upload_bytes=max_upload_bytes,
    )

    try:
        return await _run_pipeline_with_timeout(
            manager.run_pipeline(image=tmp_path),
            label="Image pipeline",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/pipeline/ocr")
async def pipeline_ocr(
    file: UploadFile = File(..., description="Medical report image (PNG, JPEG, ...)."),
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    suffix = os.path.splitext(file.filename)[1].lower() or ".png"
    if suffix not in _ALLOWED_IMAGE_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use PNG, JPEG, or similar.",
        )

    max_upload_bytes = get_settings().max_upload_bytes
    tmp_path = await _save_upload_to_temp(
        file,
        suffix=suffix,
        max_upload_bytes=max_upload_bytes,
    )

    try:
        return await _run_pipeline_with_timeout(
            manager.run_ocr_only(tmp_path),
            label="OCR pipeline",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/pipeline/symptoms")
async def pipeline_symptoms(
    body: SymptomsPipelineRequest,
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    if body.use_symptom_parser:
        return await _run_pipeline_with_timeout(
            manager.run_from_symptoms(
                body.text,
                low_confidence_threshold=body.low_confidence_threshold,
            ),
            label="Symptoms pipeline",
        )
    return await _run_pipeline_with_timeout(
        manager.run_pipeline(manual_input={"symptoms": body.text}),
        label="Symptoms pipeline",
    )


@router.post("/pipeline/diagnosis")
async def pipeline_diagnosis(
    body: DiagnosisOnlyRequest,
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    return await _run_pipeline_with_timeout(
        manager.run_diagnosis_only(body.report),
        label="Diagnosis pipeline",
    )


@router.post("/pipeline/diagnosis/symptoms")
async def pipeline_diagnosis_from_symptoms(
    body: DiagnosisFromSymptomsRequest,
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    return await _run_pipeline_with_timeout(
        manager.run_from_symptoms(
            body.text,
            low_confidence_threshold=body.low_confidence_threshold,
        ),
        label="Symptoms diagnosis pipeline",
    )


@router.post("/pipeline/diagnosis/clarify")
async def pipeline_diagnosis_clarify(
    body: ClarificationRequest,
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    return await _run_pipeline_with_timeout(
        manager.run_clarification(
            body.report,
            body.answers,
            prior_diagnosis=body.diagnosis,
            low_confidence_threshold=body.low_confidence_threshold,
        ),
        label="Clarification pipeline",
    )
