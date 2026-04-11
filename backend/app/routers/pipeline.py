"""Pipeline endpoints wrapping :class:`~manager.chat_manager.ChatManager`."""

from __future__ import annotations

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

router = APIRouter(
    tags=["pipeline"],
    dependencies=[Depends(require_service_api_key)],
)

_ALLOWED_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")
_UPLOAD_CHUNK_SIZE = 1024 * 1024


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
        result = await manager.run_pipeline(manual_input=manual_input)
    else:
        result = await manager.run_pipeline(labs=body.labs)
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
        return await manager.run_pipeline(image=tmp_path)
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
        return await manager.run_ocr_only(tmp_path)
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
        return await manager.run_from_symptoms(
            body.text,
            low_confidence_threshold=body.low_confidence_threshold,
        )
    return await manager.run_pipeline(manual_input={"symptoms": body.text})


@router.post("/pipeline/diagnosis")
async def pipeline_diagnosis(
    body: DiagnosisOnlyRequest,
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    return await manager.run_diagnosis_only(body.report)


@router.post("/pipeline/diagnosis/symptoms")
async def pipeline_diagnosis_from_symptoms(
    body: DiagnosisFromSymptomsRequest,
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    return await manager.run_from_symptoms(
        body.text,
        low_confidence_threshold=body.low_confidence_threshold,
    )


@router.post("/pipeline/diagnosis/clarify")
async def pipeline_diagnosis_clarify(
    body: ClarificationRequest,
    manager: ChatManager = Depends(get_chat_manager),
) -> Dict[str, Any]:
    return await manager.run_clarification(
        body.report,
        body.answers,
        prior_diagnosis=body.diagnosis,
        low_confidence_threshold=body.low_confidence_threshold,
    )
