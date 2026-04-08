"""Pipeline endpoints wrapping :class:`~manager.chat_manager.ChatManager`."""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.deps import get_chat_manager
from app.schemas.pipeline import LabsPipelineRequest, SymptomsPipelineRequest
from manager.chat_manager import ChatManager

router = APIRouter(tags=["pipeline"])


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
    if suffix not in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Use PNG, JPEG, or similar.",
        )
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return await manager.run_pipeline(image=tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
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
