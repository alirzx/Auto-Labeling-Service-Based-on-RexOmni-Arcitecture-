from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import StreamingResponse
import io
import json
from typing import List, Union

from app.dependencies import get_florence_service
from inference.florence.florence_service import Florence2InferenceService

router = APIRouter(prefix="/vision/florence", tags=["florence"])


def _run_task(
    service: Florence2InferenceService,
    image_bytes: bytes,
    task_name: str,
    text_input: Union[str, None] = None,
    visualize: bool = True,
):
    """
    Run a single task and return either StreamingResponse (if image) or JSON results.
    """
    result = service.run_task_from_bytes(
        image_bytes=image_bytes,
        task_name=task_name,
        text_input=text_input,
        visualize=visualize,
    )

    # Stream image if exists
    if result.get("image_bytes") and visualize:
        headers = {"X-Florence-Results": json.dumps(result["results"])}
        return StreamingResponse(
            io.BytesIO(result["image_bytes"]),
            media_type="image/png",
            headers=headers,
        )
    return {"results": result["results"]}


async def _process_files(
    service: Florence2InferenceService,
    files: List[UploadFile],
    task_name: str,
    text_input: Union[str, None] = None,
    visualize: bool = True,
):
    """
    Batch processing: supports single or multiple files.
    """
    responses = []
    for file in files:
        image_bytes = await file.read()
        responses.append(_run_task(service, image_bytes, task_name, text_input, visualize))
    return responses if len(responses) > 1 else responses[0]


# -----------------------------
# Captioning
# -----------------------------
@router.post("/caption")
async def caption(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Caption", visualize=visualize)


@router.post("/caption_detailed")
async def caption_detailed(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Detailed Caption", visualize=visualize)


@router.post("/caption_more_detailed")
async def caption_more_detailed(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "More Detailed Caption", visualize=visualize)


# -----------------------------
# Grounding
# -----------------------------
@router.post("/caption_grounding")
async def caption_grounding(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    text_input: str = Form(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Caption + Grounding", text_input, visualize)


@router.post("/caption_grounding_detailed")
async def caption_grounding_detailed(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    text_input: str = Form(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Detailed Caption + Grounding", text_input, visualize)


@router.post("/caption_grounding_more_detailed")
async def caption_grounding_more_detailed(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    text_input: str = Form(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "More Detailed Caption + Grounding", text_input, visualize)


@router.post("/caption_to_phrase_grounding")
async def caption_to_phrase_grounding(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    text_input: str = Form(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Caption to Phrase Grounding", text_input, visualize)


# -----------------------------
# Detection
# -----------------------------
@router.post("/object_detection")
async def object_detection(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Object Detection", visualize=visualize)


# @router.post("/open_vocab_detection")
# async def open_vocab_detection(
#     file: Union[UploadFile, List[UploadFile]] = File(...),
#     text_input: str = Form(...),
#     visualize: bool = Form(True),
#     service: Florence2InferenceService = Depends(get_florence_service),
# ):
#     files = file if isinstance(file, list) else [file]
#     return await _process_files(service, files, "Open Vocabulary Detection", text_input, visualize)


@router.post("/open_vocab_detection")
async def open_vocab_detection(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    categories: str = Form(...),  # <-- IMPORTANT
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(
        service,
        files,
        "Open Vocabulary Detection",
        text_input=categories,  # pass as text_input internally
        visualize=visualize
    )


# -----------------------------
# Segmentation
# -----------------------------
@router.post("/referring_expression_segmentation")
async def referring_expression_segmentation(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    text_input: str = Form(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Referring Expression Segmentation", text_input, visualize)


@router.post("/region_segmentation")
async def region_segmentation(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    text_input: str = Form(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Region to Segmentation", text_input, visualize)


# -----------------------------
# Region reasoning
# -----------------------------
@router.post("/region_category")
async def region_category(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    text_input: str = Form(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Region to Category", text_input, visualize)


@router.post("/region_description")
async def region_description(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    text_input: str = Form(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Region to Description", text_input, visualize)


@router.post("/region_proposal")
async def region_proposal(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Region Proposal", visualize=visualize)


@router.post("/dense_region_caption")
async def dense_region_caption(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "Dense Region Caption", visualize=visualize)


# -----------------------------
# OCR
# -----------------------------
@router.post("/ocr")
async def ocr(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "OCR", visualize=visualize)


@router.post("/ocr_with_region")
async def ocr_with_region(
    file: Union[UploadFile, List[UploadFile]] = File(...),
    visualize: bool = Form(True),
    service: Florence2InferenceService = Depends(get_florence_service),
):
    files = file if isinstance(file, list) else [file]
    return await _process_files(service, files, "OCR with Region", visualize=visualize)
