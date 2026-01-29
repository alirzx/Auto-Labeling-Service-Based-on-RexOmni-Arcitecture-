# app/services/result_serializer.py
import base64

def normalize_result(result: dict, task: str, model: str, image_bytes: bytes | None = None, mask_bytes: bytes | None = None):
    """
    Converts Florence raw output into a unified format for all tasks.
    """
    normalized = {
        "ok": True,
        "task": task,
        "model": model,
        "image_bytes": base64.b64encode(image_bytes).decode() if image_bytes else None,
        "mask_bytes": base64.b64encode(mask_bytes).decode() if mask_bytes else None,
        "results": {},
        "artifacts": [],
    }

    if image_bytes:
        normalized["artifacts"].append("overlay")
    if mask_bytes:
        normalized["artifacts"].append("mask")

    if not result:
        return normalized

    #task_lower = task.lower()
    task_lower = task.lower().replace(" ", "_")


    # Detection tasks
    if task_lower in {"detection", "object_detection", "open_vocab_detection", "open_vocabulary_detection"}:
        normalized["results"] = {
            "bboxes": result.get("bboxes", []),
            "labels": result.get("labels", []),
            "scores": result.get("scores", []),
        }

    # Segmentation tasks
    elif task_lower in {"region_segmentation", "region_to_segmentation"}:
        normalized["results"] = {
            "polygons": result.get("polygons", []),
            "labels": result.get("labels", []),
            "bboxes": result.get("bboxes", []),
            "masks": result.get("masks", {}),
        }

    elif task_lower in {"region_category", "region_proposal"}:
        # No image output by default
        normalized["results"] = result["results"]
        normalized["artifacts"] = []  # ensure no overlay/mask


    # All other tasks
    else:
        normalized["results"] = result

    return normalized
