"""Bounded-memory cohort generation and per-patch OpenAI visual review."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import torch
from openai import OpenAI
from PIL import Image

from ddpm_eurosat.sampling import generate_cohort_from_checkpoint

JUDGE_MODEL = "gpt-6-luna"
GENERATION_CHUNK_SIZE = 8
JUDGEMENT_BATCH_SIZE = 10
REFERENCE_COUNT = 16
GAP_CATEGORIES = [
    "color_or_palette",
    "texture_or_detail",
    "spatial_coherence",
    "noise_or_artifacts",
    "other",
]


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    pixels = (
        image.detach()
        .to(device="cpu", dtype=torch.float32)
        .add(1.0)
        .mul(127.5)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return Image.fromarray(pixels, mode="RGB")


def encode_image_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def make_reference_sheet(images: torch.Tensor) -> bytes:
    if images.ndim != 4 or images.shape[0] != REFERENCE_COUNT:
        raise ValueError(f"Expected exactly {REFERENCE_COUNT} reference patches")
    sheet = Image.new("RGB", (4 * 64, 4 * 64), color=(0, 0, 0))
    for index, image in enumerate(images):
        sheet.paste(_tensor_to_pil(image), ((index % 4) * 64, (index // 4) * 64))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue()


def generate_cohort(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    count: int,
    seed: int,
    validation_reference_images: torch.Tensor,
    validation_reference_selection: str,
    on_progress: Any = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_sheet = make_reference_sheet(validation_reference_images)
    (output_dir / "reference.png").write_bytes(reference_sheet)
    return generate_cohort_from_checkpoint(
        checkpoint_path,
        output_dir,
        num_samples=count,
        seed=seed,
        batch_size=GENERATION_CHUNK_SIZE,
        on_progress=on_progress,
        validation_reference_images=validation_reference_images,
        validation_reference_selection=validation_reference_selection,
    )


def _judgement_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "patch_id": {"type": "string"},
                        "resemblance_score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
                        "gap_categories": {
                            "type": "array",
                            "items": {"type": "string", "enum": GAP_CATEGORIES},
                        },
                        "visual_evidence": {"type": "string"},
                    },
                    "required": [
                        "patch_id",
                        "resemblance_score",
                        "gap_categories",
                        "visual_evidence",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


def judge_batch(
    client: OpenAI,
    *,
    patch_ids: list[str],
    images: torch.Tensor,
    reference_sheet: bytes,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if len(patch_ids) != len(images) or not 1 <= len(patch_ids) <= JUDGEMENT_BATCH_SIZE:
        raise ValueError("A judgement request must contain 1–10 matching patches and IDs")

    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": (
                "Judge unconditional 64x64 RGB patches from a DDPM trained on EuroSAT RGB. "
                "The reference image is a 4x4 sheet of unlabeled EuroSAT validation patches. "
                "Each generated patch is identified by the text immediately before its image. "
                "For every generated patch independently, rate resemblance to the reference "
                "patches from 1 (not similar) to 5 (strongly similar), list visible gap categories, "
                "and give one concise visual reason. A 1–2 means no clear resemblance, 3 means "
                "partial or ambiguous resemblance, and 4–5 means clear resemblance. "
                "Judge aerial/satellite texture and spatial structure, not a specific land-use class. "
                "Do not infer details that are not visible. Scores are subjective visual assessments."
            ),
        },
        {"type": "input_text", "text": "Fixed EuroSAT validation reference sheet:"},
        {
            "type": "input_image",
            "image_url": f"data:image/png;base64,{base64.b64encode(reference_sheet).decode('ascii')}",
            "detail": "high",
        },
    ]
    for patch_id, image in zip(patch_ids, images, strict=True):
        content.append({"type": "input_text", "text": f"Generated patch ID: {patch_id}"})
        content.append({
            "type": "input_image",
            "image_url": encode_image_data_url(_tensor_to_pil(image)),
            "detail": "high",
        })

    response = client.responses.create(
        model=JUDGE_MODEL,
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "eurosat_patch_judgements",
                "strict": True,
                "schema": _judgement_schema(),
            }
        },
    )
    if not response.output_text:
        raise RuntimeError("The vision model returned no structured judgement")
    parsed = json.loads(response.output_text)
    results = parsed.get("results")
    if not isinstance(results, list) or {item.get("patch_id") for item in results} != set(patch_ids):
        raise RuntimeError("The vision model response did not match the requested patch IDs")
    if len(results) != len(patch_ids):
        raise RuntimeError("The vision model returned duplicate or missing patch judgments")
    for result in results:
        score = int(result["resemblance_score"])
        result["resemblance"] = "yes" if score >= 4 else "partial" if score == 3 else "no"
    usage = response.usage
    usage_summary = {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }
    return results, usage_summary


def build_judgement_report(
    results: list[dict[str, Any]],
    generation_metadata: dict[str, Any],
) -> dict[str, Any]:
    score_counts = {str(score): 0 for score in range(1, 6)}
    gap_counts = {category: 0 for category in GAP_CATEGORIES}
    for result in results:
        score_counts[str(result["resemblance_score"])] += 1
        for category in set(result["gap_categories"]):
            gap_counts[category] += 1
    ranked = sorted(results, key=lambda result: (result["resemblance_score"], result["patch_id"]))
    count = len(results)
    return {
        "model": JUDGE_MODEL,
        "judged_images": count,
        "score_counts": score_counts,
        "resemblance_yes": sum(result["resemblance"] == "yes" for result in results),
        "resemblance_partial": sum(result["resemblance"] == "partial" for result in results),
        "resemblance_no": sum(result["resemblance"] == "no" for result in results),
        "gap_counts": gap_counts,
        "mean_resemblance_score": (
            round(sum(result["resemblance_score"] for result in results) / count, 3) if count else None
        ),
        "lowest_scoring_examples": ranked[: min(5, count)],
        "highest_scoring_examples": list(reversed(ranked[-min(5, count):])) if count else [],
        "sampler_diagnostics": generation_metadata.get("sample_statistics", {}),
        "interpretation_note": (
            "VLM ratings are subjective visual judgments, not a validated image-quality metric. "
            "Sampler diagnostics describe numeric output behavior and do not establish visual realism."
        ),
    }
