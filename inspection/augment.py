"""Mild, physically motivated training augmentations.

The preview and the unit tests use this Albumentations pipeline so bounding
boxes move with the pixels. Ultralytics training uses the same policy limits
in ``configs/train.yaml`` (geometric and photometric). Those two code paths
are not pixel-identical; both are documented.
"""

from __future__ import annotations

import cv2
import numpy as np

from inspection.classes import ASSEMBLY_CLASSES, CLASS_NAMES
from inspection.labels import clip_yolo_box

# Reasons a plant camera would see this variation. No extreme transforms.
AUGMENTATION_POLICY = [
    {
        "name": "brightness_contrast",
        "reason": "Bay lighting, fixture aging, and auto-exposure change brightness and contrast between shifts.",
        "limits": "brightness about ±0.20, contrast about ±0.15",
    },
    {
        "name": "gamma",
        "reason": "Camera gamma and HDR-like response differ across sensors.",
        "limits": "gamma about 0.85 to 1.15",
    },
    {
        "name": "small_rotation",
        "reason": "The camera bracket and the part fixture are not perfectly square.",
        "limits": "±8 degrees. Not 90-degree rotations.",
    },
    {
        "name": "slight_perspective",
        "reason": "The camera is close to perpendicular but not perfectly so.",
        "limits": "perspective scale about 0.02 to 0.05",
    },
    {
        "name": "horizontal_flip",
        "reason": "A scratch, dent, weld mark, or corrosion patch has no left-right meaning on a flat panel.",
        "limits": "Disabled when the image contains misaligned_busbar or missing_or_loose_component.",
    },
    {
        "name": "gaussian_blur",
        "reason": "Focus error and a slightly soft lens.",
        "limits": "odd kernel size 3",
    },
    {
        "name": "motion_blur",
        "reason": "Line vibration and a short exposure smear.",
        "limits": "kernel about 5",
    },
    {
        "name": "gaussian_noise",
        "reason": "Sensor read noise, especially on darker metal.",
        "limits": "std about 0.02 of full scale",
    },
    {
        "name": "scale_translate",
        "reason": "Working distance and framing shift a few percent when the fixture is reset.",
        "limits": "scale about ±10 percent, translation about ±5 percent",
    },
    {
        "name": "color_saturation",
        "reason": "White balance and the spectrum of the lamps.",
        "limits": "mild saturation and hue",
    },
    {
        "name": "mild_shadow",
        "reason": "An operator, fixture, or overhead beam casts a soft shadow.",
        "limits": "one low-contrast shadow, not a blackout",
    },
]


def _albumentations():
    import albumentations as A

    return A


def _bbox_params(A):
    return A.BboxParams(
        format="yolo",
        label_fields=["class_ids"],
        min_visibility=0.2,
        clip=True,
    )


def _compose(transforms: list, seed: int | None = 0):
    A = _albumentations()
    pipeline = A.Compose(transforms, bbox_params=_bbox_params(A), seed=seed)
    return pipeline


def preview_transforms(allow_flip: bool, seed: int = 0) -> dict:
    """Deterministic panels for the visual check. Each value is a Compose."""
    A = _albumentations()
    panels = {
        "brightness": _compose(
            [A.RandomBrightnessContrast(brightness_limit=(0.22, 0.22), contrast_limit=(0.0, 0.0), p=1.0)],
            seed=seed,
        ),
        "blur": _compose([A.MotionBlur(blur_limit=5, p=1.0)], seed=seed),
        "noise": _compose(
            [A.GaussNoise(std_range=(0.04, 0.04), mean_range=(0.0, 0.0), per_channel=True, p=1.0)],
            seed=seed,
        ),
        "rotation": _compose(
            [A.Rotate(limit=(8, 8), border_mode=cv2.BORDER_REFLECT_101, p=1.0)],
            seed=seed,
        ),
        "perspective": _compose(
            [A.Perspective(scale=(0.04, 0.04), keep_size=True, border_mode=cv2.BORDER_REFLECT_101, p=1.0)],
            seed=seed,
        ),
    }
    combined = [
        A.RandomBrightnessContrast(brightness_limit=(0.12, 0.12), contrast_limit=(0.08, 0.08), p=1.0),
        A.RandomGamma(gamma_limit=(90, 90), p=1.0),
        A.Affine(
            scale=1.06,
            translate_percent={"x": 0.03, "y": -0.02},
            rotate=6,
            shear=0,
            border_mode=cv2.BORDER_REFLECT_101,
            p=1.0,
        ),
        A.Perspective(scale=(0.03, 0.03), keep_size=True, border_mode=cv2.BORDER_REFLECT_101, p=1.0),
        A.GaussianBlur(blur_limit=(3, 3), p=1.0),
        A.GaussNoise(std_range=(0.02, 0.02), mean_range=(0.0, 0.0), per_channel=True, p=1.0),
        A.HueSaturationValue(hue_shift_limit=(4, 4), sat_shift_limit=(12, 12), val_shift_limit=(0, 0), p=1.0),
    ]
    if allow_flip:
        combined.insert(0, A.HorizontalFlip(p=1.0))
    try:
        combined.append(
            A.RandomShadow(
                shadow_roi=(0.0, 0.0, 1.0, 1.0),
                num_shadows_limit=(1, 1),
                shadow_dimension=4,
                p=1.0,
            )
        )
    except Exception:
        pass
    panels["combined"] = _compose(combined, seed=seed)
    return panels


def horizontal_flip_transform():
    A = _albumentations()
    return _compose([A.HorizontalFlip(p=1.0)], seed=0)


def flip_allowed(class_ids: list[int]) -> bool:
    names = [CLASS_NAMES[class_id] for class_id in class_ids if 0 <= class_id < len(CLASS_NAMES)]
    return not any(name in ASSEMBLY_CLASSES for name in names)


def apply_transform(image: np.ndarray, boxes: list, class_ids: list[int], transform):
    if not boxes:
        output = transform(image=image)
        return output["image"], [], []
    output = transform(image=image, bboxes=boxes, class_ids=list(class_ids))
    clipped_boxes = []
    clipped_ids = []
    for box, class_id in zip(output["bboxes"], output["class_ids"]):
        clipped = clip_yolo_box(*[float(value) for value in box])
        if clipped is None:
            continue
        clipped_boxes.append(clipped)
        clipped_ids.append(int(class_id))
    return output["image"], clipped_boxes, clipped_ids
