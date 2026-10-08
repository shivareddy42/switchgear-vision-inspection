"""The seven defect classes. Indices are the YOLO class ids."""

CLASS_NAMES = [
    "scratch",
    "dent",
    "weld_porosity",
    "weld_crack",
    "corrosion",
    "misaligned_busbar",
    "missing_or_loose_component",
]

CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

# Positional defects. A horizontal flip can change what the label means.
ASSEMBLY_CLASSES = {"misaligned_busbar", "missing_or_loose_component"}

SURFACE_CLASSES = {"scratch", "dent", "weld_porosity", "weld_crack", "corrosion"}
