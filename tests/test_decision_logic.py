"""PASS/FAIL rules. No GPU."""

from inspection.decision import Detection, decide


def _box(name: str, confidence: float) -> Detection:
    return Detection(name, confidence, (0.0, 0.0, 10.0, 10.0))


def test_pass_when_nothing_exceeds_threshold():
    decision = decide([_box("corrosion", 0.50)], default=0.50, per_class={"corrosion": 0.50})
    assert decision["result"] == "PASS"
    assert decision["failing_classes"] == []


def test_pass_when_there_are_no_detections():
    decision = decide([], default=0.50, per_class={})
    assert decision["result"] == "PASS"


def test_fail_when_a_defect_exceeds_threshold():
    decision = decide([_box("corrosion", 0.51)], default=0.50, per_class={"corrosion": 0.50})
    assert decision["result"] == "FAIL"
    assert decision["failing_classes"] == ["corrosion"]


def test_per_class_threshold_overrides_default():
    per_class = {"weld_crack": 0.40, "scratch": 0.55}
    below = decide([_box("scratch", 0.54)], default=0.50, per_class=per_class)
    above = decide([_box("weld_crack", 0.41)], default=0.50, per_class=per_class)
    assert below["result"] == "PASS"
    assert above["result"] == "FAIL"
    # The scratch score would fail the default 0.50 rule. The class rule keeps it a pass.
    assert below["detections"][0]["threshold"] == 0.55
