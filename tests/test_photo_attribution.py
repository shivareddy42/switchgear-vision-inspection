"""Published annotated photos must keep their source attribution."""
import csv
import json
from pathlib import Path


def test_every_domain_gap_preview_has_an_attribution_row():
    directory = Path(__file__).resolve().parents[1] / "outputs" / "real_photo_predictions"
    with (directory / "attribution.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {Path(image["annotated"]).name for image in json.loads((directory / "summary.json").read_text())["images"]}
    assert len(rows) == len(expected) == 67
    assert {row["annotated_file"] for row in rows} == expected
    for row in rows:
        assert (directory / row["annotated_file"]).is_file()
        assert all(row[field] for field in ("source_page", "creator", "license", "license_url", "modifications"))
