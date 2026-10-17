from glm_review.storage import atomic_write, read_jsonl, write_jsonl


def test_atomic_write_replaces_existing_content(tmp_path):
    destination = tmp_path / "artifact.md"
    atomic_write(destination, "first")
    atomic_write(destination, "second")
    assert destination.read_text() == "second"


def test_jsonl_round_trip(tmp_path):
    destination = tmp_path / "samples.jsonl"
    records = [{"sample": 0, "text": "α"}, {"sample": 1, "text": "β"}]
    write_jsonl(destination, records)
    assert read_jsonl(destination) == records
