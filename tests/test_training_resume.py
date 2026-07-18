from kego.training_resume import training_fingerprint


def test_training_fingerprint_is_stable_for_equivalent_config(tmp_path):
    source = tmp_path / "train.py"
    source.write_text("VERSION = 1\n")

    first = training_fingerprint({"model": [192, 4], "search": 10}, [source])
    second = training_fingerprint({"search": 10, "model": [192, 4]}, [source])

    assert first == second


def test_training_fingerprint_changes_with_source_or_data(tmp_path):
    source = tmp_path / "train.py"
    source.write_text("VERSION = 1\n")
    before = training_fingerprint({"model": [192, 4]}, [source])

    source.write_text("VERSION = 2\n")
    after = training_fingerprint({"model": [192, 4]}, [source])

    assert before != after
