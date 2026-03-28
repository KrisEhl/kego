from kego.cli.experiment import build_experiment_name, generate_id


def test_generate_id_is_6_chars():
    eid = generate_id()
    assert len(eid) == 6


def test_generate_id_is_hex():
    eid = generate_id()
    int(eid, 16)  # raises ValueError if not valid hex


def test_generate_id_is_unique():
    ids = {generate_id() for _ in range(100)}
    assert len(ids) == 100  # all unique


def test_build_experiment_name_uses_explicit_name():
    name = build_experiment_name("train_cnn.py", name="soundscape-v8", cli_params={})
    assert name == "soundscape-v8"


def test_build_experiment_name_uses_script_stem_when_no_name():
    name = build_experiment_name("train_cnn.py", name=None, cli_params={})
    assert name == "train_cnn"


def test_build_experiment_name_includes_key_params():
    name = build_experiment_name(
        "train_cnn.py",
        name=None,
        cli_params={"backbone": "efficientnet_b0", "epochs": "30"},
    )
    assert "train_cnn" in name
    assert "backbone=efficientnet_b0" in name


def test_build_experiment_name_excludes_infra_params():
    name = build_experiment_name(
        "train_cnn.py",
        name=None,
        cli_params={"debug": "true", "gpu": "0", "fold": "0", "backbone": "b0"},
    )
    assert "debug" not in name
    assert "gpu" not in name
    assert "backbone=b0" in name
