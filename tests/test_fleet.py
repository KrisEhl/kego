import pytest

from kego.fleet import Machine, load_fleet

FLEET_TOML = """
[hub]
name = "omarchyd"
mlflow = "http://192.168.178.32:5000"

[[machine]]
name = "omarchyd"
ssh = "kristian@omarchyd"
role = "hub"
repo = "/home/kristian/projects/kego"
data = "/home/kristian/projects/kego/data"
gpus = ["rtx3090", "rtx2080ti"]

[[machine]]
name = "m5"
ssh = "kristian@m5"
role = "cpu"
repo = "/Users/kristian/projects/kego"
"""


@pytest.fixture
def fleet_file(tmp_path):
    p = tmp_path / "fleet.toml"
    p.write_text(FLEET_TOML)
    return p


def test_load_fleet_parses_hub(fleet_file):
    fleet = load_fleet(fleet_file)
    assert fleet.hub.name == "omarchyd"
    assert fleet.hub.mlflow == "http://192.168.178.32:5000"


def test_load_fleet_parses_machines(fleet_file):
    fleet = load_fleet(fleet_file)
    assert len(fleet.machines) == 2
    assert all(isinstance(m, Machine) for m in fleet.machines)


def test_machine_lookup_by_name(fleet_file):
    m = load_fleet(fleet_file).machine("m5")
    assert m.ssh == "kristian@m5"
    assert m.role == "cpu"
    assert m.repo == "/Users/kristian/projects/kego"


def test_gpus_parsed_as_tuple(fleet_file):
    assert load_fleet(fleet_file).machine("omarchyd").gpus == ("rtx3090", "rtx2080ti")


def test_gpus_default_empty_when_omitted(fleet_file):
    assert load_fleet(fleet_file).machine("m5").gpus == ()


def test_data_optional(fleet_file):
    assert load_fleet(fleet_file).machine("m5").data is None
    assert load_fleet(fleet_file).machine("omarchyd").data == "/home/kristian/projects/kego/data"


def test_unknown_machine_raises(fleet_file):
    with pytest.raises(KeyError):
        load_fleet(fleet_file).machine("does-not-exist")


def test_machine_name_from_env(monkeypatch):
    from kego.fleet import machine_name

    monkeypatch.setenv("KEGO_MACHINE", "m5")
    assert machine_name() == "m5"


def test_machine_name_falls_back_to_hostname(monkeypatch):
    import socket

    from kego.fleet import machine_name

    monkeypatch.delenv("KEGO_MACHINE", raising=False)
    assert machine_name() == socket.gethostname()


def test_git_sha_for_this_repo():
    from pathlib import Path

    from kego.fleet import git_sha

    sha = git_sha(Path(__file__).resolve().parents[1])
    assert sha != "unknown"
    assert 7 <= len(sha) <= 40


def test_git_sha_unknown_outside_repo(tmp_path):
    from kego.fleet import git_sha

    assert git_sha(tmp_path) == "unknown"


def test_repo_fleet_toml_is_valid():
    """The shipped fleet.toml at repo root parses and defines the omarchyd hub."""
    from pathlib import Path

    from kego.fleet import load_fleet

    fleet = load_fleet(Path(__file__).resolve().parents[1] / "fleet.toml")
    assert fleet.hub.name == "omarchyd"
    assert fleet.hub.mlflow.startswith("http")
    assert fleet.machine("omarchyd").role == "hub"
    for m in fleet.machines:
        assert m.ssh and m.repo and m.role, f"{m.name} missing required field"


def test_register_self_appends_new_machine(fleet_file):
    from kego.fleet import Machine, load_fleet, register_self

    new = Machine(name="wsl", ssh="k@wsl", role="gpu", repo="/home/k/kego", gpus=("rtx2080ti",))
    added = register_self(fleet_file, new)
    assert added is True
    fleet = load_fleet(fleet_file)
    assert {m.name for m in fleet.machines} == {"omarchyd", "m5", "wsl"}
    m = fleet.machine("wsl")
    assert m.ssh == "k@wsl" and m.role == "gpu" and m.repo == "/home/k/kego"
    assert m.gpus == ("rtx2080ti",)


def test_register_self_idempotent_on_existing_name(fleet_file):
    from kego.fleet import Machine, load_fleet, register_self

    added = register_self(fleet_file, Machine(name="m5", ssh="k@m5-new", role="cpu", repo="/x"))
    assert added is False
    assert len(load_fleet(fleet_file).machines) == 2


def test_register_self_cpu_machine_omits_gpus(fleet_file):
    from kego.fleet import Machine, load_fleet, register_self

    register_self(fleet_file, Machine(name="lap", ssh="k@lap", role="cpu", repo="/r"))
    m = load_fleet(fleet_file).machine("lap")
    assert m.gpus == () and m.role == "cpu"


def test_detect_machine_cpu_without_gpus(monkeypatch, tmp_path):
    from kego.fleet import detect_machine

    monkeypatch.setenv("KEGO_MACHINE", "boxA")
    m = detect_machine(repo=tmp_path, gpus=[])
    assert m.name == "boxA"
    assert "@" in m.ssh
    assert m.repo == str(tmp_path)
    assert m.role == "cpu" and m.gpus == ()


def test_detect_machine_gpu_with_gpus(monkeypatch, tmp_path):
    from kego.fleet import detect_machine

    monkeypatch.setenv("KEGO_MACHINE", "boxB")
    m = detect_machine(repo=tmp_path, gpus=["rtx3090"])
    assert m.role == "gpu" and m.gpus == ("rtx3090",)


def test_registration_summary_added_reminds_hostname():
    from kego.fleet import Machine, registration_summary

    m = Machine(name="boxA", ssh="kristianehlert@boxA", role="cpu", repo="/r")
    msg = registration_summary(m, added=True)
    assert "boxA" in msg and "kristianehlert@boxA" in msg
    assert "Registered" in msg
    assert "KEGO_MACHINE" in msg  # reminds how to override a wrong hostname


def test_registration_summary_already_present():
    from kego.fleet import Machine, registration_summary

    msg = registration_summary(Machine(name="boxA", ssh="k@boxA", role="cpu", repo="/r"), added=False)
    assert "boxA" in msg and "lready" in msg  # "Already"/"already"


def test_registration_summary_shows_gpus():
    from kego.fleet import Machine, registration_summary

    msg = registration_summary(Machine(name="g", ssh="k@g", role="gpu", repo="/r", gpus=("rtx3090",)), added=True)
    assert "gpu" in msg and "rtx3090" in msg
