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
