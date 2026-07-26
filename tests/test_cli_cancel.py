import json
import subprocess
import time

from kego.fleet import Machine


def _spawn_sleeper():
    return subprocess.Popen(["sleep", "60"])


def test_terminate_pid_kills_running_process(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from kego.cli.commands.cancel import _terminate_pid

    proc = _spawn_sleeper()
    try:
        assert _terminate_pid(proc.pid, grace=2.0) is True
        proc.wait(timeout=5)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_terminate_pid_returns_false_for_dead_pid():
    from kego.cli.commands.cancel import _terminate_pid

    proc = _spawn_sleeper()
    proc.kill()
    proc.wait(timeout=5)
    assert _terminate_pid(proc.pid) is False


def test_cancel_local_kills_matching_pid_and_removes_run_file(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from kego.cli.commands.cancel import cancel_local

    proc = _spawn_sleeper()
    try:
        active_runs_dir = tmp_path / ".kego" / "active_runs"
        active_runs_dir.mkdir(parents=True)
        run_file = active_runs_dir / f"{proc.pid}.json"
        run_file.write_text(json.dumps({"task": "t", "config": "c", "pid": proc.pid}))

        assert cancel_local() == 1
        assert not run_file.exists()
        proc.wait(timeout=5)
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            proc.kill()


def test_cancel_local_removes_stale_run_file_without_killing_anything(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from kego.cli.commands.cancel import cancel_local

    active_runs_dir = tmp_path / ".kego" / "active_runs"
    active_runs_dir.mkdir(parents=True)
    run_file = active_runs_dir / "999999999.json"
    run_file.write_text(json.dumps({"task": "t", "config": "c", "pid": 999999999}))

    assert cancel_local() == 0
    assert not run_file.exists()


def test_cancel_local_returns_zero_when_no_active_runs_dir(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    from kego.cli.commands.cancel import cancel_local

    assert cancel_local() == 0


def test_cancel_remote_runs_locally_when_machine_is_self(monkeypatch):
    from kego.cli.commands import cancel as cancel_mod

    monkeypatch.setattr(cancel_mod, "machine_name", lambda: "here")
    calls = []

    def fake_runner(cmd, **kw):
        calls.append(cmd)

        class R:
            stdout = ""
            stderr = ""
            returncode = 0

        return R()

    m = Machine(name="here", ssh="k@here", role="cpu", repo="/r")
    cancel_mod.cancel_remote(m, runner=fake_runner)
    assert calls[0][0] == "bash"


def test_cancel_remote_uses_ssh_for_other_machines(monkeypatch):
    from kego.cli.commands import cancel as cancel_mod

    monkeypatch.setattr(cancel_mod, "machine_name", lambda: "here")
    calls = []

    def fake_runner(cmd, **kw):
        calls.append(cmd)

        class R:
            stdout = ""
            stderr = ""
            returncode = 0

        return R()

    m = Machine(name="omarchyd", ssh="k@omarchyd", role="gpu", repo="/home/k/kego")
    cancel_mod.cancel_remote(m, runner=fake_runner)
    assert calls[0][0] == "ssh"
    assert calls[0][1] == "k@omarchyd"


def test_cancel_remote_counts_and_prints_each_killed_pid(monkeypatch, capsys):
    from kego.cli.commands import cancel as cancel_mod

    monkeypatch.setattr(cancel_mod, "machine_name", lambda: "here")

    def fake_runner(cmd, **kw):
        class R:
            stdout = (
                "Killed PID 111 (group): uv run kego train-agent --task pkmn\n"
                "Killed PID 222: uv run kego league --task pkmn\n"
            )
            stderr = ""
            returncode = 0

        return R()

    m = Machine(name="omarchyd", ssh="k@omarchyd", role="gpu", repo="/home/k/kego")
    assert cancel_mod.cancel_remote(m, runner=fake_runner) == 2
    out = capsys.readouterr().out
    assert "PID 111" in out and "PID 222" in out


def test_cancel_remote_returns_zero_when_nothing_matched():
    from kego.cli.commands.cancel import cancel_remote

    def fake_runner(cmd, **kw):
        class R:
            stdout = ""
            stderr = ""
            returncode = 0

        return R()

    m = Machine(name="omarchyd", ssh="k@omarchyd", role="gpu", repo="/home/k/kego")
    assert cancel_remote(m, runner=fake_runner) == 0


def test_cancel_ray_stops_the_lone_active_job(monkeypatch, capsys):
    from kego.cli.commands import cancel as cancel_mod

    class Job:
        def __init__(self, job_id, entrypoint):
            self.job_id = job_id
            self.entrypoint = entrypoint

    monkeypatch.setattr(cancel_mod, "list_active_ray_jobs", lambda: [Job("job1", "kego train-agent")])
    stopped = []
    monkeypatch.setattr(cancel_mod, "stop_ray_job", lambda job_id: stopped.append(job_id) or True)

    assert cancel_mod.cancel_ray(None) == 1
    assert stopped == ["job1"]
    assert "job1" in capsys.readouterr().out


def test_cancel_ray_lists_and_declines_when_ambiguous(monkeypatch, capsys):
    from kego.cli.commands import cancel as cancel_mod

    class Job:
        def __init__(self, job_id, entrypoint):
            self.job_id = job_id
            self.entrypoint = entrypoint

    jobs = [Job("job1", "kego train-agent"), Job("job2", "kego league")]
    monkeypatch.setattr(cancel_mod, "list_active_ray_jobs", lambda: jobs)
    stopped = []
    monkeypatch.setattr(cancel_mod, "stop_ray_job", lambda job_id: stopped.append(job_id) or True)

    assert cancel_mod.cancel_ray(None) == 0
    assert stopped == []
    out = capsys.readouterr().out
    assert "job1" in out and "job2" in out


def test_cancel_ray_filters_by_job_id(monkeypatch):
    from kego.cli.commands import cancel as cancel_mod

    class Job:
        def __init__(self, job_id, entrypoint):
            self.job_id = job_id
            self.entrypoint = entrypoint

    jobs = [Job("job1", "kego train-agent"), Job("job2", "kego league")]
    monkeypatch.setattr(cancel_mod, "list_active_ray_jobs", lambda: jobs)
    stopped = []
    monkeypatch.setattr(cancel_mod, "stop_ray_job", lambda job_id: stopped.append(job_id) or True)

    assert cancel_mod.cancel_ray("job2") == 1
    assert stopped == ["job2"]


def test_cancel_run_prefers_local_over_ray(monkeypatch):
    from kego.cli.commands import cancel as cancel_mod

    monkeypatch.setattr(cancel_mod, "cancel_local", lambda: 1)
    ray_calls = []
    monkeypatch.setattr(cancel_mod, "cancel_ray", lambda job_id: ray_calls.append(job_id) or 0)

    assert cancel_mod.cancel_run(None, None, "task") == 0
    assert ray_calls == []  # local found something -> never asked Ray


def test_cancel_run_falls_back_to_ray_when_nothing_local(monkeypatch):
    from kego.cli.commands import cancel as cancel_mod

    monkeypatch.setattr(cancel_mod, "cancel_local", lambda: 0)
    ray_calls = []
    monkeypatch.setattr(cancel_mod, "cancel_ray", lambda job_id: ray_calls.append(job_id) or 1)

    assert cancel_mod.cancel_run(None, None, "task") == 0
    assert ray_calls == [None]


def test_cancel_run_dispatches_ray_target_with_job_id(monkeypatch):
    from kego.cli.commands import cancel as cancel_mod

    ray_calls = []
    monkeypatch.setattr(cancel_mod, "cancel_ray", lambda job_id: ray_calls.append(job_id) or 1)

    assert cancel_mod.cancel_run("ray", "job42", "task") == 0
    assert ray_calls == ["job42"]


def test_cancel_run_resolves_run_id_via_mlflow_then_cancels_remote_machine(monkeypatch, tmp_path):
    from kego.cli.commands import cancel as cancel_mod

    monkeypatch.setattr(cancel_mod, "find_repo_root", lambda: tmp_path)
    (tmp_path / "fleet.toml").write_text(
        '[hub]\nname = "hub"\nmlflow = "http://hub:5000"\n\n'
        '[[machine]]\nname = "omarchyd"\nssh = "k@omarchyd"\nrole = "gpu"\nrepo = "/home/k/kego"\n'
    )
    monkeypatch.setattr(cancel_mod, "resolve_run_machine", lambda task_name, run_id: ("abc123", "omarchyd"))
    remote_calls = []
    monkeypatch.setattr(cancel_mod, "cancel_remote", lambda machine: remote_calls.append(machine.name) or 1)

    assert cancel_mod.cancel_run("abc123", None, "pkmn") == 0
    assert remote_calls == ["omarchyd"]


def test_cancel_run_reports_when_run_id_cannot_be_resolved(monkeypatch, tmp_path, capsys):
    from kego.cli.commands import cancel as cancel_mod

    monkeypatch.setattr(cancel_mod, "find_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cancel_mod, "resolve_run_machine", lambda task_name, run_id: None)

    assert cancel_mod.cancel_run("deadbeef", None, "pkmn") == 1
    assert "Could not resolve machine" in capsys.readouterr().out


def test_kego_cancel_help_lists_command(tmp_path):
    res = subprocess.run(["uv", "run", "kego", "--help"], cwd=str(tmp_path), capture_output=True, text=True)
    assert res.returncode == 0
    assert "cancel" in res.stdout


def test_kego_cancel_kills_a_real_local_run(tmp_path):
    proc = subprocess.Popen(["sleep", "60"])
    try:
        active_runs_dir = tmp_path / ".kego" / "active_runs"
        active_runs_dir.mkdir(parents=True)
        (active_runs_dir / f"{proc.pid}.json").write_text(json.dumps({"task": "t", "config": "c", "pid": proc.pid}))

        res = subprocess.run(
            ["uv", "run", "kego", "cancel"],
            cwd=str(tmp_path),
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert res.returncode == 0
        assert str(proc.pid) in res.stdout

        for _ in range(50):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        assert proc.poll() is not None
        assert not (active_runs_dir / f"{proc.pid}.json").exists()
    finally:
        if proc.poll() is None:
            proc.kill()
