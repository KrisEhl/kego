from kego.fleet import Machine


def test_default_excludes_cover_heavy_dirs():
    from kego.dispatch import DEFAULT_EXCLUDES

    for e in (".git", ".venv", "data", "outputs", "mlruns", "notebooks", "cluster", "catboost_info"):
        assert e in DEFAULT_EXCLUDES


def test_rsync_command_ships_repo_contents():
    from kego.dispatch import rsync_command

    m = Machine(name="m5", ssh="k@m5", role="cpu", repo="/home/k/kego")
    cmd = rsync_command("/local/kego", m, [".git", "data"])
    assert cmd[0] == "rsync"
    assert "--delete" in cmd and "--force" in cmd and "--delete-excluded" not in cmd
    assert "--exclude=.git" in cmd and "--exclude=data" in cmd
    # trailing slashes: copy the *contents* of local into the remote repo dir
    assert cmd[-2] == "/local/kego/"
    assert cmd[-1] == "k@m5:/home/k/kego/"


def test_remote_launch_command_sets_run_id_and_detaches():
    from kego.dispatch import remote_launch_command

    m = Machine(name="m5", ssh="k@m5", role="cpu", repo="/home/k/kego")
    rc = remote_launch_command(
        m,
        ["train-agent", "--task", "pkmn", "--epochs", "200", "--init-checkpoint", "registry:12"],
        run_id="abc123",
        git_sha_val="66c8a32",
    )
    assert "cd /home/k/kego" in rc
    assert "KEGO_MLFLOW_RUN_ID=abc123" in rc
    assert "KEGO_GIT_SHA=66c8a32" in rc
    assert "uv run kego train-agent --task pkmn --epochs 200 --init-checkpoint registry:12" in rc
    assert "nohup" in rc and "abc123.log" in rc and "< /dev/null" in rc
    assert "disown; exit 0" in rc


def test_ssh_command_wraps_remote_in_login_shell():
    from kego.dispatch import ssh_command

    m = Machine(name="m5", ssh="k@m5", role="cpu", repo="/r")
    assert ssh_command(m, "echo hi") == ["ssh", "-f", "-n", "k@m5", "bash -lc 'echo hi'"]


def test_other_competition_excludes_keeps_active_only(tmp_path):
    from kego.dispatch import other_competition_excludes

    comps = tmp_path / "competitions"
    (comps / "keep").mkdir(parents=True)
    (comps / "drop").mkdir()
    ex = other_competition_excludes(tmp_path, keep="keep")
    assert "/competitions/drop" in ex
    assert not any("keep" in e for e in ex)
    # Anchored to the transfer root so --delete can still clean up stale
    # same-named directories elsewhere in the remote tree (e.g. kego/competitions/).
    assert all(e.startswith("/competitions/") for e in ex)


def test_dispatch_ships_then_launches():
    from kego.dispatch import dispatch

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)

        class R:
            returncode = 0

        return R()

    m = Machine(name="m5", ssh="k@m5", role="cpu", repo="/home/k/kego")
    dispatch(
        m, ["train-agent", "--task", "pkmn"], run_id="abc", local_dir="/local/kego", excludes=[".git"], runner=fake_run
    )
    assert calls[0][0] == "rsync"  # ship first
    assert calls[1][0] == "ssh"  # then launch
    assert "KEGO_MLFLOW_RUN_ID=abc" in calls[1][-1]


def test_dispatch_raises_if_rsync_fails():
    import pytest

    from kego.dispatch import dispatch

    def failing_run(cmd, **kw):
        class R:
            returncode = 1 if cmd[0] == "rsync" else 0

        return R()

    m = Machine(name="m5", ssh="k@m5", role="cpu", repo="/r")
    with pytest.raises(RuntimeError):
        dispatch(m, ["train-agent"], run_id="x", local_dir="/l", excludes=[], runner=failing_run)


def test_dispatch_raises_if_ssh_launch_fails():
    import pytest

    from kego.dispatch import dispatch

    def failing_run(cmd, **kw):
        class R:
            returncode = 1 if cmd[0] == "ssh" else 0

        return R()

    m = Machine(name="m5", ssh="k@m5", role="cpu", repo="/r")
    with pytest.raises(RuntimeError, match="ssh launch"):
        dispatch(m, ["league"], run_id="x", local_dir="/l", excludes=[], runner=failing_run)
