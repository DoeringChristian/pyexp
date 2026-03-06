"""Tests for SshHost, provisioners, and SshExecutor."""

import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from pyexp.executors import (
    EXECUTORS,
    AutoProvisioner,
    SshExecutor,
    SshHost,
    UvProvisioner,
    PipProvisioner,
    PixiProvisioner,
    _detect_provisioner,
)


# ---------------------------------------------------------------------------
# SshHost defaults
# ---------------------------------------------------------------------------

class TestSshHost:
    def test_defaults(self):
        h = SshHost("user@gpu1")
        assert h.host == "user@gpu1"
        assert h.max_tasks == 1
        assert h.setup is None
        assert h.work_dir is None

    def test_custom_values(self):
        h = SshHost("user@gpu2", max_tasks=4, setup="source ~/env/bin/activate", work_dir="/data/work")
        assert h.max_tasks == 4
        assert h.setup == "source ~/env/bin/activate"
        assert h.work_dir == "/data/work"


# ---------------------------------------------------------------------------
# UvProvisioner
# ---------------------------------------------------------------------------

class TestUvProvisioner:
    def test_basic_commands(self):
        p = UvProvisioner()
        cmds = p.provision_commands("/tmp/work")
        assert len(cmds) == 3  # bootstrap + venv + install cloudpickle
        assert "install.sh" in cmds[0]  # uv bootstrap
        assert "uv venv" in cmds[1]
        assert "--python 3.12" in cmds[1]
        assert "cloudpickle" in cmds[2]

    def test_with_requirements(self):
        p = UvProvisioner(requirements="requirements.txt")
        cmds = p.provision_commands("/tmp/work")
        assert len(cmds) == 4  # bootstrap + venv + requirements + cloudpickle
        assert "-r requirements.txt" in cmds[2]

    def test_extra_packages(self):
        p = UvProvisioner(extra_packages=["numpy", "pandas"])
        cmds = p.provision_commands("/tmp/work")
        last = cmds[-1]
        assert "cloudpickle" in last
        assert "numpy" in last
        assert "pandas" in last

    def test_custom_venv_path(self):
        p = UvProvisioner(venv_path="myenv")
        cmds = p.provision_commands("/tmp/work")
        assert "/tmp/work/myenv" in cmds[1]


# ---------------------------------------------------------------------------
# PipProvisioner
# ---------------------------------------------------------------------------

class TestPipProvisioner:
    def test_basic_commands(self):
        p = PipProvisioner()
        cmds = p.provision_commands("/tmp/work")
        assert len(cmds) == 2  # venv + install cloudpickle
        assert "python3 -m venv" in cmds[0]
        assert "cloudpickle" in cmds[1]

    def test_with_requirements(self):
        p = PipProvisioner(requirements="reqs.txt")
        cmds = p.provision_commands("/tmp/work")
        assert len(cmds) == 3
        assert "-r reqs.txt" in cmds[1]

    def test_extra_packages(self):
        p = PipProvisioner(extra_packages=["torch"])
        cmds = p.provision_commands("/tmp/work")
        assert "torch" in cmds[-1]
        assert "cloudpickle" in cmds[-1]

    def test_custom_python(self):
        p = PipProvisioner(python="python3.11")
        cmds = p.provision_commands("/tmp/work")
        assert "python3.11 -m venv" in cmds[0]


# ---------------------------------------------------------------------------
# PixiProvisioner
# ---------------------------------------------------------------------------

class TestPixiProvisioner:
    def test_basic_commands(self):
        p = PixiProvisioner()
        cmds = p.provision_commands("/tmp/work")
        assert len(cmds) == 2  # install + add cloudpickle
        assert "pixi install" in cmds[0]
        assert "pixi add" in cmds[1]
        assert "cloudpickle" in cmds[1]

    def test_with_manifest(self):
        p = PixiProvisioner(manifest="pixi.toml")
        cmds = p.provision_commands("/tmp/work")
        assert "--manifest-path pixi.toml" in cmds[0]

    def test_with_environment(self):
        p = PixiProvisioner(environment="gpu")
        cmds = p.provision_commands("/tmp/work")
        assert "-e gpu" in cmds[0]
        assert "-e gpu" in cmds[1]

    def test_extra_packages(self):
        p = PixiProvisioner(extra_packages=["numpy"])
        cmds = p.provision_commands("/tmp/work")
        assert "numpy" in cmds[-1]


# ---------------------------------------------------------------------------
# SshExecutor construction & validation
# ---------------------------------------------------------------------------

class TestSshExecutorInit:
    def test_empty_hosts_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            SshExecutor(hosts=[])

    def test_ssh_in_executors_registry(self):
        assert "ssh" in EXECUTORS
        assert EXECUTORS["ssh"] is SshExecutor

    def test_string_hosts_coerced(self):
        executor = SshExecutor(hosts=["user@gpu1", "user@gpu2"])
        assert len(executor._hosts) == 2
        assert all(isinstance(h, SshHost) for h in executor._hosts)
        assert executor._hosts[0].host == "user@gpu1"
        assert executor._hosts[1].host == "user@gpu2"
        assert executor._hosts[0].max_tasks == 1  # default
        executor.shutdown(wait=False)

    def test_mixed_hosts(self):
        executor = SshExecutor(hosts=["user@gpu1", SshHost("user@gpu2", max_tasks=4)])
        assert executor._hosts[0].host == "user@gpu1"
        assert executor._hosts[0].max_tasks == 1
        assert executor._hosts[1].host == "user@gpu2"
        assert executor._hosts[1].max_tasks == 4
        executor.shutdown(wait=False)

    def test_provision_auto_is_default(self):
        executor = SshExecutor(hosts=["user@host1"])
        assert isinstance(executor._provision, AutoProvisioner)
        executor.shutdown(wait=False)

    def test_provision_none(self):
        executor = SshExecutor(hosts=["user@host1"], provision=None)
        assert executor._provision is None
        executor.shutdown(wait=False)

    def test_provision_explicit(self):
        prov = UvProvisioner()
        executor = SshExecutor(hosts=["user@host1"], provision=prov)
        assert executor._provision is prov
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

class TestDetectProvisioner:
    def test_uv_lock(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, UvProvisioner)
        assert p.python == "3.12"
        assert p.requirements is None

    def test_uv_lock_with_requirements(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        (tmp_path / "requirements.txt").write_text("numpy\n")
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, UvProvisioner)
        assert p.requirements == "requirements.txt"

    def test_uv_lock_with_python_version(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        (tmp_path / ".python-version").write_text("3.11\n")
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, UvProvisioner)
        assert p.python == "3.11"

    def test_pixi_lock(self, tmp_path):
        (tmp_path / "pixi.lock").touch()
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, PixiProvisioner)

    def test_pixi_toml(self, tmp_path):
        (tmp_path / "pixi.toml").touch()
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, PixiProvisioner)
        assert p.manifest == "pixi.toml"

    def test_requirements_txt(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\n")
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, PipProvisioner)
        assert p.requirements == "requirements.txt"

    def test_pyproject_toml_fallback(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, UvProvisioner)

    def test_bare_fallback(self, tmp_path):
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, PipProvisioner)
        assert p.requirements is None

    def test_uv_takes_priority_over_requirements(self, tmp_path):
        """uv.lock wins even when requirements.txt also exists."""
        (tmp_path / "uv.lock").touch()
        (tmp_path / "requirements.txt").write_text("numpy\n")
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, UvProvisioner)

    def test_pixi_takes_priority_over_requirements(self, tmp_path):
        (tmp_path / "pixi.lock").touch()
        (tmp_path / "requirements.txt").write_text("numpy\n")
        p = _detect_provisioner(tmp_path)
        assert isinstance(p, PixiProvisioner)


class TestAutoProvisioner:
    def test_delegates_to_detected_uv(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        ap = AutoProvisioner(search_dir=tmp_path)
        cmds = ap.provision_commands("/remote/work")
        assert any("uv venv" in c for c in cmds)

    def test_delegates_to_detected_pixi(self, tmp_path):
        (tmp_path / "pixi.toml").touch()
        ap = AutoProvisioner(search_dir=tmp_path)
        cmds = ap.provision_commands("/remote/work")
        assert any("pixi install" in c for c in cmds)

    def test_requirements_attr(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("numpy\n")
        ap = AutoProvisioner(search_dir=tmp_path)
        assert ap.requirements == "requirements.txt"

    def test_manifest_attr_pixi(self, tmp_path):
        (tmp_path / "pixi.toml").touch()
        ap = AutoProvisioner(search_dir=tmp_path)
        assert ap.manifest == "pixi.toml"

    def test_manifest_attr_none_for_uv(self, tmp_path):
        (tmp_path / "uv.lock").touch()
        ap = AutoProvisioner(search_dir=tmp_path)
        assert ap.manifest is None

    def test_lazy_detection(self, tmp_path):
        ap = AutoProvisioner(search_dir=tmp_path)
        assert ap._delegate is None  # not yet detected
        ap.provision_commands("/work")
        assert ap._delegate is not None  # now detected


# ---------------------------------------------------------------------------
# SSH command builders
# ---------------------------------------------------------------------------

class TestSshCommandBuilders:
    def setup_method(self):
        self.executor = SshExecutor(
            hosts=[SshHost("user@host1")],
            ssh_options=["-o", "StrictHostKeyChecking=no"],
        )

    def test_ssh_cmd(self):
        cmd = self.executor._ssh_cmd("user@host1", "ls -la")
        assert cmd == ["ssh", "-o", "StrictHostKeyChecking=no", "user@host1", "ls -la"]

    def test_scp_to(self):
        cmd = self.executor._scp_to("/tmp/file.pkl", "user@host1", "/remote/file.pkl")
        assert cmd == ["scp", "-o", "StrictHostKeyChecking=no", "/tmp/file.pkl", "user@host1:/remote/file.pkl"]

    def test_scp_from(self):
        cmd = self.executor._scp_from("user@host1", "/remote/result.pkl", "/tmp/result.pkl")
        assert cmd == ["scp", "-o", "StrictHostKeyChecking=no", "user@host1:/remote/result.pkl", "/tmp/result.pkl"]

    def test_no_ssh_options(self):
        executor = SshExecutor(hosts=[SshHost("user@host1")])
        cmd = executor._ssh_cmd("user@host1", "echo hi")
        assert cmd == ["ssh", "user@host1", "echo hi"]

    def teardown_method(self):
        self.executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Provisioning idempotency
# ---------------------------------------------------------------------------

class TestProvisionHost:
    def test_provision_runs_once_per_host(self):
        host = SshHost("user@host1")
        executor = SshExecutor(hosts=[host], provision=None)

        with patch.object(executor, "_run_ssh") as mock_ssh, \
             patch.object(executor, "_run_scp") as mock_scp:
            mock_ssh.return_value = MagicMock(returncode=0, stdout="")

            executor._provision_host(host)
            executor._provision_host(host)
            executor._provision_host(host)

            # mkdir is called once (only on first provision)
            assert mock_ssh.call_count == 1
            mock_scp.assert_not_called()

        executor.shutdown(wait=False)

    def test_provision_with_provisioner(self):
        host = SshHost("user@host1")
        prov = UvProvisioner()
        executor = SshExecutor(hosts=[host], provision=prov)

        with patch.object(executor, "_run_ssh") as mock_ssh, \
             patch.object(executor, "_run_scp") as mock_scp:
            mock_ssh.return_value = MagicMock(returncode=0, stdout="")
            mock_scp.return_value = MagicMock(returncode=0, stdout="")

            executor._provision_host(host)

            # mkdir + provision commands (bootstrap + venv + install)
            assert mock_ssh.call_count == 4  # mkdir + 3 provision cmds

        executor.shutdown(wait=False)

    def test_provision_threadsafe(self):
        host = SshHost("user@host1")
        executor = SshExecutor(hosts=[host], provision=None)

        call_count = 0
        original_run_ssh = executor._run_ssh

        def counting_run_ssh(host_str, cmd, **kwargs):
            nonlocal call_count
            call_count += 1
            return MagicMock(returncode=0, stdout="")

        with patch.object(executor, "_run_ssh", side_effect=counting_run_ssh):
            threads = [threading.Thread(target=executor._provision_host, args=(host,)) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Should only provision once despite concurrent calls
            assert call_count == 1

        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Host selection
# ---------------------------------------------------------------------------

class TestHostSelection:
    def test_work_dir_override(self):
        host = SshHost("user@h1", work_dir="/custom/dir")
        executor = SshExecutor(hosts=[host], work_dir="/default/dir")
        assert executor._work_dir_for(host) == "/custom/dir"
        executor.shutdown(wait=False)

    def test_work_dir_default(self):
        host = SshHost("user@h1")
        executor = SshExecutor(hosts=[host], work_dir="/default/dir")
        assert executor._work_dir_for(host) == "/default/dir"
        executor.shutdown(wait=False)

    def test_setup_override(self):
        host = SshHost("user@h1", setup="source ~/myenv/bin/activate")
        executor = SshExecutor(hosts=[host], setup="source ~/default/bin/activate")
        assert executor._setup_for(host) == "source ~/myenv/bin/activate"
        executor.shutdown(wait=False)

    def test_setup_default(self):
        host = SshHost("user@h1")
        executor = SshExecutor(hosts=[host], setup="source ~/default/bin/activate")
        assert executor._setup_for(host) == "source ~/default/bin/activate"
        executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Integration tests — require localhost SSH
# ---------------------------------------------------------------------------

def _can_ssh_localhost():
    """Check if we can SSH to localhost without a password."""
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
             "localhost", "echo ok"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=5, text=True,
        )
        return result.returncode == 0 and "ok" in result.stdout
    except Exception:
        return False


requires_ssh = pytest.mark.skipif(
    not _can_ssh_localhost(),
    reason="Cannot SSH to localhost (no key-based auth configured)",
)


@requires_ssh
class TestSshExecutorIntegration:
    def test_basic_submit(self):
        executor = SshExecutor(
            hosts=[SshHost("localhost", max_tasks=2)],
            ssh_options=["-o", "StrictHostKeyChecking=no"],
        )
        try:
            future = executor.submit(lambda x: x * 2, 21)
            assert future.result(timeout=30) == 42
        finally:
            executor.shutdown()

    def test_error_captures_traceback(self):
        executor = SshExecutor(
            hosts=[SshHost("localhost")],
            ssh_options=["-o", "StrictHostKeyChecking=no"],
        )
        try:
            def bad_fn():
                raise ValueError("test error")

            future = executor.submit(bad_fn)
            with pytest.raises(RuntimeError):
                future.result(timeout=30)
            assert "ValueError" in future.log or "test error" in future.log
        finally:
            executor.shutdown()

    def test_output_capture(self):
        executor = SshExecutor(
            hosts=[SshHost("localhost")],
            ssh_options=["-o", "StrictHostKeyChecking=no"],
        )
        try:
            def chatty_fn():
                print("hello from ssh")
                return 99

            future = executor.submit(chatty_fn)
            assert future.result(timeout=30) == 99
            assert "hello from ssh" in future.log
        finally:
            executor.shutdown()

    def test_concurrent_execution(self):
        import time

        executor = SshExecutor(
            hosts=[SshHost("localhost", max_tasks=3)],
            ssh_options=["-o", "StrictHostKeyChecking=no"],
        )
        try:
            def slow_fn(x):
                import time
                time.sleep(0.5)
                return x

            futures = [executor.submit(slow_fn, i) for i in range(3)]
            start = time.time()
            results = [f.result(timeout=30) for f in futures]
            elapsed = time.time() - start

            assert sorted(results) == [0, 1, 2]
            # With max_tasks=3, all 3 should run concurrently (~0.5s not ~1.5s)
            assert elapsed < 2.0
        finally:
            executor.shutdown()
