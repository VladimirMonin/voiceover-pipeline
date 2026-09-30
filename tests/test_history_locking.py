"""Contract tests for the interprocess run lock (``history.locking``).

Every fixture is synthetic and lives under ``tmp_path``: no real ``VOICEOVER_HOME``,
no ``.env``, no provider, model, database, or network path is touched. The lock
contract is exercised fail-first: a relative run root is rejected before the home
is created, a second process holding the same canonical root gets
:class:`HistoryRunLockedError`, a different root is independent, and an insecure
or symlinked home or lock file fails closed instead of being relaxed.

Case-sensitive and case-insensitive volumes are distinguished through the
module's volume probe, which tests replace on any host; the Darwin probe itself
is additionally exercised natively against the real ``getattrlist`` answer of
the temporary directory. The Windows backend is exercised only through a mocked
``msvcrt``; no native Windows execution is claimed here.

Cross-process coordination is deterministic: a child process acquires the lock,
prints a readiness marker, and holds it until the parent closes its stdin. The
parent waits for that marker with a bounded ``select`` timeout instead of a sleep,
and kills and reaps a holder that misses that timeout before reading its output.
"""

import errno
import os
import select
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from voiceover_pipeline.history import locking
from voiceover_pipeline.history.locking import (
    LOCK_FILE_MODE,
    LOCKS_DIR_NAME,
    HistoryRunLockedError,
    HistoryRunLockError,
    acquire_run_lock,
    run_lock_path,
)
from voiceover_pipeline.history.paths import (
    HistoryPathsError,
    ensure_history_home,
    ensure_private_directory,
    history_database_path,
)

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
_HOLDER_TIMEOUT = 15.0
# Readiness timeout shortened for the test that forces a timeout on purpose.
_READINESS_TIMEOUT = 0.5

_CHILD_HOLDER = (
    "import sys\n"
    "from voiceover_pipeline.history.locking import acquire_run_lock\n"
    "run_root, home = sys.argv[1], sys.argv[2]\n"
    "with acquire_run_lock(run_root, home=home):\n"
    "    print('LOCKED', flush=True)\n"
    "    sys.stdin.readline()\n"
)

# A holder that writes to stderr and then blocks forever without ever printing the
# readiness marker, so a readiness timeout leaves it alive with stderr open.
_STUCK_HOLDER = (
    "import sys\nprint('holder-stuck', file=sys.stderr, flush=True)\nsys.stdin.readline()\n"
)


def _release_lock_holder(child: subprocess.Popen[str]) -> None:
    if child.stdin is not None:
        try:
            child.stdin.write("release\n")
            child.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        finally:
            try:
                child.stdin.close()
            except OSError:
                pass
    try:
        child.wait(timeout=_HOLDER_TIMEOUT)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=_HOLDER_TIMEOUT)


def _reap_lock_holder(child: subprocess.Popen[str]) -> None:
    """Kill and reap a holder before anything is read from its pipes.

    A holder that missed the readiness timeout may still be alive with stderr
    open, so reading it first would block indefinitely. Killing it and reaping
    it with a bounded wait makes the later read terminate instead of hanging.
    """
    if child.poll() is None:
        child.kill()
    try:
        child.wait(timeout=_HOLDER_TIMEOUT)
    except subprocess.TimeoutExpired:  # pragma: no cover - kill cannot be ignored
        raise AssertionError("lock holder was killed but not reaped") from None


def _start_lock_holder(
    run_root: Path,
    home: Path,
    *,
    program: str = _CHILD_HOLDER,
    timeout: float = _HOLDER_TIMEOUT,
) -> subprocess.Popen[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    child = subprocess.Popen(
        [sys.executable, "-c", program, str(run_root), str(home)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert child.stdout is not None
    ready, _, _ = select.select([child.stdout], [], [], timeout)
    marker = child.stdout.readline() if ready else ""
    if marker.strip() != "LOCKED":
        # The child may still be alive with stderr open, so it is killed and
        # reaped first; only then is reading its stderr bounded.
        _reap_lock_holder(child)
        stderr = child.stderr.read() if child.stderr is not None else ""
        raise AssertionError(f"lock holder did not report readiness: {marker!r} {stderr!r}")
    return child


def test_relative_run_root_is_rejected_before_creating_home(tmp_path):
    home = tmp_path / "home"

    with pytest.raises(HistoryRunLockError):
        with acquire_run_lock("relative-run-root", home=home):
            pass

    assert not home.exists()
    assert list(tmp_path.iterdir()) == []


def test_acquire_creates_private_layout_and_locks_file(tmp_path):
    home = tmp_path / "home"
    run_root = tmp_path / "run-root"

    with acquire_run_lock(run_root, home=home) as lock_path:
        assert lock_path.is_file()
        assert lock_path.parent == home / LOCKS_DIR_NAME
        assert stat.S_IMODE(lock_path.stat().st_mode) == LOCK_FILE_MODE

    assert stat.S_IMODE(home.stat().st_mode) & 0o077 == 0


def test_lock_filename_is_opaque_and_stable(tmp_path):
    home = tmp_path / "home"

    lock_path = run_lock_path(tmp_path / "secret-label-root", home=home)

    assert lock_path.parent == home / LOCKS_DIR_NAME
    assert lock_path.name.endswith(".lock")
    assert "secret" not in lock_path.name
    assert len(lock_path.stem) == 64
    assert run_lock_path(tmp_path / "secret-label-root", home=home) == lock_path


def test_same_root_aliases_map_to_same_lock(tmp_path):
    home = tmp_path / "home"
    base = tmp_path / "runs" / "abc"

    assert run_lock_path(base, home=home) == run_lock_path(f"{base}/", home=home)
    dotdot = tmp_path / "runs" / "other" / ".." / "abc"
    assert run_lock_path(base, home=home) == run_lock_path(dotdot, home=home)

    real = tmp_path / "real-root"
    real.mkdir()
    link = tmp_path / "linked-root"
    os.symlink(real, link)
    assert run_lock_path(real, home=home) == run_lock_path(link, home=home)


def test_different_roots_get_distinct_locks(tmp_path):
    home = tmp_path / "home"
    root_a = tmp_path / "root-a"
    root_b = tmp_path / "root-b"

    assert run_lock_path(root_a, home=home) != run_lock_path(root_b, home=home)
    with acquire_run_lock(root_a, home=home):
        with acquire_run_lock(root_b, home=home) as lock_path:
            assert lock_path.is_file()


def test_lock_key_material_prefixes_the_anchor_device(tmp_path):
    anchor, device = locking._anchor_device(tmp_path / "missing" / "run-root")

    assert anchor == tmp_path
    assert device == os.stat(tmp_path).st_dev
    assert locking._lock_key_material(tmp_path / "missing" / "run-root").startswith(
        os.fsencode(f"{device}\0")
    )


def test_case_insensitive_volume_folds_case_and_unicode_aliases(monkeypatch, tmp_path):
    home = tmp_path / "home"
    lower = tmp_path / "run-root"
    lower.mkdir()
    monkeypatch.setattr(locking, "_volume_case_sensitivity", lambda path: True)

    assert run_lock_path(tmp_path / "RUN-ROOT", home=home) == run_lock_path(lower, home=home)
    composed = tmp_path / "caf\u00e9"
    decomposed = tmp_path / "cafe\u0301"
    assert run_lock_path(composed, home=home) == run_lock_path(decomposed, home=home)


def test_case_insensitive_alias_contends_for_one_lock(monkeypatch, tmp_path):
    home = tmp_path / "home"
    lower = tmp_path / "run-root"
    lower.mkdir()
    monkeypatch.setattr(locking, "_volume_case_sensitivity", lambda path: True)

    with acquire_run_lock(lower, home=home):
        with pytest.raises(HistoryRunLockedError):
            with acquire_run_lock(tmp_path / "RUN-ROOT", home=home):
                pass


def test_case_sensitive_volume_keeps_distinct_roots_distinct(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setattr(locking, "_volume_case_sensitivity", lambda path: False)

    upper = tmp_path / "MiXeD"
    lower = tmp_path / "mixed"

    assert run_lock_path(upper, home=home) != run_lock_path(lower, home=home)
    with acquire_run_lock(lower, home=home):
        with acquire_run_lock(upper, home=home) as lock_path:
            assert lock_path.is_file()


def test_undeterminable_case_sensitivity_is_refused(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setattr(locking, "_volume_case_sensitivity", lambda path: None)

    with pytest.raises(HistoryRunLockError):
        run_lock_path(tmp_path / "run-root", home=home)

    assert not home.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin getattrlist volume probe")
def test_darwin_probe_matches_real_volume_case_behavior(tmp_path):
    probe = tmp_path / "Case-Probe"
    probe.mkdir()
    alias = tmp_path / "case-probe"
    real_alias = alias.exists() and os.path.samefile(alias, probe)

    assert locking._volume_case_sensitivity(tmp_path) is real_alias

    home = tmp_path / "home"
    if real_alias:
        assert run_lock_path(probe, home=home) == run_lock_path(alias, home=home)
    else:
        assert run_lock_path(probe, home=home) != run_lock_path(alias, home=home)


def test_lock_is_reacquirable_after_normal_exit(tmp_path):
    home = tmp_path / "home"
    run_root = tmp_path / "run-root"

    with acquire_run_lock(run_root, home=home):
        pass
    with acquire_run_lock(run_root, home=home):
        pass


def test_lock_is_released_after_exception(tmp_path):
    home = tmp_path / "home"
    run_root = tmp_path / "run-root"

    with pytest.raises(ValueError):
        with acquire_run_lock(run_root, home=home):
            raise ValueError("boom")

    with acquire_run_lock(run_root, home=home):
        pass


def test_lock_file_is_persisted_after_release(tmp_path):
    home = tmp_path / "home"
    run_root = tmp_path / "run-root"

    with acquire_run_lock(run_root, home=home) as lock_path:
        pass

    assert lock_path.is_file()


def test_lock_touches_only_private_layout_not_database_or_output(tmp_path):
    home = tmp_path / "home"
    run_root = tmp_path / "untouched-output"

    with acquire_run_lock(run_root, home=home) as lock_path:
        assert lock_path.is_file()

    assert not history_database_path(home).exists()
    assert not run_root.exists()
    assert [path.name for path in (home / LOCKS_DIR_NAME).iterdir()] == [lock_path.name]


def test_insecure_home_is_rejected_without_chmodding(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    home.chmod(0o755)

    with pytest.raises(HistoryPathsError):
        with acquire_run_lock(tmp_path / "run-root", home=home):
            pass

    assert stat.S_IMODE(home.stat().st_mode) == 0o755


def test_symlinked_home_is_rejected(tmp_path):
    real = tmp_path / "real-home"
    real.mkdir(mode=0o700)
    link = tmp_path / "home-link"
    os.symlink(real, link)

    with pytest.raises(HistoryPathsError):
        with acquire_run_lock(tmp_path / "run-root", home=link):
            pass


def test_insecure_existing_lock_file_is_rejected_not_chmodded(tmp_path):
    home = tmp_path / "home"
    ensure_history_home(home)
    ensure_private_directory(home / LOCKS_DIR_NAME)
    run_root = tmp_path / "run-root"
    lock_path = run_lock_path(run_root, home=home)
    lock_path.write_text("", encoding="utf-8")
    lock_path.chmod(0o644)

    with pytest.raises(HistoryRunLockError):
        with acquire_run_lock(run_root, home=home):
            pass

    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o644


def test_symlinked_lock_file_is_rejected_not_followed(tmp_path):
    home = tmp_path / "home"
    ensure_history_home(home)
    ensure_private_directory(home / LOCKS_DIR_NAME)
    run_root = tmp_path / "run-root"
    lock_path = run_lock_path(run_root, home=home)
    target = tmp_path / "target"
    target.write_text("", encoding="utf-8")
    target.chmod(0o600)
    os.symlink(target, lock_path)

    with pytest.raises(HistoryRunLockError):
        with acquire_run_lock(run_root, home=home):
            pass

    assert lock_path.is_symlink()
    assert target.read_text(encoding="utf-8") == ""


def test_verify_lock_identity_rejects_replaced_inode(tmp_path):
    first = tmp_path / "first.lock"
    second = tmp_path / "second.lock"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")
    fd = os.open(first, os.O_RDWR)
    try:
        locking._verify_lock_identity(fd, first)
        with pytest.raises(HistoryRunLockError):
            locking._verify_lock_identity(fd, second)
    finally:
        os.close(fd)


@pytest.mark.platform_simulated
def test_windows_backend_fails_fast_on_contention_and_acquires_when_free(monkeypatch, tmp_path):
    attempts: list[int] = []

    class FakeMsvcrt:
        LK_NBLCK = 2
        LK_UNLCK = 3

        def __init__(self, *, busy: bool) -> None:
            self._busy = busy

        def locking(self, fd: int, op: int, nbytes: int) -> None:
            if op == self.LK_UNLCK:
                return
            attempts.append(op)
            if self._busy:
                raise OSError(errno.EACCES, "locked by another process")

    fd = os.open(tmp_path / "fake.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        monkeypatch.setattr(locking, "msvcrt", FakeMsvcrt(busy=True), raising=False)
        assert locking._acquire_exclusive_windows(fd) is False

        monkeypatch.setattr(locking, "msvcrt", FakeMsvcrt(busy=False), raising=False)
        assert locking._acquire_exclusive_windows(fd) is True
    finally:
        os.close(fd)

    assert attempts == [FakeMsvcrt.LK_NBLCK, FakeMsvcrt.LK_NBLCK]


def test_posix_backend_requests_exclusive_nonblocking_lock(monkeypatch, tmp_path):
    calls: list[int] = []

    class FakeFcntl:
        LOCK_EX = 1
        LOCK_NB = 4
        LOCK_UN = 8

        def flock(self, fd: int, op: int) -> None:
            calls.append(op)

    monkeypatch.setattr(locking, "_fcntl", FakeFcntl(), raising=False)
    fd = os.open(tmp_path / "fake.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert locking._acquire_exclusive_posix(fd) is True
        locking._release_lock_posix(fd)
    finally:
        os.close(fd)

    assert calls == [FakeFcntl.LOCK_EX | FakeFcntl.LOCK_NB, FakeFcntl.LOCK_UN]


def test_posix_backend_reports_contention(monkeypatch, tmp_path):
    class FakeFcntl:
        LOCK_EX = 1
        LOCK_NB = 4

        def flock(self, fd: int, op: int) -> None:
            raise OSError(errno.EAGAIN, "resource temporarily unavailable")

    monkeypatch.setattr(locking, "_fcntl", FakeFcntl(), raising=False)
    fd = os.open(tmp_path / "fake.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        assert locking._acquire_exclusive_posix(fd) is False
    finally:
        os.close(fd)


def test_readiness_timeout_reaps_holder_before_reading_stderr(monkeypatch, tmp_path):
    started: list[subprocess.Popen[str]] = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        started.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", recording_popen)

    with pytest.raises(AssertionError) as failure:
        _start_lock_holder(
            tmp_path / "run-root",
            tmp_path / "home",
            program=_STUCK_HOLDER,
            timeout=_READINESS_TIMEOUT,
        )

    assert len(started) == 1
    child = started[0]
    # The stuck holder blocked on stdin with stderr still open, so this call only
    # returns because the helper killed and reaped it before reading: an unbounded
    # read of that live pipe would never return.
    assert child.poll() is not None
    assert "holder-stuck" in str(failure.value)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX advisory locks only")
def test_cross_process_same_root_contention_fails_fast(tmp_path):
    home = tmp_path / "home"
    run_root = tmp_path / "run-root"

    child = _start_lock_holder(run_root, home)
    try:
        with pytest.raises(HistoryRunLockedError):
            with acquire_run_lock(run_root, home=home):
                pass
    finally:
        _release_lock_holder(child)

    assert child.returncode == 0
    with acquire_run_lock(run_root, home=home) as lock_path:
        assert lock_path.is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX advisory locks only")
def test_cross_process_different_roots_are_independent(tmp_path):
    home = tmp_path / "home"

    child = _start_lock_holder(tmp_path / "root-a", home)
    try:
        with acquire_run_lock(tmp_path / "root-b", home=home) as lock_path:
            assert lock_path.is_file()
    finally:
        _release_lock_holder(child)

    assert child.returncode == 0
