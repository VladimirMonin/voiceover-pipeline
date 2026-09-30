"""One interprocess run lock per canonical run root (plan section 6).

Plan section 6 keeps one small local interprocess lock so a second writer never
executes the same run while the old writer is still alive. This module owns that
lock and nothing else:

* The run root is resolved to an absolute canonical path first, so the lock is
  never silently relative to the current working directory. A relative
  ``run_root`` is rejected with :class:`HistoryRunLockError` before any directory
  is created.
* The canonical path is hashed to an opaque ``<sha256>.lock`` file name under the
  private ``<VOICEOVER_HOME>/locks/`` directory. No user label, run id, or part of
  the path appears in the file name, so one canonical run root maps to exactly
  one lock and two spellings of the same root map to the same lock.
* Name folding is volume-aware, because ``os.path.realpath`` preserves the
  spelling it was given on a case-insensitive filesystem, where two spellings
  name one directory. The volume is probed read-only on Darwin with
  ``getattrlist(ATTR_VOL_CAPABILITIES)`` and the key is then case-folded and
  Unicode-normalized, so differently capitalized or differently composed
  spellings of one run root share one lock. On a case-sensitive volume the
  spelling is preserved so two roots that differ only in case keep two locks. A
  volume that does not report a usable case capability raises
  :class:`HistoryRunLockError` instead of guessing, because guessing wrong admits
  a second writer for one run root. The nearest existing ancestor's device number
  prefixes the key, so two volumes mounted at look-alike paths never share a lock.
  Windows is case-insensitive by default and uses ``os.path.normcase``;
  per-directory Windows case sensitivity is not detected. The Darwin probe is
  exercised natively on macOS and mocked elsewhere; the Windows path is only
  exercised through a mocked backend here.
* The lock is a real OS advisory lock: ``fcntl.flock(LOCK_EX | LOCK_NB)`` on POSIX
  and ``msvcrt.locking(LK_NBLCK)`` on Windows. Contention fails fast with
  :class:`HistoryRunLockedError`; there is no stale-PID guess, no waiting, and no
  unlink/replacement race, because the lock file is opened once, verified by its
  device/inode identity, and never removed. The kernel releases the lock when its
  descriptor closes, so ownership survives neither an exception nor process
  death.

The lock never opens, migrates, or writes the history database, never touches the
run's output tree, and never calls a provider, model, or network. The Windows
path is implemented but exercised here only through a mocked backend; running it
natively requires a real Windows host.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import stat
import struct
import sys
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from .paths import ensure_history_home, ensure_private_directory, history_home

# Private directory under the history home that holds only opaque lock files.
LOCKS_DIR_NAME = "locks"
# Lock files are private: group or world bits are rejected rather than relaxed.
LOCK_FILE_MODE = 0o600

# Darwin volume capability request, from ``sys/attr.h``: ATTR_VOL_INFO selects the
# volume attribute group, ATTR_VOL_CAPABILITIES returns ``vol_capabilities_attr_t``
# (capabilities[4] then valid[4]), and VOL_CAP_FMT_CASE_SENSITIVE is bit 8 of the
# format set. getattrlist returns the u_int32 total length followed by that
# 32-byte attribute, so a shorter result cannot hold what is read below.
_DARWIN_ATTR_BIT_MAP_COUNT = 5
_DARWIN_ATTR_VOL_INFO = 0x80000000
_DARWIN_ATTR_VOL_CAPABILITIES = 0x00020000
_DARWIN_VOL_CAP_FMT_CASE_SENSITIVE = 0x00000100
_DARWIN_CAPABILITIES_RESULT_SIZE = 36

if sys.platform == "win32":
    import msvcrt

    _fcntl: Any = None
else:
    import fcntl

    _fcntl = fcntl
    msvcrt: Any = None


class HistoryRunLockError(RuntimeError):
    """Base class for run-lock contract violations."""


class HistoryRunLockedError(HistoryRunLockError):
    """Another process already holds the lock for this run root."""


class _DarwinAttrList(ctypes.Structure):
    """``struct attrlist``: two u_int16 fields then five attrgroup_t fields."""

    _fields_ = (
        ("bitmapcount", ctypes.c_ushort),
        ("reserved", ctypes.c_uint16),
        ("commonattr", ctypes.c_uint32),
        ("volattr", ctypes.c_uint32),
        ("dirattr", ctypes.c_uint32),
        ("fileattr", ctypes.c_uint32),
        ("forkattr", ctypes.c_uint32),
    )


def _darwin_getattrlist() -> Any:
    """Return a bound libSystem ``getattrlist``, or None when unavailable."""
    try:
        libc: Any = ctypes.CDLL(None, use_errno=True)
        getattrlist: Any = libc.getattrlist
    except (OSError, AttributeError):
        return None
    getattrlist.argtypes = (
        ctypes.c_char_p,
        ctypes.POINTER(_DarwinAttrList),
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_uint,
    )
    getattrlist.restype = ctypes.c_int
    return getattrlist


def _darwin_case_insensitive(path: Path) -> bool | None:
    """Report whether ``path`` is on a case-insensitive volume, or None.

    One read-only ``getattrlist`` call against an existing path asks the volume
    for ATTR_VOL_CAPABILITIES. None means the volume did not answer with a usable
    format capability, so the caller must not guess the answer.
    """
    getattrlist = _darwin_getattrlist()
    if getattrlist is None:
        return None
    attributes = _DarwinAttrList(
        bitmapcount=_DARWIN_ATTR_BIT_MAP_COUNT,
        reserved=0,
        commonattr=0,
        volattr=_DARWIN_ATTR_VOL_INFO | _DARWIN_ATTR_VOL_CAPABILITIES,
        dirattr=0,
        fileattr=0,
        forkattr=0,
    )
    buffer = ctypes.create_string_buffer(_DARWIN_CAPABILITIES_RESULT_SIZE)
    try:
        result = getattrlist(
            os.fsencode(path), ctypes.byref(attributes), buffer, ctypes.sizeof(buffer), 0
        )
    except OSError:
        return None
    if result != 0:
        return None
    raw = buffer.raw
    if struct.unpack_from("<I", raw, 0)[0] < _DARWIN_CAPABILITIES_RESULT_SIZE:
        return None
    capabilities = struct.unpack_from("<4I", raw, 4)
    valid = struct.unpack_from("<4I", raw, 20)
    if not valid[0] & _DARWIN_VOL_CAP_FMT_CASE_SENSITIVE:
        return None
    return not capabilities[0] & _DARWIN_VOL_CAP_FMT_CASE_SENSITIVE


def _volume_case_sensitivity(path: Path) -> bool | None:
    """Report whether ``path`` is on a case-insensitive volume, or None.

    Only Darwin needs the probe: other POSIX platforms compare names byte-wise,
    so the spelling already identifies the directory and folding would wrongly
    merge distinct roots. None means the answer could not be determined.
    """
    if sys.platform != "darwin":
        return False
    return _darwin_case_insensitive(path)


def _anchor_device(path: Path) -> tuple[Path, int | None]:
    """Return the nearest existing ancestor of ``path`` and its device number.

    Case sensitivity is a volume property, so the nearest existing ancestor
    answers it for a run root whose own directory does not exist yet. The device
    number is None only when no ancestor can be stat'd at all.
    """
    candidate = path
    while True:
        try:
            return candidate, os.stat(candidate).st_dev
        except OSError:
            parent = candidate.parent
            if parent == candidate:
                return candidate, None
            candidate = parent


def _lock_key_text(canonical: Path, *, anchor: Path) -> str:
    """Return the path text hashed into the lock name for ``canonical``.

    The spelling is preserved unless the volume itself folds name comparison.
    Case and Unicode canonical folding are applied only when the probed volume
    reports case-insensitive lookup, so two spellings of one directory cannot
    mint two locks while two genuinely distinct roots on a case-sensitive volume
    stay distinct. Windows is case-insensitive by default and uses
    ``os.path.normcase`` instead. An undeterminable volume is refused rather than
    guessed, because guessing wrong admits a second writer for one run root.
    """
    if sys.platform == "win32":
        # Per-directory Windows case sensitivity is not detected here; the
        # default whole-volume behavior is assumed.
        return os.path.normcase(str(canonical))
    insensitive = _volume_case_sensitivity(anchor)
    if insensitive is None:
        raise HistoryRunLockError(
            f"cannot determine the case sensitivity of the volume holding {anchor}; "
            "refusing to create a run lock that may alias another run root"
        )
    if insensitive:
        # The volume compares canonically equivalent names as equal, so the
        # key is folded and normalized to its canonical form before hashing.
        return unicodedata.normalize("NFC", str(canonical).casefold())
    return str(canonical)


def _lock_key_material(canonical: Path) -> bytes:
    """Return the hashed key input: anchor device, then the run-root key text.

    Prefixing the nearest existing ancestor's device number means two volumes
    mounted at look-alike paths can never collide on one lock, even if their path
    text compares equal after folding.
    """
    anchor, device = _anchor_device(canonical)
    text = _lock_key_text(canonical, anchor=anchor)
    prefix = f"{device}\0" if device is not None else ""
    return os.fsencode(prefix + text)


def _canonical_run_root(run_root: Path | str) -> Path:
    """Return the absolute canonical run root, rejecting a relative input.

    A relative ``run_root`` would otherwise be interpreted against the current
    working directory, so it is refused instead of quietly resolved.
    ``os.path.realpath`` collapses ``..`` and redundant separators and resolves
    existing symlink prefixes. The spelling it returns is preserved, because only
    the lock key may fold spelling, and only where the volume itself does.
    """
    candidate = Path(run_root).expanduser()
    if not candidate.is_absolute():
        raise HistoryRunLockError(f"run root must be an absolute path, got {str(run_root)!r}")
    return Path(os.path.realpath(str(candidate)))


def run_lock_path(run_root: Path | str, *, home: Path | str | None = None) -> Path:
    """Return the canonical lock-file path for ``run_root`` without writing.

    The file name is the SHA-256 of the nearest existing ancestor's device plus
    the volume-normalized canonical run root, so it is stable across processes
    and carries no user label, run id, or secret path. Two spellings of one root
    on a case-insensitive volume share one digest; two distinct roots on a
    case-sensitive volume keep distinct digests. A volume whose case behavior
    cannot be determined raises :class:`HistoryRunLockError`.
    """
    canonical = _canonical_run_root(run_root)
    digest = hashlib.sha256(_lock_key_material(canonical)).hexdigest()
    return history_home(home) / LOCKS_DIR_NAME / f"{digest}.lock"


@contextmanager
def acquire_run_lock(run_root: Path | str, *, home: Path | str | None = None) -> Iterator[Path]:
    """Hold the exclusive cross-process lock for ``run_root`` for the block.

    The private home layout and its ``locks/`` directory are created or verified
    first, so an absent home is created privately while an existing insecure or
    symlinked home is rejected. Contention raises :class:`HistoryRunLockedError`
    immediately and the lock is released when the block ends, whether by normal
    exit or exception. The lock file itself is left in place on release so no
    another process can lock a replaced inode through a freed name.
    """
    lock_path = run_lock_path(run_root, home=home)
    ensure_history_home(home)
    ensure_private_directory(lock_path.parent)
    fd = _open_lock_file(lock_path)
    try:
        if not _acquire_exclusive(fd):
            raise HistoryRunLockedError(f"run root is already locked: {lock_path}")
        _verify_lock_identity(fd, lock_path)
        try:
            yield lock_path
        finally:
            _release_lock(fd)
    finally:
        os.close(fd)


def _open_lock_file(lock_path: Path) -> int:
    """Open or create the lock file as a private regular file, failing closed."""
    if lock_path.is_symlink():
        raise HistoryRunLockError(f"run lock path must not be a symlink: {lock_path}")
    create_flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        create_flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, create_flags, LOCK_FILE_MODE)
    except FileExistsError:
        return _open_existing_lock_file(lock_path)
    except OSError as exc:
        raise HistoryRunLockError(f"cannot create run lock file {lock_path}: {exc}") from exc
    try:
        # The O_CREAT mode is reduced by umask; set the private mode explicitly on
        # the file this process just created, not on a pre-existing user file.
        if hasattr(os, "fchmod"):
            os.fchmod(fd, LOCK_FILE_MODE)
        _require_private_lock_file(fd, lock_path)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _open_existing_lock_file(lock_path: Path) -> int:
    """Open an existing lock file without following a symlink or chmodding it."""
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(lock_path, flags)
    except OSError as exc:
        raise HistoryRunLockError(f"cannot open existing run lock file {lock_path}: {exc}") from exc
    try:
        _require_private_lock_file(fd, lock_path)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _require_private_lock_file(fd: int, lock_path: Path) -> None:
    try:
        status = os.fstat(fd)
    except OSError as exc:
        raise HistoryRunLockError(f"cannot stat run lock file {lock_path}: {exc}") from exc
    if not stat.S_ISREG(status.st_mode):
        raise HistoryRunLockError(f"run lock path is not a regular file: {lock_path}")
    # Windows does not expose meaningful POSIX mode bits, so only enforce the
    # group/world check where those bits describe real access.
    if os.name == "posix" and stat.S_IMODE(status.st_mode) & 0o077:
        raise HistoryRunLockError(
            f"run lock file {lock_path} is group- or world-accessible; refusing to use it "
            "instead of changing permissions of an existing file"
        )


def _acquire_exclusive(fd: int) -> bool:
    """Return True when the exclusive lock is held, False on contention."""
    if sys.platform == "win32":
        return _acquire_exclusive_windows(fd)
    return _acquire_exclusive_posix(fd)


def _acquire_exclusive_posix(fd: int) -> bool:
    try:
        _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise HistoryRunLockError(f"cannot acquire run lock: {exc}") from exc
    return True


def _acquire_exclusive_windows(fd: int) -> bool:
    # LK_NBLCK never blocks and the lock is released when the handle closes, even
    # if the owner dies. The current file position must cover the locked byte.
    os.lseek(fd, 0, os.SEEK_SET)
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EDEADLK):
            return False
        raise HistoryRunLockError(f"cannot acquire run lock: {exc}") from exc
    return True


def _release_lock(fd: int) -> None:
    if sys.platform == "win32":
        _release_lock_windows(fd)
    else:
        _release_lock_posix(fd)


def _release_lock_posix(fd: int) -> None:
    try:
        _fcntl.flock(fd, _fcntl.LOCK_UN)
    except OSError:
        pass


def _release_lock_windows(fd: int) -> None:
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    except OSError:
        pass


def _verify_lock_identity(fd: int, lock_path: Path) -> None:
    """Refuse to continue when the path no longer names the locked inode.

    If another process unlinked and recreated the lock file between open and
    lock, the held lock protects a now-orphaned inode while the live path could
    be locked again. Comparing the descriptor's device/inode with a fresh stat
    of the path detects that replacement and fails closed.
    """
    try:
        fd_status = os.fstat(fd)
        path_status = os.stat(lock_path)
    except OSError as exc:
        raise HistoryRunLockError(
            f"cannot verify run lock identity for {lock_path}: {exc}"
        ) from exc
    if (fd_status.st_dev, fd_status.st_ino) != (path_status.st_dev, path_status.st_ino):
        raise HistoryRunLockError(
            f"run lock file {lock_path} was replaced while held; refusing to continue"
        )
