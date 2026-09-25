"""Portable, read-only access to explicitly declared local image paths."""

from pathlib import Path, PurePosixPath, PureWindowsPath


def relative_posix_path(value: str) -> str:
    """Reject normalization ambiguities, traversal and Windows drive/ADS paths."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or ":" in value
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or any(part.endswith((" ", ".")) for part in value.split("/"))
        or PureWindowsPath(value).is_reserved()
    ):
        raise ValueError(f"Unsafe relative POSIX path: {value!r}")
    return value


def declared_file(root: Path, relative: str) -> Path:
    """Resolve only a declared file; symlinks/junctions may never escape root.

    Call again at read time. Sources must stay immutable during the operation;
    this is not a lock against concurrent hostile filesystem changes.
    """
    relative_posix_path(relative)
    root = root.resolve()
    path = root.joinpath(*relative.split("/"))
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Declared path escapes root: {relative}")
    if not path.is_file():
        raise ValueError(f"Declared file is missing or not a file: {relative}")
    return path
