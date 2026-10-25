from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


class PublishSafetyError(RuntimeError):
    """Raised when a snapshot publish would violate the immutable-data policy."""


MUTABLE_ROOT_FILES = {"diagnostics.jsonl", "reconciliation-index.json"}
ACTUALS_PATH = re.compile(r"actuals/[^/]+/gw\d{2}\.jsonl\Z")
SNAPSHOT_PATH = re.compile(r"\d{4}-\d{2}/gw\d{2}/(?:24h|6h)/.+\Z")


def _git(repository: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise PublishSafetyError(
            f"git {' '.join(args)} failed ({result.returncode}): {result.stderr.strip()}"
        )
    return result


def _mutable(path: str) -> bool:
    return path in MUTABLE_ROOT_FILES or ACTUALS_PATH.fullmatch(path) is not None


def _source_files(source: Path) -> list[tuple[Path, str]]:
    if not source.is_dir():
        raise PublishSafetyError(f"Snapshot source directory does not exist: {source}")
    files: list[tuple[Path, str]] = []
    for item in sorted(source.rglob("*")):
        if item.is_symlink():
            raise PublishSafetyError(f"Symbolic links are not accepted in snapshot source: {item}")
        if not item.is_file():
            continue
        relative = item.relative_to(source).as_posix()
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or not relative:
            raise PublishSafetyError(f"Invalid snapshot path: {relative}")
        files.append((item, relative))
    if not files:
        raise PublishSafetyError("Snapshot source contains no files")
    return files


def _verified_base_commit(repository: Path, branch: str) -> str | None:
    """Return the verified current commit, or None only for a proven unborn branch."""
    head = _git(repository, "rev-parse", "--verify", "HEAD", check=False)
    if head.returncode == 0:
        base_commit = head.stdout.strip()
        if not base_commit:
            raise PublishSafetyError("HEAD resolved to an empty commit identifier")
        _git_bytes(repository, "cat-file", "-e", f"{base_commit}^{{commit}}")
        _git_bytes(repository, "cat-file", "-e", f"{base_commit}^{{tree}}")
        return base_commit

    symbolic_head = _git(repository, "symbolic-ref", "-q", "HEAD", check=False)
    expected_ref = f"refs/heads/{branch}"
    if symbolic_head.returncode != 0 or symbolic_head.stdout.strip() != expected_ref:
        raise PublishSafetyError(
            f"Could not verify repository HEAD (expected unborn {expected_ref})"
        )
    refs = _git(
        repository, "for-each-ref", "--format=%(refname)", expected_ref, check=False
    )
    if refs.returncode != 0:
        raise PublishSafetyError(
            f"Could not verify whether {expected_ref} exists: {refs.stderr.strip()}"
        )
    if expected_ref in refs.stdout.splitlines():
        raise PublishSafetyError(
            f"{expected_ref} exists but its commit object could not be verified"
        )
    return None


def _staged_changes(
    repository: Path, base_commit: str | None
) -> list[tuple[str, str]]:
    args = ["diff", "--cached", "--name-status", "-z", "--no-renames"]
    if base_commit is None:
        args.append("--root")
    else:
        args.append(base_commit)
    result = _git(repository, *args)
    fields = result.stdout.split("\0")
    if fields and fields[-1] == "":
        fields.pop()
    if len(fields) % 2:
        raise PublishSafetyError("Could not parse the staged Git change list")
    return [(fields[index], fields[index + 1]) for index in range(0, len(fields), 2)]


def _git_bytes(repository: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode:
        raise PublishSafetyError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}"
        )
    return result.stdout


def _head_bytes(
    repository: Path, relative: str, base_commit: str | None
) -> bytes | None:
    """Return a prior blob, None only when the verified base tree lacks the path."""
    if base_commit is None:
        return None
    entries = _git_bytes(
        repository, "ls-tree", "-r", "-z", "--full-tree", base_commit,
        "--", f":(literal){relative}",
    )
    for entry in entries.split(b"\0"):
        if not entry:
            continue
        try:
            metadata, entry_path = entry.split(b"\t", 1)
            _mode, object_type, object_id = metadata.split(b" ", 2)
            decoded_path = entry_path.decode("utf-8")
        except (ValueError, UnicodeDecodeError) as exc:
            raise PublishSafetyError(
                f"Could not parse prior tree entry for {relative}: {exc}"
            ) from exc
        if decoded_path != relative:
            continue
        if object_type != b"blob":
            raise PublishSafetyError(f"Prior path is not a blob: {relative}")
        # Reading by object ID triggers promisor lazy fetch when available. Any
        # unavailable object is an error, never evidence that the path is new.
        return _git_bytes(repository, "cat-file", "blob", object_id.decode("ascii"))
    return None


def _staged_bytes(repository: Path, relative: str) -> bytes:
    exists = _git(repository, "cat-file", "-e", f":{relative}", check=False)
    if exists.returncode:
        raise PublishSafetyError(f"Staged path is missing from index: {relative}")
    return _git_bytes(repository, "show", f":{relative}")

def _validate_append_only_jsonl(relative: str, before: bytes, after: bytes) -> None:
    if before and not before.endswith(b"\n"):
        raise PublishSafetyError(f"Existing JSONL history has no final newline: {relative}")
    if not after.startswith(before):
        raise PublishSafetyError(f"Mutable JSONL history was rewritten or truncated: {relative}")
    try:
        for line_number, line in enumerate(after.splitlines(), 1):
            if not line.strip():
                raise ValueError("blank line")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("record is not a JSON object")
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PublishSafetyError(
            f"Invalid JSONL record in {relative} at line {line_number}: {exc}"
        ) from exc


def _validate_mutable_bytes(relative: str, before: bytes, after: bytes) -> None:
    if relative.endswith(".jsonl"):
        _validate_append_only_jsonl(relative, before, after)
    elif relative == "reconciliation-index.json":
        try:
            value = json.loads(after)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PublishSafetyError(f"Invalid reconciliation index JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise PublishSafetyError("Reconciliation index must be a JSON object")


_BASE_UNSET = object()


def validate_staged_changes(
    repository: str | Path, *, base_commit: str | None | object = _BASE_UNSET
) -> list[tuple[str, str]]:
    """Print and validate staged changes against a verified base tree."""
    repository = Path(repository).resolve()
    if base_commit is _BASE_UNSET:
        branch = _git(repository, "branch", "--show-current").stdout.strip()
        base_commit = _verified_base_commit(repository, branch)
    if base_commit is not None and not isinstance(base_commit, str):
        raise PublishSafetyError("Invalid verified base commit")

    changes = _staged_changes(repository, base_commit)
    if changes:
        print("Staged changes:")
        for status, relative in changes:
            print(f"  {status} {relative}")
    else:
        print("Staged changes: none")

    for status, relative in changes:
        if status == "D":
            raise PublishSafetyError(f"Staged deletion is forbidden: {relative}")
        before = _head_bytes(repository, relative, base_commit)
        if before is None:
            if status != "A":
                raise PublishSafetyError(
                    f"Prior path state could not be verified for staged {status}: {relative}"
                )
            if not _mutable(relative) and SNAPSHOT_PATH.fullmatch(relative) is None:
                raise PublishSafetyError(f"New file is outside the snapshot namespace: {relative}")
            if _mutable(relative):
                _validate_mutable_bytes(relative, b"", _staged_bytes(repository, relative))
            continue
        if status == "A":
            raise PublishSafetyError(f"Staged addition already exists in the verified base: {relative}")
        if not _mutable(relative):
            raise PublishSafetyError(
                f"Existing immutable snapshot or non-mutable path changed: {relative}"
            )
        _validate_mutable_bytes(relative, before, _staged_bytes(repository, relative))
    return changes

def publish(source: str | Path, repository: str | Path, *,
            remote: str = "origin", branch: str = "fpl-snapshots",
            push: bool = True) -> str | None:
    """Copy, stage, guard, commit and optionally push one partial snapshot workspace."""
    source = Path(source).resolve()
    repository = Path(repository).resolve()
    files = _source_files(source)

    if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
        if os.environ.get("GITHUB_REF") != "refs/heads/main":
            raise PublishSafetyError("Persistent snapshot publishing is allowed only on main")

    current_branch = _git(repository, "branch", "--show-current").stdout.strip()
    if current_branch != branch:
        raise PublishSafetyError(
            f"Publisher repository is on {current_branch!r}, expected {branch!r}"
        )
    # Capture and verify the base before staging. None means a proven unborn
    # branch only; shallow/missing commits and tree errors are fatal.
    base_commit = _verified_base_commit(repository, branch)
    if _staged_changes(repository, base_commit):
        validate_staged_changes(repository, base_commit=base_commit)
        raise PublishSafetyError("Publisher repository already had staged changes before publication")

    # The source is intentionally partial. Copy only supplied files; never sync deletions.
    for source_file, relative in files:
        destination = repository.joinpath(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, destination)

    # --ignore-removal is a first defense against partial checkouts. The staged
    # diff and verified-base checks below independently reject deletions/edits.
    top_level = sorted({PurePosixPath(relative).parts[0] for _, relative in files})
    _git(repository, "add", "--ignore-removal", "--", *top_level)
    changes = validate_staged_changes(repository, base_commit=base_commit)
    if not changes:
        print("Snapshot branch already contains all captured data.")
        return None

    _git(repository, "-c", "user.name=github-actions[bot]",
          "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
          "commit", "-m", "Capture immutable FPL deadline snapshot")

    # Verify this commit has exactly the base captured above, then compare its
    # tree directly with that known commit. Do not infer root status from HEAD^:
    # the workflow uses depth-one clones whose earlier parent may be unavailable.
    commit = _git(repository, "rev-parse", "--verify", "HEAD").stdout.strip()
    parents = _git(repository, "rev-list", "--parents", "-n", "1", commit).stdout.split()
    expected_parents = [commit] if base_commit is None else [commit, base_commit]
    if parents != expected_parents:
        raise PublishSafetyError(
            f"Committed parent state could not be verified; expected {expected_parents}, got {parents}"
        )

    diff_args = ["diff-tree", "--no-commit-id", "--name-status", "-r", "--no-renames"]
    diff_args.extend(["--root", commit] if base_commit is None else [base_commit, commit])
    committed_status = _git(repository, *diff_args).stdout.splitlines()
    for line in committed_status:
        status, relative = line.split("\t", 1)
        if status == "D":
            raise PublishSafetyError("Committed snapshot update contains a deletion; refusing push")
        prior = _head_bytes(repository, relative, base_commit)
        if prior is None:
            if status != "A":
                raise PublishSafetyError(
                    f"Committed prior path state could not be verified: {relative}; refusing push"
                )
            if not _mutable(relative) and SNAPSHOT_PATH.fullmatch(relative) is None:
                raise PublishSafetyError(
                    f"Committed new path is outside the snapshot namespace: {relative}; refusing push"
                )
            if _mutable(relative):
                after = _git_bytes(repository, "show", f"{commit}:{relative}")
                _validate_mutable_bytes(relative, b"", after)
            continue
        if status == "A":
            raise PublishSafetyError(
                f"Committed addition already exists in verified base: {relative}; refusing push"
            )
        if not _mutable(relative):
            raise PublishSafetyError(
                f"Committed tree changed existing immutable path: {relative}; refusing push"
            )
        after = _git_bytes(repository, "show", f"{commit}:{relative}")
        _validate_mutable_bytes(relative, prior, after)

    if push:
        _git(repository, "push", remote, f"HEAD:{branch}")
    return commit

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="fpl-snapshots")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args(argv)
    try:
        commit = publish(
            args.source, args.repository, remote=args.remote, branch=args.branch, push=args.push
        )
    except (OSError, PublishSafetyError, subprocess.CalledProcessError) as exc:
        print(f"Snapshot publish rejected: {exc}", file=sys.stderr)
        return 1
    if commit:
        print(f"Published snapshot commit {commit}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
