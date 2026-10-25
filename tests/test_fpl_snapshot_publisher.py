import gzip
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fpl_snapshot_publisher as publisher


class SnapshotPublisherIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bare = self.root / "remote.git"
        self.seed = self.root / "seed"
        self.seed.mkdir()
        self._git(self.seed, "init", "-b", "fpl-snapshots")
        self._git(self.seed, "config", "user.name", "Snapshot Test")
        self._git(self.seed, "config", "user.email", "snapshot@example.test")
        self._git(self, "init", "--bare", "--initial-branch=fpl-snapshots", str(self.bare))

        self.old_manifest = "2026-27/gw05/24h/manifest.json"
        self.old_payload = "2026-27/gw05/24h/payloads/bootstrap-static.json.gz"
        self.old_diag = "diagnostics.jsonl"
        self.old_actual = "actuals/2026-27/gw04.jsonl"
        self.old_index = "reconciliation-index.json"
        initial = {
            self.old_manifest: b'{"identity":{"gameweek":5,"window_label":"24h"}}\n',
            self.old_payload: gzip.compress(b'{"elements":[]}', mtime=0),
            self.old_diag: b'{"kind":"previous-diagnostic"}\n',
            self.old_actual: b'{"record_type":"actual_score","player_id":1,"revision":1}\n',
            self.old_index: b'{"2026-27:GW04":"2026-10-01T12:00:00+00:00"}\n',
        }
        self.initial_bytes = initial
        for relative, body in initial.items():
            target = self.seed / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
        self._git(self.seed, "add", ".")
        self._git(self.seed, "commit", "-m", "Seed existing 24h snapshot and mutable history")
        self._git(self.seed, "remote", "add", "origin", str(self.bare))
        self._git(self.seed, "push", "-u", "origin", "fpl-snapshots")
        self.initial_head = self._git(self, "--git-dir", str(self.bare), "rev-parse", "refs/heads/fpl-snapshots").stdout.strip()
        self.repo = self._clone_partial("publisher")
        self.source = self.root / "partial-source"
        self.source.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def _git(self, cwd, *args, check=True):
        directory = cwd if isinstance(cwd, Path) else self.root
        command = (
            ["git", *args]
            if len(args) >= 1 and args[0] == "--git-dir"
            else ["git", "-C", str(directory), *args]
        )
        result = subprocess.run(
            command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode and check:
            self.fail(f"git {args!r} failed: {result.stderr}")
        return result

    def _clone_partial(self, name):
        destination = self.root / name
        result = subprocess.run(
            ["git", "clone", "--no-checkout", "--branch", "fpl-snapshots", str(self.bare), str(destination)],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode:
            self.fail(f"git clone failed: {result.stderr}")
        self._git(destination, "read-tree", "HEAD")
        return destination

    def _write_source_file(self, relative, content):
        target = self.source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def _source_with_new_6h_snapshot(self):
        manifest = "2026-27/gw05/6h/manifest.json"
        payload = "2026-27/gw05/6h/payloads/fixtures.json.gz"
        self._write_source_file(manifest, b'{"identity":{"gameweek":5,"window_label":"6h"}}\n')
        self._write_source_file(payload, gzip.compress(b"[]", mtime=0))
        return manifest, payload

    def _remote_blob(self, relative):
        result = subprocess.run(
            ["git", "--git-dir", str(self.bare), "show", f"refs/heads/fpl-snapshots:{relative}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout

    def test_partial_6h_publish_preserves_existing_24h_and_appends_mutable_records(self):
        manifest, payload = self._source_with_new_6h_snapshot()
        self._write_source_file(self.old_diag, self.initial_bytes[self.old_diag] + b'{"kind":"new-diagnostic"}\n')
        self._write_source_file(self.old_actual, self.initial_bytes[self.old_actual] + b'{"record_type":"actual_score","player_id":2,"revision":1}\n')
        self._write_source_file(self.old_index, b'{"2026-27:GW04":"updated","2026-27:GW05":"now"}\n')

        commit = publisher.publish(self.source, self.repo, push=True)
        self.assertIsNotNone(commit)
        self.assertEqual(self._remote_blob(manifest), (self.source / manifest).read_bytes())
        self.assertEqual(self._remote_blob(payload), (self.source / payload).read_bytes())
        self.assertEqual(self._remote_blob(self.old_manifest), self.initial_bytes[self.old_manifest])
        self.assertEqual(self._remote_blob(self.old_payload), self.initial_bytes[self.old_payload])
        self.assertEqual(
            self._remote_blob(self.old_diag),
            self.initial_bytes[self.old_diag] + b'{"kind":"new-diagnostic"}\n',
        )
        self.assertEqual(
            self._remote_blob(self.old_actual),
            self.initial_bytes[self.old_actual]
            + b'{"record_type":"actual_score","player_id":2,"revision":1}\n',
        )
        self.assertEqual(self._remote_blob(self.old_index), b'{"2026-27:GW04":"updated","2026-27:GW05":"now"}\n')
        changed = self._git(
            self.repo, "diff-tree", "--no-commit-id", "--name-status", "-r",
            "--no-renames", "HEAD^", "HEAD",
        ).stdout.splitlines()
        self.assertTrue(changed)
        self.assertFalse(any(row.startswith("D\t") for row in changed))
        self.assertEqual(self._git(self.repo, "diff", "--cached", "--quiet", "HEAD", check=False).returncode, 0)
        self.assertEqual(
            self._git(self, "--git-dir", str(self.bare), "rev-parse", "refs/heads/fpl-snapshots").stdout.strip(),
            commit,
        )

    def test_first_publish_to_empty_local_bare_remote_succeeds(self):
        empty_bare = self.root / "empty.git"
        empty_repo = self.root / "empty-publisher"
        empty_source = self.root / "first-capture"
        empty_source.mkdir()
        self._git(self, "init", "--bare", "--initial-branch=fpl-snapshots", str(empty_bare))
        self._git(self, "init", "-b", "fpl-snapshots", str(empty_repo))
        self._git(empty_repo, "remote", "add", "origin", str(empty_bare))
        relative = "2026-27/gw01/24h/manifest.json"
        target = empty_source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'{"identity":{"gameweek":1,"window_label":"24h"}}\n')

        commit = publisher.publish(empty_source, empty_repo, push=True)

        self.assertEqual(self._remote_blob_from(empty_bare, relative), target.read_bytes())
        self.assertEqual(
            self._git(self, "--git-dir", str(empty_bare), "rev-parse", "refs/heads/fpl-snapshots").stdout.strip(),
            commit,
        )

    def _remote_blob_from(self, remote, relative):
        result = subprocess.run(
            ["git", "--git-dir", str(remote), "show", f"refs/heads/fpl-snapshots:{relative}"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return result.stdout

    def test_modifying_existing_immutable_snapshot_is_rejected_without_push(self):
        self._write_source_file(self.old_manifest, b'{"rewritten":true}\n')
        before = self._git(self, "--git-dir", str(self.bare), "rev-parse", "refs/heads/fpl-snapshots").stdout.strip()
        with self.assertRaisesRegex(publisher.PublishSafetyError, "immutable"):
            publisher.publish(self.source, self.repo, push=True)
        after = self._git(self, "--git-dir", str(self.bare), "rev-parse", "refs/heads/fpl-snapshots").stdout.strip()
        self.assertEqual(after, before)
        self.assertEqual(self._git(self.repo, "rev-parse", "HEAD").stdout.strip(), self.initial_head)

    def test_modifying_existing_source_payload_is_rejected_without_push(self):
        self._write_source_file(self.old_payload, b"rewritten-source-payload")
        with self.assertRaisesRegex(publisher.PublishSafetyError, "immutable"):
            publisher.publish(self.source, self.repo, push=True)
        after = self._git(self, "--git-dir", str(self.bare), "rev-parse", "refs/heads/fpl-snapshots").stdout.strip()
        self.assertEqual(after, self.initial_head)

    def test_staged_immutable_deletion_is_rejected_without_push(self):
        self._git(self.repo, "rm", "--cached", "--", self.old_manifest)
        self._source_with_new_6h_snapshot()
        with self.assertRaisesRegex(publisher.PublishSafetyError, "deletion"):
            publisher.publish(self.source, self.repo, push=True)
        after = self._git(self, "--git-dir", str(self.bare), "rev-parse", "refs/heads/fpl-snapshots").stdout.strip()
        self.assertEqual(after, self.initial_head)

    def test_mutable_jsonl_rewrite_or_truncation_is_rejected(self):
        self._write_source_file(self.old_diag, b'{"kind":"replacement"}\n')
        with self.assertRaisesRegex(publisher.PublishSafetyError, "rewritten or truncated"):
            publisher.publish(self.source, self.repo, push=True)
        after = self._git(self, "--git-dir", str(self.bare), "rev-parse", "refs/heads/fpl-snapshots").stdout.strip()
        self.assertEqual(after, self.initial_head)

    def test_invalid_new_actual_ledger_is_rejected(self):
        relative = "actuals/2026-27/gw06.jsonl"
        self._write_source_file(relative, b"not-json\n")
        with self.assertRaisesRegex(publisher.PublishSafetyError, "Invalid JSONL record"):
            publisher.publish(self.source, self.repo, push=True)
        after = self._git(self, "--git-dir", str(self.bare), "rev-parse", "refs/heads/fpl-snapshots").stdout.strip()
        self.assertEqual(after, self.initial_head)

    def test_workflow_publisher_fails_closed_off_main(self):
        self._source_with_new_6h_snapshot()
        with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_REF": "refs/heads/feature/test"}):
            with self.assertRaisesRegex(publisher.PublishSafetyError, "only on main"):
                publisher.publish(self.source, self.repo, push=True)
        self.assertEqual(self._git(self.repo, "rev-parse", "HEAD").stdout.strip(), self.initial_head)

    def test_workflow_calls_shared_main_only_publisher_with_push_branch(self):
        root = Path(__file__).parents[1]
        workflow = (root / ".github/workflows/fpl-auto-learning.yml").read_text(encoding="utf-8-sig")
        self.assertIn("if: github.ref == 'refs/heads/main'", workflow)
        self.assertIn("python fpl_snapshot_publisher.py", workflow)
        self.assertIn("--branch fpl-snapshots", workflow)
        self.assertIn("--push", workflow)
        self.assertNotIn('git add -- "', workflow)


if __name__ == "__main__":
    unittest.main(verbosity=2)
