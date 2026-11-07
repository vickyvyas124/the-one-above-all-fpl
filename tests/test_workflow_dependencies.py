"""Guards against dependency drift between the dashboard, the scheduled worker and CI.

Background: the scheduled "FPL Auto Learning Log" workflow installed only numpy,
requests and pandas, but tests/test_app_navigation.py imported Streamlit at module
level, so the validation step failed before the worker ever ran. These tests keep
the worker environment (requirements-worker.txt), the dashboard environment
(requirements.txt) and every workflow consistent. Standard library only: the worker
environment has no PyYAML or Streamlit.
"""
import ast
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
WORKER_WORKFLOWS = ("fpl-auto-learning.yml", "fpl-historical-backtest.yml")
# Modules executed by the scheduled workflows, directly or via imports.
WORKER_ENTRYPOINTS = ("fpl_background_worker", "historical_backtest", "fpl_snapshot_publisher", "fpl_learning")


def read_requirements(name):
    """Return {package: specifier} for a requirements file, ignoring comments."""
    packages = {}
    for raw in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"([A-Za-z0-9_.\-]+)\s*(.*)$", line)
        packages[match.group(1).lower().replace("_", "-")] = match.group(2).replace(" ", "")
    return packages


def local_modules():
    return {path.stem for path in ROOT.glob("*.py")}


def imported_top_level_names(path):
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def worker_import_closure():
    """Local modules reachable from the worker entrypoints, plus third-party imports."""
    local = local_modules()
    seen, queue, third_party = set(), list(WORKER_ENTRYPOINTS), set()
    while queue:
        module = queue.pop()
        if module in seen:
            continue
        seen.add(module)
        for name in imported_top_level_names(ROOT / f"{module}.py"):
            if name in local:
                queue.append(name)
            elif name not in sys.stdlib_module_names and name != "__future__":
                third_party.add(name)
    return seen, third_party


def pip_install_lines(workflow_text):
    return [line.strip() for line in workflow_text.splitlines() if "pip install" in line and not line.strip().startswith("#")]


class StreamlitIsolationTests(unittest.TestCase):
    def test_test_modules_only_import_streamlit_inside_an_import_guard(self):
        offenders = []
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            for node in ast.parse(path.read_text(encoding="utf-8")).body:
                bare = isinstance(node, ast.Import) and any(a.name.split(".")[0] == "streamlit" for a in node.names)
                bare = bare or (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "streamlit")
                if bare:
                    offenders.append(f"{path.name}:{node.lineno}")
        self.assertEqual(offenders, [], "Unguarded module-level Streamlit import in tests; wrap it in "
                                        "try/except ModuleNotFoundError so the worker environment can skip it.")

    def test_worker_code_never_imports_streamlit(self):
        modules, third_party = worker_import_closure()
        self.assertNotIn("app", modules)
        self.assertNotIn("streamlit", third_party)

    def test_dashboard_tests_skip_cleanly_when_streamlit_is_absent(self):
        """Behavioural check of the exact failure: run the dashboard tests with Streamlit unavailable."""
        script = (
            "import sys, unittest\n"
            "for name in ('streamlit', 'streamlit.testing', 'streamlit.testing.v1'):\n"
            "    sys.modules[name] = None\n"
            "sys.path[:0] = [sys.argv[1], sys.argv[1] + '/tests']\n"
            "suite = unittest.defaultTestLoader.loadTestsFromName('test_app_navigation')\n"
            "result = unittest.TextTestRunner(stream=open('/dev/null', 'w')).run(suite)\n"
            "print(result.testsRun, len(result.skipped), len(result.errors), len(result.failures))\n"
        )
        done = subprocess.run([sys.executable, "-c", script, str(ROOT)], capture_output=True, text=True, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr)
        run, skipped, errors, failures = (int(x) for x in done.stdout.split())
        self.assertGreater(run, 0)
        self.assertEqual((errors, failures), (0, 0))
        self.assertEqual(skipped, run, "Dashboard tests must be skipped, not silently passed, without Streamlit.")


class DependencyDriftTests(unittest.TestCase):
    def test_worker_requirements_cover_every_third_party_import_of_worker_code(self):
        _, third_party = worker_import_closure()
        declared = {name.replace("-", "_") for name in read_requirements("requirements-worker.txt")}
        self.assertEqual(sorted(third_party - declared), [],
                         "Worker code imports packages missing from requirements-worker.txt.")

    def test_shared_packages_use_identical_specifiers_in_both_requirement_files(self):
        worker, dashboard = read_requirements("requirements-worker.txt"), read_requirements("requirements.txt")
        for package in sorted(set(worker) & set(dashboard)):
            self.assertEqual(worker[package], dashboard[package], f"{package} specifier differs between files")
        self.assertNotIn("streamlit", worker, "The worker environment must stay free of the dashboard framework.")
        self.assertIn("streamlit", dashboard)

    def test_scheduled_workflows_install_from_requirements_worker_only(self):
        for name in WORKER_WORKFLOWS:
            with self.subTest(workflow=name):
                text = (WORKFLOWS / name).read_text(encoding="utf-8")
                lines = pip_install_lines(text)
                self.assertEqual(len(lines), 1, lines)
                self.assertIn("-r requirements-worker.txt", lines[0])
                tokens = lines[0].split("pip install", 1)[1].split()
                ad_hoc = [t for t in tokens if not t.startswith("-") and not t.endswith(".txt")]
                self.assertEqual(ad_hoc, [], "Install packages through requirements-worker.txt, not ad hoc lists.")
                self.assertIn("cache-dependency-path: requirements-worker.txt", text)

    def test_scheduled_workflows_run_the_unit_tests_before_touching_data(self):
        text = (WORKFLOWS / "fpl-auto-learning.yml").read_text(encoding="utf-8")
        tests_at = text.index("python -m unittest discover -s tests")
        self.assertLess(tests_at, text.index("Prepare persistent forecast branch"))
        self.assertLess(tests_at, text.index("python fpl_background_worker.py"))

    def test_pr_workflow_tests_both_the_dashboard_and_the_worker_environment(self):
        text = (WORKFLOWS / "test-pr.yml").read_text(encoding="utf-8")
        installs = pip_install_lines(text)
        self.assertEqual(len(installs), 2, installs)
        full = [line for line in installs if "-r requirements.txt" in line]
        worker_only = [line for line in installs if "-r requirements.txt" not in line]
        self.assertEqual(len(full), 1)
        self.assertIn("-r requirements-worker.txt", full[0])  # superset: dashboard + worker packages
        self.assertEqual(len(worker_only), 1)
        self.assertIn("-r requirements-worker.txt", worker_only[0])
        self.assertIn("python -m unittest discover -s tests", text.split("test-worker-env:", 1)[1])

    def test_backtest_workflow_refuses_to_publish_an_unusable_model(self):
        text = (WORKFLOWS / "fpl-historical-backtest.yml").read_text(encoding="utf-8")
        check = text.index("python fpl_learning.py check")
        self.assertIn("--strict", text[check:check + 200])
        self.assertLess(check, text.index("cp \"$RUNNER_TEMP/historical-output/trained_fpl_model.json\""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
