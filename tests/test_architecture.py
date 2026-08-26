"""The seams, asserted rather than described.

Every rule here is one that a reader of CLAUDE.md is told holds. A documented invariant with
no test is a suggestion, and the way it fails is always the same: someone adds a reasonable
line, nothing complains, and the property is gone months before anyone notices. Two of these
were claimed in CLAUDE.md while no test existed — an adversarial review found that out, which
is exactly the wrong way to discover it.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "shipvision"
CSRC = REPO / "csrc"

#: The native source tree. It mirrors ``shipvision/`` directory for directory, and ``csrc/``
#: itself is the include root, so ``#include "shipvision/imgproc/image_ops.h"`` resolves.
NATIVE = CSRC / "shipvision"

#: The single file allowed to name a vendor API directly. Everything else goes through its
#: ``gpu*`` aliases.
PLATFORM_HEADER = NATIVE / "core" / "platform.h"

#: Paths under ``csrc/shipvision/``, relative to it, with no Python counterpart. ``core`` is the
#: C++-only foundation — the vendor aliases and the scratch allocators — and has nothing to
#: mirror.
NATIVE_ONLY = {Path("core")}

#: Every C/C++/CUDA suffix that carries code. ``.hpp`` is still scanned even though no file
#: may use it, so a re-introduced one is caught by the vendor guard as well as by the layout
#: test that forbids it.
SOURCE_SUFFIXES = {".cpp", ".cu", ".cuh", ".h", ".hpp"}

#: ``#include "shipvision/..."`` — the project's own includes, as opposed to ``<system>`` ones.
PROJECT_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)

#: Anything that would make the library unusable where it must remain usable: on a laptop, in
#: CI, and in the offline test tier.
HEAVY_MODULES = ("torch", "scipy", "cv2", "tensorrt", "optuna")

VENDOR_CALL = re.compile(r"\b(?:cuda|hip)[A-Z]\w*")


def strip_c_comments(text: str) -> str:
    """Remove // and /* */ comments, so prose about `cudaErrorMisalignedAddress` is allowed.

    Naming an error in a comment that explains why an alias exists is the opposite of a
    violation, and a grep that flagged it would be turned off within a week.
    """
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


class TestVendorApiBoundary:
    """One C++ source tree compiles for CUDA and for HIP. That only holds if nothing outside
    ``core/platform.h`` names a vendor function, because the alias layer is what makes the
    two builds the same code. A raw ``cudaMalloc`` elsewhere breaks the ROCm build silently —
    it compiles on the machine that added it and fails on a machine nobody tests on."""

    def test_the_platform_header_exists_where_the_rule_says_it_does(self) -> None:
        assert PLATFORM_HEADER.is_file(), (
            f"{PLATFORM_HEADER} is the only file allowed to name a vendor API; if it moved, "
            f"this test and CLAUDE.md both need updating"
        )

    def test_no_source_outside_the_platform_header_names_a_vendor_api(self) -> None:
        offenders: list[str] = []
        for path in sorted(CSRC.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            if path.resolve() == PLATFORM_HEADER.resolve():
                continue
            code = strip_c_comments(path.read_text())
            for line_number, line in enumerate(code.splitlines(), start=1):
                for match in VENDOR_CALL.finditer(line):
                    offenders.append(
                        f"{path.relative_to(REPO)}:{line_number}: {match.group(0)}"
                    )

        assert not offenders, (
            "raw vendor calls outside core/platform.h — these compile for one vendor and "
            "break the other silently:\n  " + "\n  ".join(offenders)
        )

    def test_the_guard_would_actually_catch_a_violation(self) -> None:
        """A guard nobody has seen fail is a guard nobody should trust."""
        assert VENDOR_CALL.search("auto err = cudaMalloc(&p, n);")
        assert VENDOR_CALL.search("hipStreamSynchronize(stream);")
        assert not VENDOR_CALL.search("gpuMalloc(&p, n);")
        assert not VENDOR_CALL.search("cuda_available()"), "snake_case is ours, not theirs"
        assert not strip_c_comments("// cudaErrorMisalignedAddress is sticky").strip()


class TestLibraryIndependence:
    """This library must not import the server that consumes it. The dependency is one-way —
    ShipInfer calls in here — and reversing it would mean an algorithm could no longer be
    evaluated without a serving stack, which is the entire reason the split exists."""

    def test_nothing_imports_shipinfer(self) -> None:
        offenders: list[str] = []
        for path in sorted(PACKAGE.rglob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name == "shipinfer" or name.startswith("shipinfer."):
                        offenders.append(f"{path.relative_to(REPO)}:{node.lineno}: {name}")

        assert not offenders, "shipvision must not import shipinfer:\n  " + "\n  ".join(
            offenders
        )


class TestImportWeight:
    """`import shipvision` has to work on a laptop with no GPU and no build. Each family
    defers its own heavy backends, but importing a family at all runs the module body that
    registers them — so the top level must not import the families."""

    @pytest.mark.parametrize("module", HEAVY_MODULES)
    def test_importing_the_package_does_not_load(self, module: str) -> None:
        """A fresh interpreter, because this test process has imported plenty already."""
        code = f"import sys, shipvision; print({module!r} in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )

        assert result.stdout.strip() == "False", (
            f"import shipvision pulled in {module} — something in a __init__.py imports a "
            f"family or a backend eagerly"
        )

    def test_the_top_level_does_not_import_a_family(self) -> None:
        """The structural version of the test above: it fails on the line that would break it
        rather than on the consequence, and it keeps holding if a family's dependencies
        happen to be absent from the machine running the suite."""
        tree = ast.parse((PACKAGE / "__init__.py").read_text())
        families = {"imgproc", "detection", "reid", "mot", "tracking", "mtmc", "eval", "tune"}
        offenders: list[str] = []
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
            elif isinstance(node, ast.Import):
                module = node.names[0].name
            if module and module.startswith("shipvision."):
                head = module.split(".")[1]
                if head in families:
                    offenders.append(f"line {node.lineno}: {module}")

        assert not offenders, (
            "shipvision/__init__.py imports a family eagerly; use the lazy __getattr__ "
            "instead:\n  " + "\n  ".join(offenders)
        )


class TestMarkers:
    """The two opt-in tiers must stay opt-in, or the offline suite quietly starts needing
    hardware and CI goes red for reasons nobody can reproduce locally."""

    def test_the_markers_are_declared(self) -> None:
        config = (REPO / "pyproject.toml").read_text()

        for marker in ("gpu:", "native:", "slow:"):
            assert marker in config, f"marker {marker!r} is used but not declared"

    def test_the_runner_deselects_the_opt_in_tiers_and_hides_the_devices(self) -> None:
        """CLAUDE.md tells a developer to use this script rather than bare pytest, and the
        reason is that it reproduces CI exactly. That claim was false once."""
        script = (REPO / "scripts" / "run_tests.sh").read_text()

        assert 'CUDA_VISIBLE_DEVICES=""' in script
        assert 'HIP_VISIBLE_DEVICES=""' in script
        assert "not gpu" in script and "not native" in script


class TestNativeLayout:
    """A component's header and its translation unit live next to each other, in a tree that
    mirrors the Python package. The split this replaced — ``csrc/include/shipvision/imgproc/``
    against ``csrc/src/imgproc_image_ops.cu`` — put the declaration and the definition four
    directories apart under two different names, so editing one and forgetting the other was
    the normal outcome rather than an unusual one. These tests are what stop it drifting back:
    the layout is load-bearing for the include root, and nothing else notices until a build
    that only runs on a GPU machine fails."""

    def test_the_old_include_src_split_is_gone(self) -> None:
        for stale in (CSRC / "include", CSRC / "src"):
            assert not stale.exists(), (
                f"{stale.relative_to(REPO)} is back — a header belongs beside its translation "
                f"unit, not in a parallel tree"
            )

    def test_no_header_uses_the_hpp_suffix(self) -> None:
        offenders = sorted(str(p.relative_to(REPO)) for p in CSRC.rglob("*.hpp"))

        assert not offenders, "headers are .h in this repository, not .hpp:\n  " + "\n  ".join(
            offenders
        )

    def test_every_translation_unit_has_its_header_beside_it(self) -> None:
        """The bindings are the deliberate exception: ``module.cpp`` declares nothing for
        anyone else to include, so a ``module.h`` would be an empty file kept in sync."""
        units = [p for p in sorted(NATIVE.rglob("*")) if p.suffix in {".cu", ".cpp"}]
        offenders = [
            str(p.relative_to(REPO)) for p in units if not p.with_suffix(".h").is_file()
        ]

        assert units, (
            f"found no .cu/.cpp under {NATIVE.relative_to(REPO)}; without this the test below "
            f"passes on an empty tree and stops meaning anything"
        )
        assert not offenders, (
            "translation units with no sibling header — either the header is in another "
            "directory or the two are named differently:\n  " + "\n  ".join(offenders)
        )

    def test_the_native_tree_mirrors_the_python_package(self) -> None:
        """At every depth, not only at the top. ``csrc/shipvision/mot/trackers/sort/`` earns its
        place by ``shipvision/mot/trackers/sort/`` existing, and checking only the first level
        would have accepted the source branch's ``mot/core/sort/`` — a second name for the same
        five algorithms, which is how a reader ends up unsure which pair is the pair."""
        offenders: list[str] = []
        for path in sorted(NATIVE.rglob("*")):
            relative = path.relative_to(NATIVE)
            if not path.is_dir() or relative in NATIVE_ONLY:
                continue
            if not (PACKAGE / relative / "__init__.py").is_file():
                offenders.append(str(path.relative_to(REPO)))

        assert not offenders, (
            "native directories with no Python package of the same path. A fused kernel is only "
            "trustworthy if a readable implementation agrees with it, and the two are found by "
            "having the same name:\n  " + "\n  ".join(offenders)
        )

    def test_every_project_include_resolves_from_the_include_root(self) -> None:
        """The property that makes ``csrc/`` the include root: every quoted include is a path
        relative to it. Nothing else checks this offline — a stale ``#include`` fails at
        compile time, on a machine with a CUDA toolchain, which is not where it gets noticed."""
        offenders: list[str] = []
        for path in sorted(CSRC.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            for included in PROJECT_INCLUDE.findall(path.read_text()):
                if not (CSRC / included).is_file():
                    offenders.append(f'{path.relative_to(REPO)}: #include "{included}"')

        assert not offenders, "includes that do not resolve against csrc/:\n  " + "\n  ".join(
            offenders
        )

    def test_every_csrc_path_cmake_names_still_exists(self) -> None:
        """CMake globs the tree, so a move does not break the build loudly — it produces an
        empty source list and a link error about a missing module, which reads like anything
        but a path typo."""
        cmake = (REPO / "CMakeLists.txt").read_text()
        offenders: list[str] = []
        for reference in re.findall(r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/(csrc[\w/]*)", cmake):
            if not (REPO / reference).is_dir():
                offenders.append(reference)

        missing = "\n  ".join(offenders)

        assert (
            not offenders
        ), f"CMakeLists.txt names csrc directories that do not exist:\n  {missing}"
        assert "csrc/shipvision/*.cu" in cmake, (
            "the CUDA glob must recurse into csrc/shipvision/; a flat csrc/src/*.cu glob would "
            "now match nothing"
        )


class TestTheBindingsTakeNoGilPolicy:
    """Whether a call may run concurrently with other Python is the *embedding server's*
    decision. ShipInfer runs one worker thread per model instance and releases the lock at its
    own boundary; a ``py::gil_scoped_release`` in here would be a second, invisible policy
    underneath that one, and the library cannot see what else is on the thread to know whether
    it is right. What this library owes its embedder instead is that a concurrent call is safe,
    which is what the per-session mutex in ``bindings/mot.cpp`` is for.

    Written as a test because the alternative is a sentence in CLAUDE.md, and a scoped release
    is exactly the kind of line that gets added back by someone optimising one entry point."""

    def test_no_binding_releases_or_acquires_the_interpreter_lock(self) -> None:
        offenders: list[str] = []
        for path in sorted(CSRC.rglob("*")):
            if not path.is_file() or path.suffix not in SOURCE_SUFFIXES:
                continue
            code = strip_c_comments(path.read_text())
            for line_number, line in enumerate(code.splitlines(), start=1):
                if "gil_scoped" in line:
                    offenders.append(f"{path.relative_to(REPO)}:{line_number}: {line.strip()}")

        assert not offenders, (
            "GIL policy belongs to the server that embeds this library, not to the library:\n  "
            + "\n  ".join(offenders)
        )

    def test_only_the_bindings_directory_names_pybind_at_all(self) -> None:
        """The other half of the same boundary. ``csrc/shipvision/`` is a plain C++/CUDA library
        that has never heard of an interpreter — a ``py::`` type reaching a kernel's header is
        what makes the algorithms unusable from anything but Python, and untestable without
        one."""
        offenders = [
            str(path.relative_to(REPO))
            for path in sorted(NATIVE.rglob("*"))
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and "pybind11" in strip_c_comments(path.read_text())
        ]

        assert not offenders, (
            "pybind11 outside csrc/bindings/ — the library half must build and be usable with "
            "no interpreter present:\n  " + "\n  ".join(offenders)
        )


class TestNativeTreeIsNotShipped:
    """``csrc/shipvision/`` has the same name as the Python package on purpose — it is the
    include root's view of the same components. That makes one packaging mistake possible and
    catastrophic: shipping it would put a directory of .cu files on ``sys.path`` under the name
    ``shipvision``, shadowing the real package with something that has no ``__init__.py``."""

    def test_package_discovery_does_not_find_the_native_tree(self) -> None:
        setuptools = pytest.importorskip("setuptools", reason="build backend not installed")

        found = setuptools.find_packages(where=str(REPO), include=["shipvision*"])

        assert "shipvision" in found, "the real package must still be discovered"
        assert not [name for name in found if "csrc" in name]

    def test_the_native_tree_has_no_init_file(self) -> None:
        """The first of the two independent reasons discovery skips it."""
        assert not (NATIVE / "__init__.py").exists()
