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

#: The single file allowed to name a vendor API directly. Everything else goes through its
#: ``gpu*`` aliases.
PLATFORM_HEADER = CSRC / "include" / "shipvision" / "core" / "platform.hpp"

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
    ``core/platform.hpp`` names a vendor function, because the alias layer is what makes the
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
            if not path.is_file() or path.suffix not in {".cpp", ".cu", ".hpp", ".cuh", ".h"}:
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
            "raw vendor calls outside core/platform.hpp — these compile for one vendor and "
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
        families = {"imgproc", "detection", "reid", "tracking", "mtmc", "eval", "tune"}
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
