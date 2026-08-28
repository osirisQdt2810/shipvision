"""The guard against running another checkout's compiled extension."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import shipvision
from shipvision import _native


def _extension(path: Path) -> types.ModuleType:
    module = types.ModuleType("shipvision._C")
    module.__file__ = str(path)
    return module


def _install(monkeypatch: pytest.MonkeyPatch, module: types.ModuleType | None) -> None:
    """Put it where ``from shipvision import _C`` looks — both places.

    Once the real extension has been imported anywhere, ``shipvision._C`` is an *attribute*
    of the package, and the import statement finds it there without consulting
    ``sys.modules``. Patching only the one a test happens to think of is how a test like
    this passes alone and lies in a full run.
    """
    monkeypatch.setitem(sys.modules, "shipvision._C", module)
    if module is None:
        monkeypatch.delattr(shipvision, "_C", raising=False)
    else:
        monkeypatch.setattr(shipvision, "_C", module, raising=False)


@pytest.fixture()
def _clear(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(_native.ALLOW_FOREIGN, raising=False)


class TestAForeignExtensionIsTreatedAsAbsent:
    def test_an_extension_from_another_checkout_is_refused_and_says_so(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, _clear
    ) -> None:
        _install(monkeypatch, _extension(tmp_path / "_C.so"))

        with pytest.warns(RuntimeWarning, match="different checkout"):
            module, reason = _native.load_extension()

        assert module is None
        assert "was built in a different checkout" in (reason or "")

    def test_the_operator_can_keep_it_deliberately(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A cross-tree build is legitimate when somebody chose it; silence is not."""
        foreign = _extension(tmp_path / "_C.so")
        _install(monkeypatch, foreign)
        monkeypatch.setenv(_native.ALLOW_FOREIGN, "1")

        assert _native.load_extension()[0] is foreign


class TestALocalExtensionIsKept:
    def test_one_built_inside_this_package_is_used(
        self, monkeypatch: pytest.MonkeyPatch, _clear
    ) -> None:
        here = Path(shipvision.__file__).parent / "_C.cpython-000.so"
        local = _extension(here)
        _install(monkeypatch, local)

        assert _native.load_extension() == (local, None)


class TestTheHeaderSaysWhichOneIsLive:
    def test_it_names_the_file_when_one_is_loaded(
        self, monkeypatch: pytest.MonkeyPatch, _clear
    ) -> None:
        here = Path(shipvision.__file__).parent / "_C.cpython-000.so"
        _install(monkeypatch, _extension(here))

        assert _native.provenance().endswith("_C.cpython-000.so")

    def test_it_says_absent_with_the_reason_otherwise(
        self, monkeypatch: pytest.MonkeyPatch, _clear
    ) -> None:
        _install(monkeypatch, None)

        assert _native.provenance().startswith("shipvision._C: absent")
