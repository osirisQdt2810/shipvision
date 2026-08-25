"""Optuna is optional, and its absence must be a typed failure.

The guarantee has two halves and both need testing: the parts that do not need optuna stay
importable without it, and the parts that do fail with
:class:`~shipvision.errors.BackendUnavailableError` naming the install command rather than an
ImportError raised from four frames down inside a study.

The second half is tested twice — once in process by poisoning ``sys.modules``, which is fast
and precise, and once in a fresh interpreter with a blocking import hook, which is the only way
to be sure the real answer is not being masked by the optuna that *is* installed in this
environment.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from shipvision.errors import BackendUnavailableError, ShipVisionError

BLOCKED = textwrap.dedent(
    """
    import sys


    class Block:
        \"\"\"A meta-path finder that makes one module un-importable.\"\"\"

        def find_spec(self, name, path=None, target=None):
            if name == "optuna" or name.startswith("optuna."):
                raise ImportError("optuna is blocked for this test")
            return None


    sys.meta_path.insert(0, Block())
    for name in [m for m in sys.modules if m.startswith("optuna")]:
        del sys.modules[name]
    {body}
    """
)


def run_without_optuna(body: str) -> subprocess.CompletedProcess[str]:
    """Run ``body`` in a fresh interpreter where importing optuna raises."""
    return subprocess.run(
        [sys.executable, "-c", BLOCKED.format(body=textwrap.indent(body, ""))],
        capture_output=True,
        text=True,
        check=False,
    )


class TestTheImportableHalf:
    def test_the_package_imports_without_optuna(self) -> None:
        """The search spaces and the objective are pure numpy, and they are the parts worth unit
        testing. Making the package unimportable without a sampler would move the whole tuning
        test suite into an optional tier."""
        result = run_without_optuna("import shipvision.tune; print('ok')")

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_the_search_spaces_are_usable_without_optuna(self) -> None:
        result = run_without_optuna(
            "from shipvision.tune import space_for; print(len(space_for('bytetrack')))"
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "6"

    def test_the_objective_scores_a_case_without_optuna(self) -> None:
        """The line that would have been an ImportError in a naive design is the one a CI
        machine without the extra actually runs."""
        result = run_without_optuna(
            textwrap.dedent(
                """
                import numpy as np
                from shipvision.eval.sequence import EvaluationCase, ObjectFrame, TrackSequence
                from shipvision.tune import Objective
                from shipvision.types import Detection, Detections, FrameTag

                box = np.array([0.0, 0.0, 30.0, 60.0], dtype=np.float32)
                gt = TrackSequence(
                    name="s",
                    frames=tuple(
                        ObjectFrame(frame_id=t, ids=np.array([1]), boxes=box[None, :])
                        for t in (1, 2, 3)
                    ),
                    length=3,
                )
                case = EvaluationCase(
                    name="s",
                    detections=tuple(
                        Detections(
                            tag=FrameTag(camera_id="s", frame_id=t),
                            items=[Detection(box=box, score=0.9)],
                        )
                        for t in (1, 2, 3)
                    ),
                    ground_truth=gt,
                )
                print(round(Objective("sort", (case,)).score({"min_hits": 1}), 6))
                """
            )
        )

        assert result.returncode == 0, result.stderr
        assert float(result.stdout.strip()) > 0.9


class TestTheOptunaDependentHalf:
    def test_importing_run_study_raises_the_typed_error(self) -> None:
        """The failure a caller sees is on the line the caller wrote."""
        result = run_without_optuna(
            textwrap.dedent(
                """
                from shipvision.errors import BackendUnavailableError
                try:
                    from shipvision.tune import run_study
                except BackendUnavailableError as error:
                    print("typed:", "shipvision[tune]" in str(error))
                else:
                    print("NOT RAISED")
                """
            )
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "typed: True"

    def test_importing_the_study_module_raises_the_typed_error(self) -> None:
        result = run_without_optuna(
            textwrap.dedent(
                """
                from shipvision.errors import BackendUnavailableError
                try:
                    import shipvision.tune.study
                except BackendUnavailableError:
                    print("typed")
                else:
                    print("NOT RAISED")
                """
            )
        )

        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "typed"

    def test_the_guard_raises_in_process_when_the_import_fails(self, monkeypatch) -> None:
        """The fast version, so the property is checked on every run rather than only where a
        subprocess is cheap. ``None`` in ``sys.modules`` is what Python itself uses to mark an
        import as failed, and ``import optuna`` then raises ImportError."""
        from shipvision.tune._optuna import require_optuna

        monkeypatch.setitem(sys.modules, "optuna", None)

        with pytest.raises(BackendUnavailableError) as raised:
            require_optuna()

        assert "shipvision[tune]" in str(raised.value)
        assert isinstance(raised.value, ShipVisionError)

    def test_an_unknown_attribute_is_still_an_attribute_error(self) -> None:
        """The lazy resolver must not turn every typo into a dependency complaint."""
        import shipvision.tune as tune

        name = "run_stydy"  # a plausible typo, spelled out so ruff sees a dynamic access

        with pytest.raises(AttributeError, match="has no attribute"):
            getattr(tune, name)

    def test_the_lazy_names_are_advertised_in_dir(self) -> None:
        import shipvision.tune as tune

        assert {"run_study", "StudyResult"} <= set(dir(tune))
