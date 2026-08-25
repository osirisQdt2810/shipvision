#!/usr/bin/env bash
# The offline test invocation, in one place, so ci.yml and pr-pipeline.yml cannot drift.
#
# Extra arguments are appended, which is how a coverage job adds its flags.
set -euo pipefail

# `python` on a CI runner (setup-python provides it), `python3` on a distro that does not
# ship the unversioned name, `$PYTHON` when a caller wants a specific interpreter. Guessing
# wrong fails with "python: not found", which reads like a broken test suite.
PYTHON="${PYTHON:-$(command -v python || command -v python3)}"

# Hide the accelerators, even on a box that has eight of them.
#
# The offline tier is *defined* as the part that runs with no accelerator and no build, and
# the only honest way to check that is to run it that way. Deselecting the markers is not the
# same guarantee: an unmarked test can take a CUDA path by accident, pass on a dev box and
# fail on the runner. That is not hypothetical — it is how `torch.empty(pin_memory=True)`
# reached CI in the sibling repository, where it raises.
export CUDA_VISIBLE_DEVICES=""
export HIP_VISIBLE_DEVICES=""

# And deselect the two opt-in tiers explicitly. `native` needs the compiled extension,
# `gpu` needs a real device; a developer asks for either directly with `pytest -m native`,
# which does not go through this script.
exec "$PYTHON" -m pytest -ra --strict-markers --strict-config \
  -m "not gpu and not native" "$@"
