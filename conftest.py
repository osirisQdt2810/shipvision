"""Root conftest: say which compiled extension is live, in every run's header.

The native path and the numpy path are different code. A run that silently exercised the
wrong one has already happened — an editable install of another checkout resolved
``shipvision._C`` to a ``.so`` from that tree — and it cost a review round, because every
mutation result in it was meaningless. One line in the header makes it visible.
"""

from __future__ import annotations


def pytest_report_header() -> str:
    from shipvision._native import provenance

    return provenance()
