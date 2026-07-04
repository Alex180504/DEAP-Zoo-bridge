"""Streamlit UI for deap-bridge experiment results.

This subpackage is a *pure consumer* of the on-disk artifacts produced
by :mod:`deap_pettingzoo.utils.logging` and processed by
:mod:`deap_pettingzoo.analysis`. Nothing in :mod:`deap_pettingzoo.baselines`
or :mod:`deap_pettingzoo.utils.logging` imports from here, so the
backend stays fully usable without the UI installed.

Entry points
------------

The simplest way to launch the UI is via docker-compose::


Alternatively, the UI service can be started directly via the command line::

    streamlit run deap_pettingzoo/ui/app.py

    or:

    python -m deap_pettingzoo.ui


Optional dependencies (install with ``pip install deap-pettingzoo[ui]``):

- ``streamlit>=1.30``
- ``plotly>=5.0``
- ``pillow>=10.0``
- ``imageio>=2.31``
- transitively, the analysis stack (pandas, scipy, etc.)

Public surface
--------------

The UI's public API is intentionally just :func:`run`, which spawns a
``streamlit`` subprocess pointing at the bundled ``app.py``. Page
modules under :mod:`deap_pettingzoo.ui.pages` are not imported eagerly
because Streamlit discovers them by file name on launch.
"""

from __future__ import annotations

from deap_pettingzoo.ui._launcher import run

__all__ = ["run"]
