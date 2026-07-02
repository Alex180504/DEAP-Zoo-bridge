"""Opt-in QDax (quality-diversity) integration for deap-bridge.

This subpackage is **not imported by the core** and requires the
``qdax`` / ``jax`` / ``flax`` optional extras:

.. code-block:: bash

    pip install 'deap-pettingzoo[qdax]'

What lives here:

* :mod:`.policy` — :class:`FlaxMLPPolicy` (pure JAX MLP topologically
  equivalent to :class:`~deap_pettingzoo.genomes.FixedTopologyGenome`)
  plus :class:`FlaxPolicyAdapter` (satisfies
  :class:`~deap_pettingzoo.baselines.base.PolicyProtocol`) and
  flat-weight\u2194pytree bridges.
* :mod:`.descriptors` — the generic :class:`DescriptorProtocol`
  contract for behavioural-descriptor callables (trajectory-summary
  dict -> 1-D :class:`jnp.ndarray`). Concrete descriptors are supplied
  by the caller via ``descriptor_fn`` (e.g. ``examples/qdax/descriptors.py``).
* :mod:`.scoring_jax` / :mod:`.scoring_numpy` — scoring-function
  builders for Goal 1 (fast JAX-native rollout) and Goal 2 (slow
  PettingZoo-env rollout via the numpy boundary).
* :mod:`.trainer` — :class:`MAPElitesTrainer` and
  :class:`CMAMETrainer` (both :class:`BaselineTrainer` subclasses).
* :mod:`.coevolution` — :class:`QDaxRepertoirePool` (satisfies
  :class:`~deap_pettingzoo.baselines.coevolution.OpponentPool`) and
  :class:`QDaxSideState` (:class:`SideState` subclass) so an EA-vs-QD
  co-evolution experiment is a single-line ``Side`` config change.

Nothing in this module eagerly imports JAX / QDax at package-import
time. Each submodule's heavy dependencies are gated to function
bodies / class constructors so that an ``import deap_pettingzoo``
never pays the JAX import cost.
"""

from __future__ import annotations

_INSTALL_HINT = (
    "QDax integration requires the 'qdax' optional extra. "
    "Install with: pip install 'deap-pettingzoo[qdax]' "
    "(resolves jax, flax, jaxlib, qdax)."
)


def _require_qdax():
    """Import qdax / jax / flax or raise a clear :class:`ImportError`."""
    try:
        import flax  # noqa: F401
        import jax  # noqa: F401
        import qdax  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(_INSTALL_HINT) from exc


__all__ = ["_require_qdax"]
