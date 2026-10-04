"""Dream-RSI: Recursive Self-Improvement through Evolving Worlds (local reproduction).

Package layout (see AGENTS.md):

* :mod:`src.core`   -- discovery-tree data structures and the LLM client
* :mod:`src.policy` -- the evolving exploration policy (``search_policy.py``)
* :mod:`src.engine` -- the real-world explorer and the offline Dream Engine
* :mod:`src.meta`   -- the meta-optimizer that mutates the policy
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
