"""Bulgarian Belot's neural stack (Q-GRU, residual belief net, fast policy, bid net),
trained with cardbot-framework's trainers (cardbot.train) through thin adapters."""

import sys as _sys
from pathlib import Path as _Path

# cardbot-framework's Python package, from the submodule, unless it is installed.
try:
    import cardbot as _cardbot  # noqa: F401
except ImportError:
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "cardbot-framework" / "python"))
