""" Deprecated: renamed to :mod:`kingdon.reductions`. """
import warnings

from kingdon.reductions import AdditionChains, power_supply, tree_sum  # noqa: F401

warnings.warn("kingdon.powers has been renamed to kingdon.reductions.", FutureWarning, stacklevel=2)
