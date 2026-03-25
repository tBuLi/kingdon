from __future__ import annotations

import operator
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Dict, Tuple


@dataclass
class AdditionChains:
    limit: int

    @cached_property
    def minimal_chains(self) -> Dict[int, Tuple[int, ...]]:
        chains = {1: (1,)}
        while any(i not in chains for i in range(1, self.limit + 1)):
            for chain in chains.copy().values():
                right_summand = chain[-1]
                for left_summand in chain:
                    value = left_summand + right_summand
                    if value <= self.limit and value not in chains:
                        chains[value] = (*chain, value)
        return chains

    def __getitem__(self, n: int) -> Tuple[int, ...]:
        return self.minimal_chains[n]

    def __contains__(self, item):
        return self[item]


def power_supply(x, exponents: Tuple[int, ...], operation: Callable = operator.mul):
    """
    Generates powers of a given multivector using the least amount of multiplications.
    For example, to raise a multivector :math:`x` to the power :math:`a = 15`, only 5
    multiplications are needed since :math:`x^{2} = x * x`, :math:`x^{3} = x * x^2`,
    :math:`x^{5} = x^2 * x^3`, :math:`x^{10} = x^5 * x^5`, :math:`x^{15} = x^5 * x^{10}`.
    The :class:`power_supply` uses :class:`AdditionChains` to determine these shortest
    chains.

    When called with only a single integer, e.g. :code:`power_supply(x, 15)`, iterating
    over it yields the above sequence in order; ending with :math:`x^{15}`.

    When called with a sequence of integers, the generator instead returns only the requested terms.

    :param x: The object to be raised to a power.
    :param exponents: When an :code:`int`, this generates the shortest possible way to
        get to :math:`x^a`, where :math:`x`
    """
    if isinstance(exponents, int):
        target = exponents
        addition_chains = AdditionChains(target)
        exponents = addition_chains[target]
    else:
        addition_chains = AdditionChains(max(exponents))

    powers = {1: x}
    for step in exponents:
        if step not in powers:
            chain = addition_chains[step]
            powers[step] = operation(powers[chain[-2]], powers[step - chain[-2]])

        yield powers[step]
