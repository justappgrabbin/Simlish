"""Exact binary hypercube isometries and safe identity-preserving address transforms.

A cube isometry is a coordinate permutation followed by XOR with a mask.
This module does not claim psychological validity or infer physical DOF.
"""
from dataclasses import dataclass
from itertools import combinations

@dataclass(frozen=True)
class CubeIsometry:
    permutation: tuple[int, ...]
    mask: int = 0

    def __post_init__(self):
        n = len(self.permutation)
        if n < 1 or tuple(sorted(self.permutation)) != tuple(range(n)):
            raise ValueError('permutation must contain each coordinate exactly once')
        if not isinstance(self.mask, int) or not 0 <= self.mask < (1 << n):
            raise ValueError('mask outside cube')

    @property
    def dimension(self): return len(self.permutation)

    def apply(self, state: int) -> int:
        n = self.dimension
        if not isinstance(state, int) or not 0 <= state < (1 << n):
            raise ValueError('state outside cube')
        return sum(((state >> src) & 1) << dest for dest, src in enumerate(self.permutation)) ^ self.mask

    def inverse(self):
        n = self.dimension
        inverse = [0] * n
        for dest, src in enumerate(self.permutation): inverse[src] = dest
        perm = tuple(inverse)
        inverse_mask = sum(((self.mask >> src) & 1) << dest for dest, src in enumerate(perm))
        return CubeIsometry(perm, inverse_mask)

    def compose(self, before):
        """Return self ∘ before."""
        if self.dimension != before.dimension: raise ValueError('dimensions differ')
        perm = tuple(before.permutation[src] for src in self.permutation)
        transformed_mask = sum(((before.mask >> src) & 1) << dest for dest, src in enumerate(self.permutation))
        return CubeIsometry(perm, transformed_mask ^ self.mask)


def hamming(a: int, b: int) -> int:
    if not isinstance(a, int) or not isinstance(b, int) or a < 0 or b < 0:
        raise ValueError('nonnegative integer states required')
    return (a ^ b).bit_count()


def verify_codebook(states, dimension: int, minimum_distance: int = 5):
    if not isinstance(dimension, int) or dimension < 1 or not 1 <= minimum_distance <= dimension:
        raise ValueError('invalid dimension or minimum distance')
    states = tuple(states)
    if len(set(states)) != len(states) or any(type(s) is not int or not 0 <= s < (1 << dimension) for s in states):
        raise ValueError('duplicate or invalid state')
    actual = min((hamming(a,b) for a,b in combinations(states,2)), default=None)
    return {'size': len(states), 'minimum_distance': actual,
            'meets_constraint': actual is None or actual >= minimum_distance,
            'guaranteed_bit_error_correction': (actual-1)//2 if actual is not None else None}


def transform_codebook(states, transform: CubeIsometry, minimum_distance=5):
    before = verify_codebook(states, transform.dimension, minimum_distance)
    transformed = tuple(transform.apply(s) for s in states)
    after = verify_codebook(transformed, transform.dimension, minimum_distance)
    if before['minimum_distance'] != after['minimum_distance']:
        raise AssertionError('isometry failed distance invariant')
    return {'states': transformed, 'before': before, 'after': after,
            'identity_preserved': True, 'mapping': dict(zip(states, transformed))}


def five_axis_address(bits: int, axes: tuple[int, int, int, int, int]):
    """Project five *selected* binary coordinates; not five independent physical DOF."""
    if len(axes) != 5 or len(set(axes)) != 5 or any(type(i) is not int or i < 0 for i in axes):
        raise ValueError('five distinct coordinate indices required')
    if type(bits) is not int or bits < 0 or bits >= 1 << (max(axes)+1):
        raise ValueError('invalid state')
    return dict(zip(('Movement','Evolution','Being','Design','Space'),
                    ((bits >> i) & 1 for i in axes)))
