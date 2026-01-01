from kingdon import  MultiVector


class BladeMap:
    """
    A map relating blades in two different algebras.

    Can be initialized with an explicit list of blade pairs or by
    automatically aligning blades with the same names between two algebras.

    :param blade_map: List of (multivector, multivector) pairs specifying
        the blade mapping. If None, auto-alignment is used.
    :param map_scalars: If True (default), include scalar mapping.
    :param alg1: First algebra for auto-alignment.
    :param alg2: Second algebra for auto-alignment.
    
    Examples
    --------
    **sub-algebra example:**
    
    >>> from kingdon import Algebra
    >>> pga3d = Algebra(3, 0, 1, start_index=0)
    >>> pga2d = Algebra(2, 0, 1, start_index=0)
    >>> bm = BladeMap(alg1=pga3d, alg2=pga2d)
    >>> bm(pga3d.e1) # Maps e1 from pga3d to pga2d
    >>> bm(pga2d.e1) # Maps e1 from pga2d back to pga3d
    
    **Space-time algebra example (Electromagnetic field decomposition):**
    
    This example shows how to decompose an electromagnetic field bivector in 
    spacetime algebra (STA) into electric and magnetic field components in 
    3D space algebra, based on:
    https://clifford.readthedocs.io/en/latest/tutorials/space-time-algebra.html
    
    >>> from kingdon import Algebra
    >>> sta = Algebra(1, 3, 0, start_index=0)  # spacetime algebra
    >>> vga3 = Algebra(3, 0, 0, start_index=1)  # space algebra
    >>> D = sta.blades  # Dirac basis
    >>> P = vga3.blades  # Pauli basis
    >>> 
    >>> bm = BladeMap([(D.e01, P.e1),
    ...                (D.e02, P.e2),
    ...                (D.e03, P.e3),
    ...                (D.e12, P.e12),
    ...                (D.e23, P.e23),
    ...                (D.e13, P.e13),
    ...                (D.e0123, P.e123)])
    >>> 
    >>> def split(X):  # space-time split
    ...     return bm(X.odd * D.e0 + X.even)
    >>> 
    >>> # Space-time vector decomposition
    >>> X = sta.vector([1, 2, 3, 4])
    >>> xt = bm(X * D.e0)
    >>> x, t = xt.odd, xt.even  # space and time components
    >>> 
    >>> # Electromagnetic field decomposition
    >>> F = sta.bivector([1, 2, 3, 4, 5, 6])
    >>> EiB = split(F)
    >>> E, B = EiB.odd, EiB.even.dual()  # electric and magnetic fields
    >>> 
    >>> # Invariants of the electromagnetic field
    >>> i = P.e123
    >>> split(F**2), E**2 - B**2 + (2*E|B)*i
    """

    def __init__(self, blade_map=None, alg1=None, alg2=None, map_scalars=True):
        if blade_map is None:
            if alg1 is None or alg2 is None:
                raise ValueError("alg1 and alg2 must be provided when blade_map is None")

            blades1 = set(alg1.canon2bin.keys())
            blades2 = set(alg2.canon2bin.keys())
            common_blades = blades1 & blades2

            self.blades_map = []
            for blade_name in sorted(common_blades):
                if blade_name != 'e':
                    mv1 = MultiVector(alg1, **{blade_name: 1})
                    mv2 = MultiVector(alg2, **{blade_name: 1})
                    self.blades_map.append((mv1, mv2))
        else:
            self.blades_map = blade_map

        if map_scalars:
            if blade_map is None:
                s1 = MultiVector(alg1, e=1)
                s2 = MultiVector(alg2, e=1)
            else:
                s1 = MultiVector(self.b1[0].algebra, e=1)
                s2 = MultiVector(self.b2[0].algebra, e=1)
            self.blades_map = [(s1, s2)] + self.blades_map

    @property
    def b1(self):
        return [k[0] for k in self.blades_map]

    @property
    def b2(self):
        return [k[1] for k in self.blades_map]

    @property
    def alg1(self):
        return self.b1[0].algebra

    @property
    def alg2(self):
        return self.b2[0].algebra

    def __call__(self, A):
        """Map a multivector A according to the blade mapping."""
        if A.algebra == self.alg1:
            from_b = self.b1
            to_b = self.b2
        elif A.algebra == self.alg2:
            from_b = self.b2
            to_b = self.b1
        else:
            raise ValueError('A does not belong to either Algebra in this Map')

        result_dict = {}
        for from_mv, to_mv in zip(from_b, to_b):
            coeff = self._extract_coefficient(A, from_mv)
            try:
                import numpy as np
                if isinstance(coeff, np.ndarray):
                    self._add_contribution(result_dict, to_mv, coeff)
                else:
                    if coeff != 0:
                        self._add_contribution(result_dict, to_mv, coeff)
            except ImportError:
                if coeff != 0:
                    self._add_contribution(result_dict, to_mv, coeff)
            except Exception:
                self._add_contribution(result_dict, to_mv, coeff)

        return MultiVector(to_b[0].algebra, **result_dict)

    def _extract_coefficient(self, A, basis_mv):
        """Extract coefficient of basis element from multivector A."""
        if len(basis_mv._keys) != 1:
            return 0
        blade_key = basis_mv._keys[0]
        blade_name = basis_mv.algebra.bin2canon[blade_key]

        A_dict = {A.algebra.bin2canon[key]: value for key, value in zip(A._keys, A._values)}
        return A_dict.get(blade_name, 0)

    def _add_contribution(self, result_dict, to_mv, coeff):
        """Add coefficient contribution to result dictionary."""
        if len(to_mv._keys) == 1:
            blade_key = to_mv._keys[0]
            blade_name = to_mv.algebra.bin2canon[blade_key]
            result_dict[blade_name] = result_dict.get(blade_name, 0) + coeff

