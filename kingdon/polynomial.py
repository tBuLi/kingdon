import itertools
import math
from collections import defaultdict
from fractions import Fraction

from sympy import Mul, Add, Symbol, RealNumber

from kingdon.powers import power_supply


# ---------------------------------------------------------------------------
# Polynomial CSE helpers (ported from polynomial.js)
# ---------------------------------------------------------------------------



def poly_format(poly_args):
    """Format a raw polynomial args list (or 0) to a Python code string."""
    if poly_args == 0 or poly_args == []:
        return '0'
    terms = []
    for monomial in poly_args:
        coeff = monomial[0]
        factors = monomial[1:]
        if not factors:
            terms.append(str(coeff))
        else:
            parts = []
            for f in factors:
                if isinstance(f, list):
                    inner = poly_format(f)
                    parts.append(f'({inner})')
                elif f != 1:
                    parts.append(str(f))
            if coeff == 1:
                terms.append('*'.join(parts) if parts else '1')
            elif coeff == -1:
                terms.append('-' + '*'.join(parts) if parts else '-1')
            else:
                terms.append('*'.join([str(coeff)] + parts))
    terms.sort(key=lambda t: 1 if t.startswith('-') else 0)
    return '+'.join(terms).replace('+-', '-')


def _find_shared_sums(expr, iso_vars, prelude, start_count=0, sum_map=None):
    """Phase 1: Find shared sums across components and extract them."""
    array_count = sum(1 for e in expr if isinstance(e, list) and e)
    if not iso_vars or array_count < 2:
        return start_count

    res_map = {}
    used = set()
    sum_count = start_count

    for ci, e in enumerate(expr):
        if not isinstance(e, list):
            continue
        for v in iso_vars:
            r_terms, r_idx = [], []
            for ti, t in enumerate(e):
                idx = -1
                for fi in range(1, len(t)):
                    if t[fi] == v:
                        idx = fi
                        break
                if idx < 0:
                    continue
                r_terms.append(t[:idx] + t[idx+1:])
                r_idx.append(ti)
            if len(r_terms) < 2:
                continue

            sizes = [len(r_terms)]
            if len(r_terms) == 4:
                sizes.append(3)

            for sz in sizes:
                subsets = ([list(range(len(r_terms)))] if sz == len(r_terms)
                           else [[1,2,3], [0,2,3], [0,1,3], [0,1,2]])
                for si in subsets:
                    sub = [r_terms[i] for i in si]
                    sub_i = [r_idx[i] for i in si]

                    g = abs(sub[0][0])
                    for i in range(1, len(sub)):
                        g = math.gcd(abs(int(g)), abs(int(sub[i][0])))

                    norm = [[t[0] // g if isinstance(t[0], int) else t[0] / g, *t[1:]]
                            for t in sub]
                    norm.sort(key=lambda t: ','.join(str(x) for x in t[1:]))
                    sign = 1
                    if norm[0][0] < 0:
                        norm = [[-t[0], *t[1:]] for t in norm]
                        sign = -1

                    key = '|'.join(','.join(str(x) for x in t) for t in norm)
                    if key not in res_map:
                        res_map[key] = []
                    res_map[key].append({
                        'comp': ci, 'v': v, 'sign': sign, 'gcd': g,
                        'idx': sub_i, 'norm': [list(t) for t in norm]
                    })

    cands = []
    for key, occs in res_map.items():
        cs = set(o['comp'] for o in occs)
        if len(cs) >= 2:
            cands.append({'key': key, 'occs': occs, 'score': len(cs) * 10 + len(occs)})
    cands.sort(key=lambda c: -c['score'])

    replacements = []
    for cand in cands:
        valid = [o for o in cand['occs']
                 if not any((o['comp'], i) in used for i in o['idx'])]
        vc = set(o['comp'] for o in valid)
        if len(vc) < 2:
            continue

        sn = 't' + str(sum_count)
        sum_count += 1
        norm = cand['occs'][0]['norm']
        prelude.append((sn, [list(t) for t in norm]))

        if (sum_map is not None and len(norm) == 2
                and len(norm[0]) == 2 and len(norm[1]) == 2):
            # offset_sign: +1 for difference (t=a-b → a=t+b), -1 for sum (t=a+b → a=t-b)
            offset_sign = 1 if norm[1][0] < 0 else -1
            sum_map[norm[0][1]] = {'tn': sn, 'offset': norm[1][1], 'offset_sign': offset_sign}

        for occ in valid:
            for i in occ['idx']:
                used.add((occ['comp'], i))
            replacements.append({
                'comp': occ['comp'],
                'indices': occ['idx'],
                'term': [occ['gcd'] * occ['sign'], occ['v'], sn]
            })

    repls_by_comp = defaultdict(list)
    for r in replacements:
        repls_by_comp[r['comp']].append(r)

    for ci in range(len(expr)):
        e = expr[ci]
        if not isinstance(e, list):
            continue
        repls = repls_by_comp[ci]
        if not repls:
            continue
        remove_set = set()
        for r in repls:
            for i in r['indices']:
                remove_set.add(i)
        new_terms = [t for i, t in enumerate(e) if i not in remove_set]
        for r in repls:
            new_terms.append(r['term'])
        e.clear()
        e.extend(new_terms)

    return sum_count


def _isolate(expr, iso_list):
    """Phase 2: Factor out variables in iso_list."""
    for p in iso_list:
        for e in expr:
            if not isinstance(e, list):
                continue

            terms_with_p, terms_without_p = [], []
            for product in e:
                if isinstance(p, str):
                    # String variable: skip nested terms, search from position 1
                    if isinstance(product[-1], list):
                        terms_without_p.append(product)
                        continue
                    try:
                        idx_p = product.index(p, 1)
                    except ValueError:
                        terms_without_p.append(product)
                        continue
                    r = product[:idx_p] + product[idx_p+1:]
                    if len(r) > 1 and isinstance(r[0], str):
                        r = [1] + r
                    terms_with_p.append(list(r))
                else:
                    # Numeric p: match coefficient at position 0 (like JS indexOf)
                    coeff = product[0] if isinstance(product[0], (int, float, Fraction)) else None
                    if coeff == p:
                        # Positive: remove coefficient, prepend 1 if next is string
                        r = list(product[1:])
                        if r and isinstance(r[0], str):
                            r = [1] + r
                        terms_with_p.append(r)
                    elif coeff is not None and coeff == -p:
                        # Negative: keep full term, map -p to -1
                        r = list(product)
                        r[0] = -1
                        terms_with_p.append(r)
                    else:
                        terms_without_p.append(product)

            if len(terms_with_p) <= 1:
                continue

            # Find common factors across all terms_with_p
            common_count, common_elem = {}, {}
            for t in terms_with_p:
                this_run = set()
                for f in t:
                    n = str(f).lstrip('-')
                    if n in ('1', ''):
                        continue
                    if n not in this_run:
                        common_count[n] = common_count.get(n, 0) + 1
                        common_elem[n] = (-f if isinstance(f, (int, float)) and f < 0 else f)
                        this_run.add(n)
            common = [common_elem[n] for n, c in common_count.items()
                      if c == len(terms_with_p)]

            if common:
                for t in terms_with_p:
                    for cf in common:
                        if cf == p or cf == 1:
                            continue
                        if cf in t:
                            idx = t.index(cf)
                            if idx == 0:
                                t[idx] = 1
                            else:
                                t.pop(idx)
                        elif isinstance(cf, (int, float)) and -cf in t:
                            idx = t.index(-cf)
                            if idx == 0:
                                t[idx] = -1
                            else:
                                t.pop(idx)

            # Factor out -1 if all inner terms negative
            all_neg = not any(
                isinstance(t[0], str) or (isinstance(t[0], (int, float)) and t[0] > 0)
                for t in terms_with_p
            )
            sign = []
            if all_neg:
                sign = [-1]
                for t in terms_with_p:
                    t[0] *= -1

            cf = [x for x in common if x != p and x != 1]
            has_num_coeff = (cf and not isinstance(cf[0], str)) or bool(sign)
            prefix = [] if has_num_coeff else [1]
            new_term = prefix + cf + sign + [p, terms_with_p]

            e.clear()
            e.extend(terms_without_p + [new_term])


def _walk_terms(expr):
    """Yield all leaf terms in a (possibly nested) polynomial structure."""
    for term in expr:
        if term and isinstance(term[-1], list):
            yield from _walk_terms(term[-1])
        else:
            yield term


def _find_shared_products(expr, prot, prelude):
    """Phase 3: Find repeated factor pairs and extract them."""
    prods = {}
    prot_set = set(prot or [])

    def count_pairs(term):
        seen = set()
        for i in range(1, len(term) - 1):
            for j in range(i + 1, len(term)):
                ti, tj = term[i], term[j]
                if isinstance(ti, list) or isinstance(tj, list):
                    continue
                if '(' in str(ti) or '(' in str(tj):
                    continue
                if ti in prot_set or tj in prot_set:
                    continue
                key = (ti, tj)
                if key not in seen:
                    seen.add(key)
                    prods[key] = prods.get(key, 0) + 1

    for e in expr:
        if isinstance(e, list):
            for term in _walk_terms(e):
                count_pairs(term)

    prod_list = [k for k, v in prods.items() if v > 1]

    def substitute_pairs(term):
        i = 1
        while i < len(term) - 1:
            j = i + 1
            while j < len(term):
                ti, tj = term[i], term[j]
                if isinstance(ti, list) or isinstance(tj, list):
                    j += 1
                    continue
                if '(' in str(ti) or '(' in str(tj):
                    j += 1
                    continue
                if prods.get((ti, tj), 0) > 1:
                    combined = f'{ti}{tj}'
                    term[i] = combined
                    term.pop(j)
                else:
                    j += 1
            i += 1

    for e in expr:
        if isinstance(e, list):
            for term in _walk_terms(e):
                substitute_pairs(term)

    for k in prod_list:
        combined = f'{k[0]}{k[1]}'
        prelude.append((combined, [[1, k[0], k[1]]]))


def _substitute_extracted(expr, sum_map):
    """Phase 4: Substitute extracted sums to reveal more shared structure."""
    for ci in range(len(expr)):
        e = expr[ci]
        if not isinstance(e, list):
            continue
        new_terms, changed = [], False
        for t in e:
            sub_idx = -1
            for fi in range(1, len(t)):
                if t[fi] in sum_map:
                    sub_idx = fi
                    break
            if sub_idx < 0:
                new_terms.append(t)
                continue
            changed = True
            info = sum_map[t[sub_idx]]
            coeff = t[0]
            rest = t[1:sub_idx] + t[sub_idx+1:]
            # offset_sign: +1 for difference (t=a-b → a=t+b, both positive)
            #              -1 for sum (t=a+b → a=t-b, offset is negated)
            offset_sign = info.get('offset_sign', 1)
            new_terms.append([coeff, *sorted(rest + [info['tn']])])
            new_terms.append([coeff * offset_sign, *sorted(rest + [info['offset']])])
        if not changed:
            continue
        simplified = sum((Polynomial([t]) for t in new_terms), Polynomial([]))
        if not simplified:
            e.clear()
        else:
            e.clear()
            e.extend(simplified.args)


def _detect_linear_deps(expr):
    """Phase 5: Detect if heaviest component is a linear combo of others."""
    norm = []
    for e in expr:
        if not isinstance(e, list):
            norm.append(e)
        else:
            norm.append(sum((Polynomial([t]) for t in _walk_terms(e)), Polynomial([])))

    heaviest, max_weight = -1, 0
    for i, n in enumerate(norm):
        if not isinstance(n, Polynomial):
            continue
        w = sum(len(t) for t in n.args)
        if w > max_weight:
            max_weight = w
            heaviest = i

    if heaviest < 0 or max_weight <= 6:
        return None

    other_vars = set()
    for i, n in enumerate(norm):
        if i == heaviest or not isinstance(n, Polynomial):
            continue
        for t in n.args:
            for j in range(1, len(t)):
                other_vars.add(t[j])

    exclusive_vars = set()
    for t in norm[heaviest].args:
        for j in range(1, len(t)):
            if t[j] not in other_vars:
                exclusive_vars.add(t[j])

    if not exclusive_vars:
        return None

    remainder = norm[heaviest]
    deps, used_comps = [], set()

    for cv in exclusive_vars:
        for oi in range(len(norm)):
            if oi == heaviest or oi in used_comps or not isinstance(norm[oi], Polynomial):
                continue
            prod_p = norm[oi] * Polynomial([[1, cv]])

            r_plus = remainder + prod_p
            r_minus = remainder - prod_p

            cur_len = len(remainder)
            plus_len = len(r_plus)
            minus_len = len(r_minus)

            if plus_len < cur_len:
                remainder = r_plus
                deps.append({'cv': cv, 'comp': oi, 'sign': 1})
                used_comps.add(oi)
                break
            elif minus_len < cur_len:
                remainder = r_minus
                deps.append({'cv': cv, 'comp': oi, 'sign': -1})
                used_comps.add(oi)
                break

    if remainder != 0:
        return None
    return {'heaviest': heaviest, 'deps': deps}


def poly_cse(expr, prot=None, iso=None):
    """
    Common Subexpression Elimination for raw polynomial args lists.

    Ported from polynomial.js.

    :param expr: list of raw polynomial args lists (each [[coeff, var, ...], ...] or 0).
                 Modified in-place.
    :param prot: protected variable names (won't be combined in products).
    :param iso: variable names to use for sum detection and isolation.
    :return: (prelude, expr) where prelude is a list of assignment strings.
    """
    if not isinstance(expr, list):
        return [], expr

    # Shallow-copy the outer list and each component so in-place mutations
    # in the CSE phases don't affect the caller's data.
    expr = [list(e) if isinstance(e, list) else e for e in expr]

    prelude = []
    iso_vars = [x for x in (iso or []) if isinstance(x, str)]
    iso_nums = [x for x in (iso or []) if not isinstance(x, str)]

    # Phase 1: Find shared sums
    sum_map = {}
    has_mixed = _find_shared_sums(expr, iso_vars, prelude, 0, sum_map)

    # Phase 2: Substitute extracted sums and find more shared structure
    if has_mixed and sum_map:
        _substitute_extracted(expr, sum_map)
        t_vars = list(set(v['tn'] for v in sum_map.values()))
        sum_map2 = {}
        r2 = _find_shared_sums(expr, t_vars, prelude, has_mixed, sum_map2)
        if r2 > has_mixed:
            has_mixed = r2
        if sum_map2:
            _substitute_extracted(expr, sum_map2)

    # Phase 3: Detect linear dependencies (before isolation)
    dep = _detect_linear_deps(expr) if has_mixed else None

    # Phase 4: Isolate variables
    iso_list = (list(reversed(iso_vars)) + iso_nums if has_mixed
                else list(prot or []) + list(reversed(iso_vars)) + iso_nums)
    _isolate(expr, iso_list)

    # Apply linear dependencies (after isolation)
    if dep:
        for d in dep['deps']:
            rn = 'u' + str(d['comp'])
            prelude.append((rn, list(expr[d['comp']])))
            expr[d['comp']] = [[1, rn]]
        expr[dep['heaviest']] = [[-d['sign'], d['cv'], 'u' + str(d['comp'])]
                                  for d in dep['deps']]

    # Phase 5: Find shared products
    _find_shared_products(expr, prot, prelude)

    return prelude, expr

def compare(a, b):
    if a is None: return 1
    if b is None: return -1

    la = len(a)
    lb = len(b)
    l = min(la, lb)
    for i in range(1, l):
        if a[i] < b[i]: return -1
        elif a[i] > b[i]: return 1
    return la - lb


class mathstr(str):
    """ Lightweight string subclass that overloads maths operators to form expressions. """
    def __add__(self, other: str):
        if other[0] == '-':
            return self.__class__(f'{self}{other}')
        return self.__class__(f'{self}+{other}')

    def __sub__(self, other: str):
        if other[0] == '-':
            return self.__class__(f'{self}+{other[1:]}')
        return self.__class__(f'{self}-{other}')

    def __neg__(self):
        if self[0] == '-':
            return self.__class__(self[1:])
        return self.__class__('-'+self)

    def __mul__(self, other: str):
        if other[0] != '-':
            return self.__class__(f'{self}*{other}')
        elif self[0] == '-':
            return self.__class__(f'{self[1:]}*{other[1:]}')
        return self.__class__(f'-{self}*{other[1:]}')

    def __pow__(self, power):
        if power == 0.5:
            return self.__class__(f'({self}**(1/2))')  # SymPy conversion needs 1/2 instead of 0.5
        return self.__class__(f'({self}**{power})')


class Polynomial:

    def __init__(self, coeff):
        if isinstance(coeff, self.__class__):
            self.args = coeff.args
        elif isinstance(coeff, (list, tuple)):
            self.args = coeff
        elif isinstance(coeff, (int, float)):
            self.args = [[coeff]]
        elif isinstance(coeff, str):
            self.args = [[1, coeff]] if coeff[0] != "-" else [[-1, coeff[1:]]]

    @classmethod
    def fromname(cls, name):
        return cls([[1, name]])

    def __len__(self):
        return len(self.args)

    def __getitem__(self, item):
        return self.args[item]

    def __eq__(self, other):
        if other == 0 and (not self.args or self.args == [[0]]): return True
        if other == 1 and self.args == [[1]]: return True
        if self.__class__ != other.__class__: return False
        return self.args == other.args

    def __add__(self, other):
        if other == 0:
            return self
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        ai = bi = 0
        al = len(self)
        bl = len(other)
        res = []

        while not (ai == al and bi == bl):
            ea = self[ai] if ai < al else None
            eb = other[bi] if bi < bl else None
            diff = compare(ea, eb)
            if diff < 0:
                res.append(ea)
                ai += 1
            elif diff > 0:
                res.append(eb)
                bi += 1
            else:
                ea = ea.copy()
                ea[0] += eb[0]
                if ea[0] != 0:
                    res.append(ea)
                ai += 1
                bi += 1
        return self.__class__(res)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if self == 0 or other == 0:
            return self.__class__([])

        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        res = Polynomial([])
        al = len(self)
        bl = len(other)
        for ai, bi in itertools.product(range(0, al), range(0, bl)):
            A = self[ai]
            B = other[bi]
            C = [A[0] * B[0]]
            i = 1
            j = 1
            while i < len(A) or j < len(B):
                ea = A[i] if i < len(A) else None
                eb = B[j] if j < len(B) else None
                # if ea is None and eb is None: break
                if eb is None or (ea is not None and ea < eb):
                    if isinstance(ea, str): C.append(ea)
                    else: C[0] *= ea
                    i += 1
                else:
                    if isinstance(eb, str): C.append(eb)
                    else: C[0] *= eb
                    j += 1
            res = res + Polynomial([C])
        return Polynomial(res)

    __rmul__ = __mul__

    def __neg__(self):
        return self.__class__([[-monomial[0], *monomial[1:]] for monomial in self.args])

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, power, modulo=None):
        *_, last = power_supply(self, power)
        return last

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return RationalPolynomial(self, other)
        # Assume scalar
        return self * (1 / other)

    def __str__(self):
        return poly_format(self.args)

    def tosympy(self):
        """ Return a sympy version of this Polynomial. """
        def _to_sym(s):
            if isinstance(s, list):
                return Polynomial(s).tosympy()
            return Symbol(s) if isinstance(s, str) else s
        preprocessed = (monomial if len(monomial) == 1 else monomial[1:] if monomial[0] == 1 else monomial
                        for monomial in self.args)
        sympified = ([_to_sym(s) for s in monomial] for monomial in preprocessed)
        terms = (Mul(*monomial, evaluate=True) for monomial in sympified)
        res = Add(*terms, evaluate=True)
        return res

    def __bool__(self):
        if len(self.args) == 1:
            return bool(self.args[0][0])
        return bool(self.args)


class RationalPolynomial:
    def __init__(self, numer, denom=None):
        if isinstance(numer, self.__class__):
            orig = numer
            numer = orig.numer
            denom = orig.denom
        elif isinstance(numer, (list, tuple)):
            numer = Polynomial(numer)
        if denom is None:
            denom = Polynomial([[1]])
        elif isinstance(denom, (list, tuple)):
            denom = Polynomial(denom)
        self.numer = numer
        self.denom = denom

    @classmethod
    def fromname(cls, name):
        return cls([[1, name]])

    def __eq__(self, other):
        if other == 0 and (self.numer == 0): return True
        if other == 1 and (self.numer == 1 and self.denom == 1): return True
        if self.__class__ != other.__class__: return False
        return self.numer == other.numer and self.denom == other.denom

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)

        if other == 0: return self
        if self == 0: return other

        na, da = self.numer, self.denom
        nb, db = other.numer, other.denom

        if len(da) == len(db) and da == db:
            nn = na + nb
            nd = da
        else:
            nn, nd = na * db + nb * da, da * db

        if nn == 0: return RationalPolynomial([])
        if len(nn) == len(nd) and nn == nd: return RationalPolynomial([[1]])
        return RationalPolynomial(nn, nd)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__([[other]])

        if self == 0: return self
        if other == 0: return other
        if other == 1: return self
        if self == 1: return other

        na, da = self.numer, self.denom
        nb, db = other.numer, other.denom
        numer, denom = na * nb, da * db

        if numer == 0: return RationalPolynomial([[0]])
        if len(numer) == len(denom) and numer == denom: return RationalPolynomial([[1]])
        if len(numer) == 1 and len(denom) == 1:
            # Remove common factors from simple expressions
            fl1, fl2 = numer[0], denom[0]
            nnn, nnd = [fl1[0]], [fl2[0]]
            p1 = p2 = 1
            while p1 < len(fl1) or p2 < len(fl2):
                f1 = fl1[p1] if p1 < len(fl1) else None
                f2 = fl2[p2] if p2 < len(fl2) else None
                if f1 == f2:
                    p1 += 1; p2 += 1; continue;
                if f2 is None or (f1 is not None and f1 < f2):
                    nnn.append(f1); p1 += 1;
                else:
                    nnd.append(f2); p2 += 1;
            return self.__class__([nnn], [nnd])
        return self.__class__(numer, denom)

    __rmul__ = __mul__

    def inv(self):
        if self == 0: return 0
        return self.__class__(self.denom, self.numer)

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            return self * other.inv()
        return self.__class__(self.numer / other, self.denom)

    def __rtruediv__(self, other):
        return self.__class__(other * self.denom, self.numer)

    def __neg__(self):
        return self.__class__(-self.numer, self.denom)

    def __pos__(self):
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, power, modulo=None):
        if power < 0:
            *_, last = power_supply(self, -power)
            return 1 / last
        if power == 0.5:
            return self.fromname(mathstr(self)**0.5)
        *_, last = power_supply(self, power)
        return last

    def __str__(self):
        numer_str = f"({self.numer})" if len(self.numer) > 1 else f"{self.numer}"
        if self.denom.args == [[1]]:
            return numer_str
        denom_str = f"({self.denom})" if len(self.denom) > 1 else f"{self.denom}"
        return f"(({numer_str}) / ({denom_str}))"

    def tosympy(self):
        """ Return a sympy version of this Polynomial. """
        return self.numer.tosympy() / self.denom.tosympy()

    def __bool__(self):
        return self.numer.__bool__()
