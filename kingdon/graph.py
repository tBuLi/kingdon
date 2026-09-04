import math
from collections import defaultdict
from collections.abc import Callable
from types import GeneratorType
import inspect

import anywidget
import traitlets
import pathlib
import numpy as np
import sympy as sp
from sympy.printing.glsl import GLSLPrinter

from kingdon.multivector import MultiVector


TREE_TYPES = (list, tuple)

DEFAULT_STYLE = {
    'width': 'min( 100%, 1024px )',
    'height': 'auto',
    'aspectRatio': '16 / 6',
    'background': 'white',
}


def walker(encoded_generator, tree_types=TREE_TYPES):
    result = []
    for item in encoded_generator:
        if isinstance(item, GeneratorType):
            result.extend(walker(item))
        elif isinstance(item, tree_types):
            result.append(walker(item))
        else:
            result.append(item)
    return result

def encode_ndarray(arr: np.ndarray) -> dict:
    """ Serialize an array. The result is a dict of::

        dtype   numpy type string, always little-endian, e.g. '|u1', '|b1', '<f8'
        shape   tuple of dimensions; the blob is C-contiguous in this shape
        buffer  the raw bytes. ipywidgets pulls these out of the synced state and
                sends them as a binary buffer, so they arrive in js as a DataView.

    `toArray` in graph.js reads this back into the matching js typed arrays, turning our
    struct of arrays inside out, and must stay in sync with the dtypes emitted here.
    """
    arr = np.ascontiguousarray(arr)
    if arr.dtype.byteorder == '>':
        arr = arr.astype(arr.dtype.newbyteorder('<'))

    return {
        'dtype': arr.dtype.str,
        'shape': arr.shape,
        'buffer': arr.tobytes(),
    }

def encode(o, tree_types=TREE_TYPES, root=False, graded=False):
    if root and isinstance(o, tree_types):
        yield from (encode(value, tree_types, graded=graded) for value in o)
    elif isinstance(o, tree_types):
        yield o.__class__(encode(value, tree_types, graded=graded) for value in o)
    elif isinstance(o, MultiVector):
        # A layout with more entries than mv has keys is one with structural constants, which ganja has to be told about.
        keys, values = o._keys, o._values
        if not isinstance(values, np.ndarray) and values and all(isinstance(v, np.ndarray) for v in values):
            values = np.asarray(np.broadcast_arrays(*values))  # send a list of numpy arrays via the ndarray protocol anyway
        values = encode_ndarray(values) if isinstance(values, np.ndarray) else list(values)
        if graded:  # In >6D ganja switches to graded mode.
            yield {'mv': values, 'keys': keys, 'grades': o.grades, 'type': o.__class__.__name__.lower()}
        elif len(keys) != len(o.algebra):
            # If not full mv, also pass the keys and let ganja figure it out.
            yield {'mv': values, 'keys': keys, 'type': o.__class__.__name__.lower()}
        else:
            yield {'mv': values}
    elif isinstance(o, Callable):
        yield encode(o(), tree_types, graded=graded)
    else:
        yield o


class GraphWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "graph.js"

    # Required arguments.
    algebra = traitlets.Instance("kingdon.algebra.Algebra")
    options = traitlets.Dict().tag(sync=True)

    # Properties derived from the required arguments which have to be available to js.
    signature = traitlets.List([]).tag(sync=True)  # Signature of the algebra
    basis = traitlets.List([]).tag(sync=True)      # Basis of the algebra
    key2idx = traitlets.Dict({}).tag(sync=True)    # Conversion from binary keys to indices
    graded = traitlets.Bool({}).tag(sync=True)     # Run ganja.js in graded mode if an up function was provided
    types = traitlets.Dict({}).tag(sync=True)      # Types on the algebra, to use on the js side

    # Properties needed to paint the scene.
    raw_subjects = traitlets.List([])                          # A place to store the original input.
    pre_subjects = traitlets.List([])                          # Store the prepared subjects.
    draggable_points = traitlets.List([]).tag(sync=True)       # points at the first level of nesting are interactive.
    draggable_points_idxs = traitlets.List([]).tag(sync=True)  # indices of the draggable points in the subjects ganja receives.
    subjects = traitlets.List([]).tag(sync=True)               # Result of evaluating pre_subjects.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.algebra.basis:
            raise ValueError('ganja only renders lexicographical basis. Try e.g. Algebra(3,0,1) instead of Algebra.fromname("3DPGA"). See https://github.com/tBuLi/kingdon/issues/141. You can still use PGA types as Algebra(3,0,1, extra_types=[Point, UPoint, EVector, Translation]).')
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, data, buffers):
        """If triggered, reevaluate the subjects. """
        if data["type"] == "update_mvs":
            # TODO: only update those that are callable for better performance?
            self.subjects = self.get_subjects()

    def _get_pre_subjects(self):
        if len(self.raw_subjects) == 1 and not isinstance((s := self.raw_subjects[0]), MultiVector) and isinstance(s, Callable):
            # Assume this to be a function returning a list of suitable subjects.
            pre_subjects = s()
            if not isinstance(pre_subjects, TREE_TYPES):
                pre_subjects = [pre_subjects]
        else:
            pre_subjects = self.raw_subjects
        return pre_subjects

    @traitlets.default('key2idx')
    def get_key2idx(self):
        d = self.algebra.d
        allkeys = list(self.algebra.canon2bin.values())
        if ('up' not in self.options) and d <= 6:
            return {k: i for i, k in enumerate(allkeys)}
        # From >6D, ganja wants graded input. In this case we return indices by grade
        key2idx = defaultdict(dict)
        i = 0
        tot = 0
        for g in range(d + 1):
            num = math.comb(d, g)
            for _ in range(num):
                key2idx[g][allkeys[i]] = i - tot
                i += 1
            tot += num
        return key2idx

    @traitlets.default('signature')
    def get_signature(self):
        return self.algebra.signature

    @traitlets.default('basis')
    def get_basis(self):
        return [b if b != 'e' else '1' for b in self.algebra.canon2bin]

    @traitlets.default('types')
    def get_types(self):
        """ Construct the layouts for the js side. Free values in the layout are ignored."""
        alg = self.algebra
        return {cls.__name__.lower(): {k: v for k, v in layout.items() if v != ...}
                for cls, layout in alg._type_layouts.items()}

    @traitlets.default('pre_subjects')
    def get_pre_subjects(self):
        return self._get_pre_subjects()

    @traitlets.default('subjects')
    def get_subjects(self):
        # Encode all the subjects
        return walker(encode(self._get_pre_subjects(), root=True, graded=self.graded))

    @traitlets.default('draggable_points')
    def get_draggable_points(self):
        if self.options.get('up'):
            return []
        # Extract the draggable points.
        d = self.algebra.d
        points = [s for s in self.pre_subjects if isinstance(s, MultiVector) and not s.shape]
        if self.algebra.r == 1 and (d == 3 or d == 4):  # PGA
            points = [p for p in points if p.grades == (d - 1,) and not p.shape]
        elif self.options.get('conformal'):
            points = [p for p in points if p.grades == (1,) and not p.shape]
        return walker(encode(points, graded=self.graded))

    def _draggable_idxs(self):
        """
        Yield ``(subject index, slot index)`` for every draggable point. The first indexes
        :attr:`pre_subjects`, the second the list of subjects ganja ends up with. They drift
        apart because js spreads a multivector of shape ``(N, ...)`` over ``N`` slots.
        """
        # Extract the draggable points.
        d = self.algebra.d
        if self.algebra.r == 1 and (d == 3 or d == 4):  # PGA
            filter_func = lambda s: isinstance(s, MultiVector) and s.grades == (d - 1,) and not s.shape
        elif self.options.get('conformal'):
            filter_func = lambda s: isinstance(s, MultiVector) and s.grades == (1,) and not s.shape
        else:
            filter_func = lambda s: isinstance(s, MultiVector) and not s.shape
        slot = 0
        for j, (pre_subject, subject) in enumerate(zip(self.pre_subjects, self.subjects)):
            if filter_func(pre_subject):
                yield j, slot
            # Count the encoded subjects, in which callables are already evaluated. Their
            # shape is (nkeys, *mv.shape), so js spreads them over shape[1] slots.
            values = subject.get('mv') if isinstance(subject, dict) else None
            slot += values['shape'][1] if isinstance(values, dict) and len(values['shape']) > 1 else 1

    @traitlets.default('draggable_points_idxs')
    def get_draggable_points_idxs(self):
        if self.options.get('up'):
            return []
        return [slot for _, slot in self._draggable_idxs()]

    @traitlets.default('graded')
    def get_graded(self):
        return 'up' in self.options

    @traitlets.observe('draggable_points')
    def _observe_draggable_points(self, change):
        """ If draggable_points is changed, replace the raw_subjects in place. """
        self.inplacereplace(self.pre_subjects, zip((j for j, _ in self._draggable_idxs()), change['new']))
        self.subjects = self.get_subjects().copy()

    @traitlets.validate("options")
    def _valid_options(self, proposal):
        options = proposal['value']
        if camera := options.get('camera'):
            options['camera'] = list(encode(camera, graded=self.graded))[0]
        if up := options.get('up'):
            sig = inspect.signature(up)
            up_glsl = up(*[sp.Symbol(param) for param in sig.parameters]).map(GLSLPrinter().doprint)
            options['up'] = list(encode(up_glsl, graded=self.graded))[0]['mv']
        style = {**DEFAULT_STYLE, **options.get('style', {})}
        if 'width' in options:
            style['width'] = options['width']
        if 'height' in options:
            style['height'] = options['height']
        style.setdefault('marginLeft', f"calc( (100% - {style['width']}) / 2 )")
        options['style'] = style
        return options

    def inplacereplace(self, old_subjects, new_subjects: list[tuple[int, dict]]):
        """
        Given the old and the new subjects, replace the values inplace iff they have changed.
        """
        for j, new_subject in new_subjects:
            old_subject = old_subjects[j]
            old_vals = old_subject._values
            new_vals = new_subject['mv']
            if len(old_vals) == len(self.algebra):
                # If full mv, we can be quick.
                for j, val in enumerate(new_vals):
                    if old_vals[j] != val:
                        old_vals[j] = val
            else:
                for j, k in enumerate(old_subject._keys):
                    val = new_vals[self.key2idx[k]]
                    if old_vals[j] != val:
                        old_vals[j] = val

    def update(self, *subjects, **options):
        """
        Update the subjects and options. Same API as :meth:`~kingdon.algebra.Algebra.graph`.
        if no `options` are provided, then the existing options are kept; if
        options are provided, then the existing options are replaced.
        """
        with self.hold_sync():  # Only update after all changes are made.
            if options:
                self.options = options
            self.raw_subjects = subjects
            self.pre_subjects = self.get_pre_subjects()
            self.subjects = self.get_subjects()
            # Temporarily disable the observer so we can safely update the draggable points.
            # To do this properly, we should use unobserve and observe to temporarily disable the observer, but that requires
            # access to the ObserveHandler. which we don't seem to have. So modify the list inplace to bypass the observer.
            self.draggable_points[:] = self.get_draggable_points()
            self.draggable_points_idxs = self.get_draggable_points_idxs()
