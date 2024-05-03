from collections.abc import Callable
from types import GeneratorType
from typing import List, Tuple

import anywidget
import traitlets
import pathlib
import numpy as np

from kingdon.multivector import MultiVector


TREE_TYPES = (list, tuple)


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


def encode(o, tree_types=TREE_TYPES, root=False):
    if root and isinstance(o, tree_types):
        yield from (encode(value, tree_types) for value in o)
    elif isinstance(o, tree_types):
        yield o.__class__(encode(value, tree_types) for value in o)
    elif isinstance(o, MultiVector) and len(o.shape) > 1:
        #     if isinstance(o._values, np.ndarray):
        #         ovals = o._values.T
        #         yield {'mvs': ovals.tobytes(), 'shape': ovals.shape}
        #     else:
        yield from (encode(value) for value in o.itermv())
    elif isinstance(o, MultiVector):
        values = (
            o._values.tobytes()
            if isinstance(o._values, np.ndarray)
            else o._values.copy()
        )
        if len(o) != len(o.algebra):
            # If not full mv, also pass the keys and let ganja figure it out.
            yield {"mv": values, "keys": o._keys}
        else:
            yield {"mv": values}
    elif isinstance(o, Callable):
        yield encode(o(), tree_types)
    else:
        yield o


class GraphWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "graph.js"

    # Required arguments.
    algebra = traitlets.Instance("kingdon.algebra.Algebra")
    options = traitlets.Dict().tag(sync=True)

    # Properties derived from the required arguments which have to be available to js.
    signature = traitlets.List([]).tag(sync=True)  # Signature of the algebra
    cayley = traitlets.List([]).tag(sync=True)  # Cayley table of the algebra
    key2idx = traitlets.Dict({}).tag(
        sync=True
    )  # Conversion from binary keys to indices

    # Properties needed to paint the scene.
    raw_subjects = traitlets.List([])  # A place to store the original input.
    pre_subjects = traitlets.List([])  # Store the prepared subjects.
    draggable_points = traitlets.List([]).tag(
        sync=True
    )  # points at the first level of nesting are interactive.
    draggable_points_idxs = traitlets.List([]).tag(
        sync=True
    )  # indices of the draggable points in pre_subjects.
    subjects = traitlets.List([]).tag(sync=True)  # Result of evaluating pre_subjects.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, data, buffers):
        """If triggered, reevaluate the subjects."""
        if data["type"] == "update_mvs":
            # TODO: only update those that are callable for better performance?
            self.subjects = self.get_subjects()

    def _get_pre_subjects(self):
        if (
            len(self.raw_subjects) == 1
            and not isinstance((s := self.raw_subjects[0]), MultiVector)
            and isinstance(s, Callable)
        ):
            # Assume this to be a function returning a list of suitable subjects.
            pre_subjects = s()
            if not isinstance(pre_subjects, TREE_TYPES):
                pre_subjects = [pre_subjects]
        else:
            pre_subjects = self.raw_subjects
        return pre_subjects

    @traitlets.default("key2idx")
    def get_key2idx(self):
        return {k: i for i, k in enumerate(self.algebra.canon2bin.values())}

    @traitlets.default("signature")
    def get_signature(self):
        return [int(s) for s in self.algebra.signature]

    @traitlets.default("cayley")
    def get_cayley(self):
        cayley_table = [
            [
                s if (s := self.algebra.cayley[eJ, eI])[-1] != "e" else f"{s[:-1]}1"
                for eI in self.algebra.canon2bin
            ]
            for eJ in self.algebra.canon2bin
        ]
        return cayley_table

    @traitlets.default("pre_subjects")
    def get_pre_subjects(self):
        return self._get_pre_subjects()

    @traitlets.default("subjects")
    def get_subjects(self):
        # Encode all the subjects
        return walker(encode(self._get_pre_subjects(), root=True))

    @traitlets.default("draggable_points")
    def get_draggable_points(self):
        # Extract the draggable points. TODO: pseudovectors only?
        return walker(
            encode([s for s in self.pre_subjects if isinstance(s, MultiVector)])
        )

    @traitlets.default("draggable_points_idxs")
    def get_draggable_points_idxs(self):
        # Extract the draggable points. TODO: pseudovectors only?
        return [
            j for j, s in enumerate(self.pre_subjects) if isinstance(s, MultiVector)
        ]

    @traitlets.observe("draggable_points")
    def _observe_draggable_points(self, change):
        """If draggable_points is changed, replace the raw_subjects in place."""
        self.inplacereplace(
            self.pre_subjects, zip(self.draggable_points_idxs, change["new"])
        )
        self.subjects = self.get_subjects().copy()

    @traitlets.validate("options")
    def _valid_options(self, proposal):
        options = proposal["value"]
        if "camera" in options:
            options["camera"] = list(encode(options["camera"]))[0]
        return options

    def inplacereplace(self, old_subjects, new_subjects: List[Tuple[int, dict]]):
        """
        Given the old and the new subjects, replace the values inplace iff they have changed.
        """
        for j, new_subject in new_subjects:
            old_subject = old_subjects[j]
            old_vals = old_subject._values
            new_vals = new_subject["mv"]
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


"""
The colormaps below were generated with the following:

```
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
# Generate a list of colors from the colormap
def colors(name='Set2', N=None):
    cmap = plt.get_cmap(name)
    if N is None:
        N = cmap.N
    return [int(to_hex(cmap(i / N))[1:],16) for i in range(N)]

cmap_names = [name for name in plt.colormaps() if not name.endswith('_r')]
cmaps = {name:colors(name=name) for name in cmap_names if len(colors(name=name))<30 }
```
"""

colormaps = {
    "Accent": [
        8374655,
        12496596,
        16629894,
        16777113,
        3697840,
        15729279,
        12540695,
        6710886,
    ],
    "Dark2": [
        1810039,
        14245634,
        7696563,
        15149450,
        6727198,
        15117058,
        10909213,
        6710886,
    ],
    "Paired": [
        10931939,
        2062516,
        11722634,
        3383340,
        16489113,
        14883356,
        16629615,
        16744192,
        13284054,
        6962586,
        16777113,
        11622696,
    ],
    "Pastel1": [
        16495790,
        11783651,
        13429701,
        14601188,
        16701862,
        16777164,
        15063229,
        16636652,
        15921906,
    ],
    "Pastel2": [
        11789005,
        16633260,
        13358568,
        16042724,
        15136201,
        16773806,
        15852236,
        13421772,
    ],
    "Set1": [
        14948892,
        3636920,
        5091146,
        9981603,
        16744192,
        16777011,
        10901032,
        16220607,
        10066329,
    ],
    "Set2": [
        6734501,
        16551266,
        9281739,
        15174339,
        10934356,
        16767279,
        15058068,
        11776947,
    ],
    "Set3": [
        9294791,
        16777139,
        12499674,
        16482418,
        8434131,
        16626786,
        11787881,
        16567781,
        14277081,
        12353725,
        13429701,
        16772463,
    ],
    "tab10": [
        2062260,
        16744206,
        2924588,
        14034728,
        9725885,
        9197131,
        14907330,
        8355711,
        12369186,
        1556175,
    ],
    "tab20": [
        2062260,
        11454440,
        16744206,
        16759672,
        2924588,
        10018698,
        14034728,
        16750742,
        9725885,
        12955861,
        9197131,
        12885140,
        14907330,
        16234194,
        8355711,
        13092807,
        12369186,
        14408589,
        1556175,
        10410725,
    ],
    "tab20b": [
        3750777,
        5395619,
        7040719,
        10264286,
        6519097,
        9216594,
        11915115,
        13556636,
        9202993,
        12426809,
        15186514,
        15190932,
        8666169,
        11356490,
        14049643,
        15177372,
        8077683,
        10834324,
        13528509,
        14589654,
    ],
    "tab20c": [
        3244733,
        7057110,
        10406625,
        13032431,
        15095053,
        16616764,
        16625259,
        16634018,
        3253076,
        7652470,
        10607003,
        13101504,
        7695281,
        10394312,
        12369372,
        14342891,
        6513507,
        9868950,
        12434877,
        14277081,
    ],
}
