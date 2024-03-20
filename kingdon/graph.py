from collections.abc import Callable
from functools import cached_property

import anywidget
import traitlets
import pathlib

from kingdon.multivector import MultiVector


TREE_TYPES = (list, tuple)


def encode(o, tree_types=TREE_TYPES):
    if isinstance(o, tree_types):
        return o.__class__(encode(value, tree_types) for value in o)
    elif isinstance(o, MultiVector):
        o = o.asfullmv()
        return {'mv': list(o.values())}
    elif isinstance(o, Callable):
        return encode(o(), tree_types)
    else:
        return o


class GraphWidget(anywidget.AnyWidget):
    _esm = pathlib.Path(__file__).parent / "graph.js"
    signature = traitlets.List([]).tag(sync=True)
    cayley = traitlets.List([]).tag(sync=True)
    options = traitlets.Dict({}).tag(sync=True)
    raw_subjects = traitlets.List([])

    subjects = traitlets.List([]).tag(sync=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_msg(self._handle_custom_msg)

    def _handle_custom_msg(self, data, buffers):
        if data["type"] == "update_mvs":
            # TODO: only update those that are callable for better performance?
            self.subjects = self._get_subjects()

    def _get_subjects(self):
        if len(self.raw_subjects) == 1 and len(self.callable_subjects) == 1:
            # This is very likely to be a function returning a list of suitable subjects.
            called_subjects = self.raw_subjects[0]()
            if isinstance(called_subjects, TREE_TYPES):
                return encode(called_subjects)
            return encode([called_subjects])

        # Otherwise just call every callable subject in-place in the encoder.
        return encode(self.raw_subjects)

    @traitlets.default('subjects')
    def get_subjects(self):
        return self._get_subjects()

    @cached_property
    def callable_subjects(self):
        """ Track which subjects are callable, by mapping their index in subjects to the function. """
        return {j: s for j, s in enumerate(self.raw_subjects)
                if not isinstance(s, MultiVector) and isinstance(s, Callable)}  # MV is callable so be careful
