from collections.abc import Callable
from functools import cached_property
from types import GeneratorType
import timeit
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
        values = o._values.tobytes() if isinstance(o._values, np.ndarray) else o._values.copy()
        if len(o) != len(o.algebra):
            # If not full mv, also pass the keys and let ganja figure it out.
            yield {'mv': values, 'keys': o._keys}
        else:
            yield {'mv': values}
    elif isinstance(o, Callable):
        yield encode(o(), tree_types)
    else:
        yield o


class GraphWidget(anywidget.AnyWidget):
    _esm = """
    const Algebra = await fetch("https://enki.ws/ganja.js/ganja.js")
                      .then(x=>x.text())
                      .then(x=>{ const ctx = {}; (new Function(x)).apply(ctx); return ctx.Algebra });

    function render({ model, el }) {
        var canvas = Algebra({metric: model.get('signature'), Cayley: model.get('cayley')}).inline((model)=>{
            // Define constants
            var key2idx = model.get('key2idx');
            var draggable_points_idxs = model.get('draggable_points_idxs');
            var options = model.get('options');
            if (options?.camera) {
                options.camera = toElement(options.camera)
            }
            // Define helper functions.
            var toElement = (o)=>{
                /* convert object to Element */
                var _values = o['mv'] instanceof DataView?new Float64Array(o['mv'].buffer):o['mv'];
                if ('keys' in o) {
                    var values = Array(Object.keys(key2idx).length).fill(0);
                    o['keys'].forEach((k, j)=>values[key2idx[k]] = _values[j]);
                    return new Element(values);
                }
                return new Element(_values);
            }

            var decode = x=>typeof x === 'object' && 'mv' in x?toElement(x):Array.isArray(x)?x.map(decode):x;
            var encode = x=>x instanceof Element?({mv:[...x]}):x.map?x.map(encode):x;

            if (options?.animate) {
                var graph_func = ()=>{
                    if (canvas?.value) {
                        model.set('draggable_points', encode(draggable_points_idxs.map(i=>canvas.value[i])));
                        model.save_changes();
                    }
                    // Send an update request. This drives the event loop.
                    model.send({ type: "update_mvs" });
                    var subjects = decode(model.get('subjects'));
                    return [...subjects];
                }
            } else {
                var graph_func = ()=>{
                    if (canvas?.value) {
                        model.set('draggable_points', encode(draggable_points_idxs.map(i=>canvas.value[i])));
                        model.save_changes();
                    }
                    var subjects = decode(model.get('subjects'));
                    return [...subjects];
                }

                // This ensures the remake is always called one last time to show the final position.
                model.on("change:subjects", ()=>{
                    if (canvas.remake) canvas = canvas.remake(0);
                    if (canvas.update) canvas.update(canvas.value);
                });
            }

            var canvas;
            canvas = this.graph(graph_func, options)
            return canvas;
        })(model)

        canvas.style.width = '100%';
        canvas.style.height = '400px';
        canvas.style.background = 'white';
        el.appendChild(canvas);
    }

    export default { render };
    """

    # Required arguments.
    algebra = traitlets.Instance("kingdon.algebra.Algebra")
    options = traitlets.Dict().tag(sync=True)

    # Properties derived from the required arguments which have to be available to js.
    signature = traitlets.List([]).tag(sync=True)  # Signature of the algebra
    cayley = traitlets.List([]).tag(sync=True)     # Cayley table of the algebra
    key2idx = traitlets.Dict({}).tag(sync=True)    # Conversion from binary keys to indices

    # Properties needed to paint the scene.
    raw_subjects = traitlets.List([])                          # A place to store the original input.
    pre_subjects = traitlets.List([])                          # Store the prepared subjects.
    draggable_points = traitlets.List([]).tag(sync=True)       # points at the first level of nesting are interactive.
    draggable_points_idxs = traitlets.List([]).tag(sync=True)  # indices of the draggable points in pre_subjects.
    subjects = traitlets.List([]).tag(sync=True)               # Result of evaluating pre_subjects.

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        return {k: i for i, k in enumerate(self.algebra.canon2bin.values())}

    @traitlets.default('signature')
    def get_signature(self):
        return list(self.algebra.signature)

    @traitlets.default('cayley')
    def get_cayley(self):
        cayley_table = [[s if (s := self.algebra.cayley[eJ, eI])[-1] != 'e' else f"{s[:-1]}1"
                         for eI in self.algebra.canon2bin]
                         for eJ in self.algebra.canon2bin]
        return cayley_table

    @traitlets.default('pre_subjects')
    def get_pre_subjects(self):
        return self._get_pre_subjects()

    @traitlets.default('subjects')
    def get_subjects(self):
        # Encode all the subjects
        return walker(encode(self._get_pre_subjects(), root=True))

    @traitlets.default('draggable_points')
    def get_draggable_points(self):
        # Extract the draggable points. TODO: pseudovectors only?
        return walker(encode([s for s in self.pre_subjects if isinstance(s, MultiVector)]))

    @traitlets.default('draggable_points_idxs')
    def get_draggable_points_idxs(self):
        # Extract the draggable points. TODO: pseudovectors only?
        return [j for j, s in enumerate(self.pre_subjects) if isinstance(s, MultiVector)]

    @traitlets.observe('draggable_points')
    def _observe_draggable_points(self, change):
        """ If draggable_points is changed, replace the raw_subjects in place. """
        self.inplacereplace(self.pre_subjects, zip(self.draggable_points_idxs, change['new']))
        self.subjects = self.get_subjects().copy()

    @traitlets.validate("options")
    def _valid_options(self, proposal):
        if 'camera' in proposal['value']:
            proposal['value']['camera'] = list(encode(proposal['value']['camera']))[0]
        return proposal['value']

    def inplacereplace(self, old_subjects, new_subjects: List[Tuple[int, dict]]):
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
