"""
The :code:`%%graph` cell magic.

This module is imported by :mod:`kingdon` only when it is imported into a running IPython, so
that IPython remains optional for everyone who has no use for it.
"""
import ast
import re
import weakref
from collections.abc import Callable

from IPython import get_ipython
from IPython.core.displayhook import DisplayHook
from IPython.core.error import UsageError
from IPython.core.magic import needs_local_scope

from kingdon.graph import TREE_TYPES
from kingdon.multivector import MultiVector


# A ganja.js option flag: -key, --key or -key=, at the start of the line or after whitespace,
# with everything up to the next flag as its value. The name has to start with a letter or an
# underscore, so the minus sign of a value such as -scale -1 does not read as the next flag.
_FLAG = re.compile(r'(?:(?<=\s)|^)--?(?P<key>[A-Za-z_]\w*)(?:\s*=)?')


def _parse_options(line, ns):
    """
    Parse the argument line of the :code:`%%graph` magic into an algebra and ganja.js options.

    Options are written either as flags (:code:`-grid -pointRadius 4`) or as keyword arguments
    (:code:`grid=1, pointRadius=4`), optionally preceded by an expression for the algebra to
    graph in. A flag without a value is :code:`True`, and a value which is not valid python is
    kept as a string, so both :code:`-width '600px'` and :code:`-width 600px` work.
    """
    algebra, line = None, line.strip()
    if line and not line.startswith('-') and '=' not in line.partition(' ')[0]:
        head, _, line = line.partition(' ')
        algebra, line = eval(head, ns), line.strip()
    if not line.startswith('-'):
        return algebra, eval(f'dict({line})', ns) if line else {}
    options, flags = {}, list(_FLAG.finditer(line))
    for flag, next_flag in zip(flags, flags[1:] + [None]):
        value = line[flag.end():next_flag.start() if next_flag else None].strip()
        try:
            options[flag['key']] = eval(value, ns) if value else True
        except Exception:
            options[flag['key']] = value
    return algebra, options


def _parse_subjects(cell, ns):
    """
    Evaluate the body of the :code:`%%graph` magic into a list of subjects.

    Every top level expression is a subject, and so is whatever an assignment assigns, in the
    order in which they appear. A top level tuple is unpacked into several subjects, so a cell
    reads exactly like the argument list of :meth:`~kingdon.algebra.Algebra.graph`. A line which
    ends in a semicolon is evaluated but left out of the scene.
    """
    subjects, lines = [], cell.splitlines()
    for node in ast.parse(cell).body:
        tail = lines[node.end_lineno - 1][node.end_col_offset:]
        if isinstance(node, ast.Expr):
            expr = node.value
        else:
            exec(compile(ast.Module([node], []), '<graph>', 'exec'), ns)
            if not isinstance(node, ast.Assign):
                continue
            expr = ast.parse(ast.unparse(node.targets[0]), mode='eval').body
        if DisplayHook.semicolon_at_end_of_expression(tail):
            continue
        value = eval(compile(ast.Expression(expr), '<graph>', 'eval'), ns)
        subjects += value if isinstance(expr, ast.Tuple) else [value]
    return subjects


def _find_algebra(subjects):
    """ Find the algebra to graph in by looking for a multivector among the subjects. """
    for subject in subjects:
        if not isinstance(subject, (MultiVector, *TREE_TYPES)) and isinstance(subject, Callable):
            subject = subject()
        if isinstance(subject, MultiVector):
            return subject.algebra
        if isinstance(subject, TREE_TYPES) and (algebra := _find_algebra(subject)):
            return algebra


_widgets = weakref.WeakValueDictionary()


@needs_local_scope
def graph(line, cell, local_ns):
    """::

        %%graph [algebra] [options]
        subject
        subject, subject, ...

    Cell magic wrapper around :meth:`~kingdon.algebra.Algebra.graph`. Every top level expression
    in the cell is a subject to be graphed, and the argument line holds the ganja.js options::

        %%graph -grid -pointRadius 4
        0xD0FFE1, [A, B, C]
        0x224488, A, "A", B, "B", C, "C"

    is the same as::

        alg.graph(0xD0FFE1, [A, B, C], 0x224488, A, "A", B, "B", C, "C",
                  grid=True, pointRadius=4)

    Whatever an assignment assigns is a subject as well, and a line which ends in a semicolon is
    evaluated but left out of the scene, so a scene can be built up in the cell itself::

        %%graph -grid
        L = A & B   # graphed
        A & C;      # not graphed

    The algebra is that of the first multivector among the subjects, unless the line starts with
    an expression for the algebra to use, e.g. :code:`%%graph alg -grid`.

    Rerunning a cell redraws the widget it made before with
    :meth:`~kingdon.graph.GraphWidget.update`, instead of making a new one.
    """
    algebra, options = _parse_options(line, local_ns)
    subjects = _parse_subjects(cell, local_ns)
    if algebra is None and (algebra := _find_algebra(subjects)) is None:
        raise UsageError('%%graph found no algebra among the subjects; name the algebra to '
                         'graph in at the start of the argument line, e.g. "%%graph alg".')
    cell_id = (getattr(get_ipython(), 'parent_header', None) or {}).get('metadata', {}).get('cellId')
    if (widget := _widgets.get(cell_id)) is not None and widget.algebra is algebra:
        with widget.hold_sync():
            widget.options = options  # Assigned separately, so dropping a flag also takes effect.
            widget.update(*subjects)
        return widget
    widget = algebra.graph(*subjects, **options)
    if cell_id is not None:
        _widgets[cell_id] = widget
    return widget


def register_magics(ipython):
    """ Register the :code:`%%graph` cell magic. """
    ipython.register_magic_function(graph, 'cell', 'graph')
