def da(f, a, tau=1e-9):
    """
    The a-differential of function :code:`f()` in direction of :code:`a`.

    Computes the directional derivative numerically using a small increment.
    Follows chapter 2, equation 1.2.

    :param f: Function to differentiate.
    :param a: Direction multivector.
    :param tau: Small increment for numerical differentiation. Default is :code:`1e-9`.
    :return: Function that computes the directional derivative.
    """
    return lambda x: (f(x + tau * a) - f(x)) / tau


def d(alg, f, grade=1, tau=1e-9, prod=None):
    """
    The grade-derivative of :code:`f`, returns :code:`df()`, a function.

    Computes the derivative with respect to basis blades of a specified grade.
    Follows chapter 2, equation 1.5.

    :param alg: The :class:`~kingdon.algebra.Algebra` instance.
    :param f: Function to differentiate.
    :param grade: Grade of multivector basis to use for differentiation. Default is :code:`1`.
    :param tau: Small increment for numerical differentiation. Default is :code:`1e-9`.
    :param prod: Product function between differential and blades (:code:`gp`, :code:`op`, :code:`ip`).
        If :code:`None` (default), uses :code:`alg.gp`.
    :return: Function that computes the derivative.
    """
    if prod is None:
        prod = alg.gp

    # Get basis blades of specified grade
    blades = alg.blades_of_grade(grade)

    if not blades:
        return lambda x: alg.multivector()  # Return zero multivector

    return lambda x: \
        sum([prod(a.inv(), da(f, a, tau)(x)) for a in blades])


def curl(alg, f, grade=1, tau=1e-9):
    """
    Curl operation using the outer product.

    Computes the curl of a function using the exterior product.

    :param alg: The :class:`~kingdon.algebra.Algebra` instance.
    :param f: Function to compute the curl of.
    :param grade: Grade of multivector basis to use. Default is :code:`1`.
    :param tau: Small increment for numerical differentiation. Default is :code:`1e-9`.
    :return: Function that computes the curl.
    """
    return d(alg, f=f, grade=grade, tau=tau, prod=alg.op)


def div(alg, f, grade=1, tau=1e-9):
    """
    Divergence operation using the inner product.

    Computes the divergence of a function using the inner product.

    :param alg: The :class:`~kingdon.algebra.Algebra` instance.
    :param f: Function to compute the divergence of.
    :param grade: Grade of multivector basis to use. Default is :code:`1`.
    :param tau: Small increment for numerical differentiation. Default is :code:`1e-9`.
    :return: Function that computes the divergence.
    """
    return d(alg, f=f, grade=grade, tau=tau, prod=alg.ip)


def d_adj(alg, f, grade=1, tau=1e-9, prod=None):
    """
    Adjoint derivative (not fully implemented).

    This function is not yet fully implemented and will raise :code:`NotImplementedError`.

    :param alg: The :class:`~kingdon.algebra.Algebra` instance.
    :param f: Function to differentiate.
    :param grade: Grade of multivector basis to use. Default is :code:`1`.
    :param tau: Small increment for numerical differentiation. Default is :code:`1e-9`.
    :param prod: Product function between differential and blades.
    :return: Not yet implemented.
    :raises NotImplementedError: Always raised as this function is not yet fully implemented.
    """
    raise NotImplementedError("d_adj not fully implemented")


def product(alg, f, g, grade=1, tau=1e-9):
    """
    Product rule for derivatives: computes :code:`d(f()*g())`.

    Applies the product rule of differentiation to compute the derivative of a product.

    :param alg: The :class:`~kingdon.algebra.Algebra` instance.
    :param f: First function.
    :param g: Second function.
    :param grade: Grade of multivector basis for derivative computation. Default is :code:`1`.
    :param tau: Small increment for numerical differentiation. Default is :code:`1e-9`.
    :return: Function computing derivative of the product :code:`f()*g()`.
    """
    blades = alg.blades_of_grade(grade)

    if not blades:
        return lambda x: alg.multivector()

    return lambda x:\
        sum([a.inv() * (da(f, a, tau)(x) * g(x) + f(x) * da(g, a, tau)(x)) for a in blades])


def chain(alg, f, g, grade=1, tau=1e-9):
    """
    Chain rule: computes :code:`d(f(g()))`.

    Applies the chain rule of differentiation to compute the derivative of a composition.

    :param alg: The :class:`~kingdon.algebra.Algebra` instance.
    :param f: Outer function.
    :param g: Inner function for composition :code:`f(g(x))`.
    :param grade: Grade of multivector basis for derivative computation. Default is :code:`1`.
    :param tau: Small increment for numerical differentiation. Default is :code:`1e-9`.
    :return: Function computing derivative of the composition :code:`f(g(x))`.
    """
    df = d(alg, f, grade=grade, tau=tau)
    blades = alg.blades_of_grade(grade)

    if not blades:
        return lambda x: alg.multivector()

    return lambda x:\
        sum([a.inv() * da(f, da(g, a, tau)(x), tau)(g(x)) for a in blades])