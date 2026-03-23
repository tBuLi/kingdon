import inspect
import itertools

from kingdon import Algebra
from sympy import Function


def test_basics():
    alg = Algebra(3)
    x = alg.vector(name='x')
    a = alg.vector(name='a')
    grad = alg.grad(x)

    def f_6_5(x):
        """ Eq 6.5 GA4Ph """
        return x | a

    alg.blades.e1.inv()
    grad_f_6_5 = (grad * f_6_5)
    assert grad_f_6_5.__name__ == 'f_6_5_grad'
    assert grad_f_6_5.__doc__ == 'Gradient of f_6_5.'
    assert inspect.signature(grad_f_6_5) == inspect.signature(f_6_5)    
    assert not (grad_f_6_5(x) - a)

    def f_6_6(x):
        """ Eq 6.6 GA4Ph """
        return x.e1
    
    assert (grad * f_6_6)(x) == alg.blades.e1.inv()

    def f_6_7(x):
        """ Eq 6.7 GA4Ph """
        return x**2
    
    assert not ((grad * f_6_7)(x) - 2 * x)

def test_symbolic():
    def partial_derivative(other, coordinate_mv, k):
        var = getattr(coordinate_mv, coordinate_mv.algebra.bin2canon[k])
        return other(coordinate_mv).map(lambda v: v.diff(var))
    
    alg = Algebra(3)
    x = alg.vector(name='x')
    J = alg.vector([Function('J1')(*x.values()), Function('J2')(*x.values()), Function('J3')(*x.values())])
    grad = alg.grad(x, partial_derivative)

    def f_6_19(x):
        """ Eq 6.19 GA4Ph: vec derivative = div + curl """
        return J

    grad_J = (grad * f_6_19)
    grad_J_x = grad_J(x)

    div_J = (grad | f_6_19)
    div_J_x = div_J(x)
    assert len(div_J_x.keys()) == 1
    assert not (div_J_x.e - sum(Ji.diff(xi) for Ji, xi in zip(J.values(), x.values())))

    curl_J = (grad ^ f_6_19)
    curl_J_x = curl_J(x)
    J1, J2, J3 = J.values()
    x1, x2, x3 = x.values()
    assert not (curl_J_x.e32 - (J2.diff(x3) - J3.diff(x2)))
    assert not (curl_J_x.e13 - (J3.diff(x1) - J1.diff(x3)))
    assert not (curl_J_x.e21 - (J1.diff(x2) - J2.diff(x1)))
    assert len(curl_J_x.keys()) == 3
    assert grad_J_x == div_J_x + curl_J_x

    # Evaluating the grad of a sympy expression results in an expression instead of a function.
    grad_J_x_direct = (grad * J)
    assert not (grad_J_x_direct - grad_J_x)
