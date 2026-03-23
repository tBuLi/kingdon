import inspect

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

    # def f_6_19(x):
    #     """ Eq 6.19 GA4Ph: vec derivative = div + curl """
    #     return alg.vector([Function('J1')(x), Function('J2')(x), Function('J3')(x)])
    
    # ans = (grad * f_6_19)
    # assert not ((grad * f_6_19)(x) - alg.blades.e12)