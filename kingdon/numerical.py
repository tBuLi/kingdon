def exp(x, n=10):
    """
    Compute the exponential function using Taylor series expansion.

    Uses incremental term computation for improved numerical stability and performance.

    :param x: The input value (scalar or multivector).
    :param n: Number of terms in the Taylor series (default: 10).
    :return: Approximation of exp(x).
    """
    result = 1.0
    term = 1.0
    for k in range(1, n):
        term *= x / k
        result += term
    return result


def is_close(a, b, tol=1e-6):
    """
    Check if two multivectors are close within tolerance.
    
    Handles comparison of multivectors with potentially different sparse representations.
    
    :param a: First multivector or scalar value.
    :param b: Second multivector or scalar value.
    :param tol: Tolerance for comparison (default: 1e-6).
    :return: True if a and b are close within tolerance, False otherwise.
    """
    if hasattr(a, 'keys') and hasattr(b, 'keys'):
        all_keys = set(a.keys()) | set(b.keys())
        for key in all_keys:
            val_a = a.e if key == 0 else getattr(a, a.algebra.bin2canon[key], 0) if key in a.keys() else 0
            val_b = b.e if key == 0 else getattr(b, b.algebra.bin2canon[key], 0) if key in b.keys() else 0
            if abs(val_a - val_b) >= tol:
                return False
        return True
    else:
        return abs(a - b) < tol
