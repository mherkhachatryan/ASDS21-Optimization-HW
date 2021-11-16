from math import exp


def bisection_derivative(f, a, b, N, tol=10 ** -4):
    if f(a) * f(b) >= 0:
        print("Bisection method fails.")
        return None
    a_n = a
    b_n = b
    for n in range(1, N + 1):
        m_n = (a_n + b_n) / 2
        f_m_n = f(m_n)
        if f(a_n) * f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n) * f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            print("Found exact solution.")
            return m_n
        elif f_m_n <= tol:
            break
        else:
            print("Bisection method fails.")
            return None
    return (a_n + b_n) / 2


if __name__ == "__main__":
    f = lambda x: exp(1 - x) + x ** 2
    derivative_f = lambda x: 2 * x - exp(1 - x)
    print(bisection_derivative(derivative_f, -1, 3, 1000))
