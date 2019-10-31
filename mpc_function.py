import random
import numpy as np


BASE = 10
PRECISION_INTEGRAL = 8
PRECISION_FRACTIONAL = 8
Q = 293973345475167247070445277780365744413

PRECISION = PRECISION_INTEGRAL + PRECISION_FRACTIONAL

assert(Q > BASE**PRECISION)


def encode(rational):
    upscaled = int(rational * 10**8)
    field_element = upscaled % Q
    return field_element


def decode(field_element):
    upscaled = field_element if field_element <= Q/2 else field_element - Q
    rational = upscaled / 10**8
    return rational


def share(x):
    x0 = random.randrange(Q)
    x1 = random.randrange(Q)
    x2 = (x - x0 - x1) % Q
    return [x0, x1, x2]


def reconstruct(shares):
    return sum(shares) % Q


def reshare(xs):
    Y = [ share(xs[0]), share(xs[1]), share(xs[2]) ]
    return [ sum(row) % Q for row in zip(*Y) ]


def add(x, y):
    return [ (xi + yi) % Q for xi, yi in zip(x, y) ]


def sub(x, y):
    return [ (xi - yi) % Q for xi, yi in zip(x, y) ]


def imul(x, k):
    return [(xi * k) % Q for xi in x]


INVERSE = 104491423396290281423421247963055991507 # inverse of BASE**FRACTIONAL_PRECISION
KAPPA = 6  # leave room for five digits overflow before leakage

assert((INVERSE * BASE**PRECISION_FRACTIONAL) % Q == 1)
assert(Q > BASE**(2*PRECISION + KAPPA))


def truncate(a):
    # map to the positive range
    b = add(a, [BASE**(2*PRECISION+1), 0, 0])
    # apply mask known only by P0, and reconstruct masked b to P1 or P2
    mask = random.randrange(Q) % BASE**(PRECISION + PRECISION_FRACTIONAL + KAPPA)
    mask_low = mask % BASE**PRECISION_FRACTIONAL
    b_masked = reconstruct(add(b, [mask, 0, 0]))
    # extract lower digits
    b_masked_low = b_masked % BASE**PRECISION_FRACTIONAL
    b_low = sub(share(b_masked_low), share(mask_low))
    # remove lower digits
    c = sub(a, b_low)
    # remove extra scaling factor
    d = imul(c, INVERSE)
    return d


def mul(x, y):
    # local computation
    z0 = (x[0]*y[0] + x[0]*y[1] + x[1]*y[0]) % Q
    z1 = (x[1]*y[1] + x[1]*y[2] + x[2]*y[1]) % Q
    z2 = (x[2]*y[2] + x[2]*y[0] + x[0]*y[2]) % Q
    # reshare and distribute
    zz = [share(z0), share(z1), share(z2)]
    ww = [sum(row) % Q for row in zip(*zz)]
    # bring precision back down from double to single
    vv = truncate(ww)
    return vv


class SecureRational(object):

    def __init__(self, secret=None):
        self.shares = share(encode(secret)) if secret is not None else []

    def reveal(self):
        return decode(reconstruct(reshare(self.shares)))

    def __repr__(self):
        return "SecureRational(%f)" % self.reveal()

    def __add__(x, y):
        z = SecureRational()
        z.shares = add(x.shares, y.shares)
        return z

    def __sub__(x, y):
        z = SecureRational()
        z.shares = sub(x.shares, y.shares)
        return z

    def __mul__(x, y):
        z = SecureRational()
        z.shares = mul(x.shares, y.shares)
        return z

    def __pow__(x, e):
        z = SecureRational(1)
        for _ in range(e):
            z = z * x
        return z


class OpenRational(object):

    def __init__(self, secret):
        self.secret = secret

    def secure(secret):
        return OpenRational(secret)

    def reveal(self):
        return self.secret

    def __repr__(self):
        return "OpenRational(%f)" % self.reveal()

    def __add__(x, y):
        return OpenRational(x.secret + y.secret)

    def __sub__(x, y):
        return OpenRational(x.secret - y.secret)

    def __mul__(x, y):
        return OpenRational(x.secret * y.secret)

    def __pow__(x, e):
        z = OpenRational(1)
        for _ in range(e):
            z = z * x
        return z


def ext_euclid(a, b):
    if b == 0:
        return 1, 0, a
    else:
        x, y, q = ext_euclid(b, a % b)  # q = gcd(a, b) = gcd(b, a%b)
        x, y = y, (x - (a // b) * y)
        return x, y, q


def mul_inverse(a, b):  # based on Extended Euclidean algorithm: ax = 1 (mod b) or ax + by = 1
    a_temp = a
    p_temp = 1
    t_temp = 0

    b_temp = b
    s_temp = 0
    q_temp = 1

    while True:
        if a_temp > b_temp:
            p_temp = p_temp - (a_temp // b_temp) * s_temp
            t_temp = t_temp - (a_temp // b_temp) * q_temp
            a_temp = a_temp % b_temp

            b_temp = b_temp
            s_temp = s_temp
            q_temp = q_temp
        else:
            a_temp = a_temp
            p_temp = p_temp
            t_temp = t_temp

            s_temp = s_temp - (b_temp // a_temp) * p_temp
            q_temp = q_temp - (b_temp // a_temp) * t_temp
            b_temp = b_temp % a_temp
        if a_temp == 1:
            inverse = p_temp
            break
        if b_temp == 1:
            inverse = s_temp
            break
    return inverse


# helper functions to map array of numbers to and from secure data type
secure = np.vectorize(lambda x: SecureRational(x))
reveal = np.vectorize(lambda x: x.reveal())
