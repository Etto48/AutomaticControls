import sympy as sp
import control
import numpy as np
import matplotlib.pyplot as plt
import mpmath

PADE_ORDER = 5
def pade(exponential):
    s = sp.Symbol("s")
    assert type(exponential) == sp.exp
    #return tbcontrol.symbolic.pade(exponential,s,PADE_ORDER,PADE_ORDER)
    taylor_coeffs = [x for x in reversed(sp.poly(sp.series(exponential,s,0,2*PADE_ORDER+1).removeO(),s).all_coeffs())]
    approx = 1
    for x in sp.approximants(taylor_coeffs,s):
        approx = x
    return sp.simplify(approx)


def get_coeffs(G):
    s = sp.symbols("s")
    for a in G.args:
        if type(a) == sp.exp:
            G = G.subs(a,pade(a))
    N = sp.poly(sp.numer(sp.together(G)),s,domain="C").all_coeffs()
    N = [float(n) for n in N]
    D = sp.poly(sp.denom(sp.together(G)),s,domain="C").all_coeffs()
    D = [float(d) for d in D]
    return N,D

def is_stable(G):
    N,D = get_coeffs(G)
    Gol = control.TransferFunction(N,D)
    return is_stable_sys(Gol)

def is_stable_sys(Gcl:control.TransferFunction):
    for p in Gcl.poles():
        if np.real(p)>=0:
            return False
    return True