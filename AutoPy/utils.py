import sympy as sp
import control
import numpy as np
import matplotlib.pyplot as plt

def get_coeffs(G):
    s = sp.symbols("s")
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