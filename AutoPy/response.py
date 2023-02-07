import control
import sympy as sp
from AutoPy import utils
import matplotlib.pyplot as plt
import numpy as np

def step(time):
    return [(0 if t<0 else 1) for t in time]

def ramp(time):
    return [(0 if t<0 else t) for t in time]

def parabola(time):
    return [(0 if t<0 else t**2) for t in time]

def diff(a,b):
    assert len(a) == len(b)
    n = len(a)
    return [a[i]-b[i] for i in range(n)]

FUNCTIONS = [step,ramp,parabola]
STEPS = 1000
T_MIN = 0
T_MAX = 100

def get_response(Gcl:control.TransferFunction):
    """Returns a list of tuples in the format (time,input,value,error)"""
    time = np.linspace(T_MIN,T_MAX,STEPS)
    ret = []
    for f in FUNCTIONS:
        t, v = control.forced_response(Gcl,time,f(time))
        f_t = f(t)
        e_t = diff(f_t,v)
        ret.append((t,f_t,v,e_t))
    return ret

def plot_response(G,K):
    Gol = control.TransferFunction(*utils.get_coeffs(G))
    Kol = control.TransferFunction(*utils.get_coeffs(K))
    Gcl = control.feedback(control.series(Gol,Kol)) if K!=0 else Gol
    gs_kw = {"hspace":0,"wspace":0}
    fig, plots = plt.subplots(len(FUNCTIONS),1,sharex="all",gridspec_kw=gs_kw)
    errors = []
    responses = get_response(Gcl)
    for i,(t,f_t,v_t,e_t) in enumerate(responses):
        plots[i].plot(t,f_t,color="gray",ls="--",label="Input" if i==0 else None)
        errors.append(e_t[-1])
        plots[i].plot(t,e_t,color="red",ls=":",label="Error" if i==0 else None)
        plots[i].plot(t,v_t,label="Output" if i==0 else None)
        plots[i].grid(True)
    fig.legend()
    print(errors)
    plt.show()

if __name__ == "__main__":
    G = sp.parse_expr(input("G(s)="))
    K = sp.parse_expr(input("K(s)="))
    plot_response(G,K)