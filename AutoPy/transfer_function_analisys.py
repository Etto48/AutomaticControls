import sympy as sp
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np

W_MIN = -5
W_MAX = +5
PRECISION = 1000

def plot_analisys(G_string):
    s,w,k,bf = sp.symbols("s,w,__K__,__BODE_FUNCTION__")
    G_expr = sp.simplify(sp.parse_expr(G_string,{"e":math.e,"pi":math.pi}))
    w_space = np.logspace(W_MIN,W_MAX,PRECISION,base=10)
    k_space = np.logspace(W_MIN,W_MAX,PRECISION,base=10)
    bode_function = G_expr.subs({s:sp.I*w})
    bode_aplitude = 20*sp.log(sp.Abs(bf),10)
    bode_phase = sp.arg(bf)/sp.pi*180
    #root_locus_function = sp.simplify(1+k*bode_function)

    bode_aplitude_array = []
    bode_phase_array = []
    bode_function_array = []
    nyquist_curve_points_x = []
    nyquist_curve_points_y = []
    root_locus_x = []
    root_locus_y = []

    PASSES = [
        "Bode Amplitude",
        "Bode Phase",
        "Nyquist Inverse Curve",
        "Nyquist Direct Curve",
        #"Root Locus"
    ]
    current_pass = 0
    for i,w_val in enumerate(w_space):
        bode_function_array.append(bode_function.subs({w:w_val}).evalf())
        print(f"\r\033[J{current_pass+1}/{len(PASSES)}({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    current_pass+=1
    for i,bf_val in enumerate(bode_function_array):
        bode_aplitude_array.append(bode_aplitude.subs({bf:bf_val}).evalf())
        bode_phase_array.append(bode_phase.subs({bf:bf_val}).evalf())
        print(f"\r\033[J{current_pass+1}/{len(PASSES)}({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    current_pass+=1
    for i,bf_val in enumerate(reversed(bode_function_array)):
        nyquist_curve_points_y.append(-sp.im(bf_val))
        nyquist_curve_points_x.append(sp.re(bf_val))
        print(f"\r\033[J{current_pass+1}/{len(PASSES)}({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    current_pass+=1
    for i,bf_val in enumerate(bode_function_array):
        nyquist_curve_points_y.append(sp.im(bf_val))
        nyquist_curve_points_x.append(sp.re(bf_val))
        print(f"\r\033[J{current_pass+1}/{len(PASSES)}({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    current_pass+=1
    """for i,k_val in enumerate(k_space):
        f = root_locus_function.subs({k:k_val})
        root = sp.nsolve((sp.re(f),sp.im(f)),[w],(0.1))
        root_locus_x.append(sp.re(root))
        root_locus_y.append(sp.im(root))
        print(f"\r\033[J{current_pass+1}/{len(PASSES)}({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    """
    print("\r\033[JDone!")
    fig = plt.figure()
    grid = gs.GridSpec(2,2,left=0.1,right=0.9,top=0.9,bottom=0.1,wspace=0,hspace=0)
    plots:list[plt.Axes] = []
    plots.append(fig.add_subplot(grid[0,0]))
    plots.append(fig.add_subplot(grid[1,0]))
    plots.append(fig.add_subplot(grid[:,1]))
    for plot in plots:
        plot.grid(True)
        plot.axhline(0,color="black")
    plots[0].plot(w_space,bode_aplitude_array,color="r",label="Bode Amplitude")
    plots[1].plot(w_space,bode_phase_array,color="b",label="Bode Phase")
    plots[2].plot(nyquist_curve_points_x,nyquist_curve_points_y,color="g",label="Nyquist Curve")
    plots[0].set_xscale("log")
    plots[0].axvline(1,color="black")
    plots[1].set_xscale("log")
    plots[1].axvline(1,color="black")
    plots[2].set_xscale("linear")
    plots[2].axvline(0,color="black")
    fig.legend()
    plt.show()

if __name__ == "__main__":
    G_string = input("G(s)=")
    plot_analisys(G_string)