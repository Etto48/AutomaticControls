import sympy as sp
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
import random

W_MIN = -5
W_MAX = +5

K_MIN = -2
K_MAX = +2
PRECISION = 1000

def plot_analisys(G_string):
    s,w,k,bf = sp.symbols("s,__W__,__K__,__BODE_FUNCTION__")
    G_expr = sp.simplify(sp.parse_expr(G_string,{"e":math.e,"pi":math.pi}))
    w_space = np.logspace(W_MIN,W_MAX,PRECISION,base=10)
    k_space = np.logspace(K_MIN,K_MAX,PRECISION)
    bode_function = G_expr.subs({s:sp.I*w})
    bode_aplitude = 20*sp.log(sp.Abs(bf),10)
    bode_phase = sp.arg(bf)/sp.pi*180
    root_locus_function = sp.numer(sp.together((1+k*G_expr)))
    zeros_x = []
    zeros_y = []
    poles_x = []
    poles_y = []
    try:
        zeros = sp.roots(sp.numer(G_expr))
        zeros_x = [sp.re(z) for z in zeros.keys()]
        zeros_y = [sp.im(z) for z in zeros.keys()]
        poles = sp.roots(sp.denom(G_expr))
        poles_x = [sp.re(p) for p in poles.keys()]
        poles_y = [sp.im(p) for p in poles.keys()]
    except sp.PolynomialError:
        pass

    bode_aplitude_array = []
    bode_phase_array = []
    bode_function_array = []
    nyquist_curve_points_x = []
    nyquist_curve_points_ix = []
    nyquist_curve_points_y = []
    nyquist_curve_points_iy = []
    root_locus_x = []
    root_locus_y = []

    PASSES = [
        "Bode Amplitude",
        "Bode Phase",
        "Nyquist Inverse Curve",
        "Nyquist Direct Curve",
        "Root Locus"
    ]
    current_pass = 0
    for i,w_val in enumerate(w_space):
        bode_function_array.append(bode_function.subs({w:w_val}).evalf())
        print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    current_pass+=1
    for i,bf_val in enumerate(bode_function_array):
        bode_aplitude_array.append(bode_aplitude.subs({bf:bf_val}).evalf())
        bode_phase_array.append(bode_phase.subs({bf:bf_val}).evalf())
        print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    current_pass+=1
    for i,bf_val in enumerate(bode_function_array):
        nyquist_curve_points_iy.append(-sp.im(bf_val))
        nyquist_curve_points_ix.append(sp.re(bf_val))
        print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    current_pass+=1
    for i,bf_val in enumerate(bode_function_array):
        nyquist_curve_points_y.append(sp.im(bf_val))
        nyquist_curve_points_x.append(sp.re(bf_val))
        print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    current_pass+=1
    try:
        for i,k_val in enumerate(k_space):
            f = root_locus_function.subs({k:k_val})
            roots = sp.roots(f,[s])
            for root, mult in roots.items():
                root_locus_x.append(sp.re(root))
                root_locus_y.append(sp.im(root))
            print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    except sp.PolynomialError:
        re_s, im_s = sp.symbols("__RE_S__,__IM_S__",real=True)
        for i,k_val in enumerate(k_space):
            f = root_locus_function.subs({k:k_val})
            f = f.subs({s:(re_s+sp.I*im_s)})
            fv = (sp.re(f),sp.im(f))
            point = 10 * random.random() * sp.exp(-2*sp.pi*sp.I*random.random())
            point = point.evalf()
            try:
                roots = sp.nsolve(fv,[re_s,im_s],[sp.re(point),sp.im(point)])
                for root in roots:
                    root_locus_x.append(sp.re(root))
                    root_locus_y.append(sp.im(root))
            except ValueError:
                pass
            print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]} / NUMERICAL) {i+1}/{len(w_space)} {(i+1)/len(w_space)*100:0.1f}%",end="")
    print("\r\033[JDone!")

    w_phase_margin = None
    w_gain_margin = None
    y_phase_margin = None 
    y_gain_margin = None
    last_y_amplitude = None
    last_y_phase = None
    for x,y_amplitude,y_phase in zip(w_space,bode_aplitude_array,bode_phase_array):
        if w_phase_margin is None and last_y_amplitude is not None and last_y_amplitude*y_amplitude<=0:
            w_phase_margin = x
            y_phase_margin = y_phase
        if w_gain_margin is None and last_y_phase is not None and (sp.Abs(last_y_phase)>179 and sp.Abs(y_phase)>179 and last_y_phase*y_phase<0):
            w_gain_margin = x
            y_gain_margin = y_amplitude

        last_y_phase = y_phase
        last_y_amplitude = y_amplitude
    


    fig = plt.figure()
    fig.text(0.5,0.95,"G(s) = "+str(G_expr),{"size":9},ha="center")
    grid = gs.GridSpec(2,3,left=0.1,right=0.9,top=0.9,bottom=0.1,wspace=0,hspace=0)
    plots:list[plt.Axes] = []
    plots.append(fig.add_subplot(grid[0,0]))
    plots.append(fig.add_subplot(grid[1,0],sharex=plots[-1]))
    plots.append(fig.add_subplot(grid[:,1]))
    plots.append(fig.add_subplot(grid[:,2]))
    for plot in plots:
        plot.grid(True)
        #plot.axhline(0,color="black")
    plots[0].plot(w_space,bode_aplitude_array,color="r",label="Bode Amplitude")
    plots[1].plot(w_space,bode_phase_array,color="b",label="Bode Phase")
    if w_phase_margin is not None:
        plots[0].axvline(w_phase_margin,linestyle=':',color='brown')
        plots[1].axvline(w_phase_margin,linestyle=':',color='brown')
        plots[1].axhline(y_phase_margin,color='brown',label=f"Phase Margin ~ {y_phase_margin+180:0.2f}dB")
    if w_gain_margin is not None:
        plots[0].axvline(w_gain_margin,linestyle=':',color='teal')
        plots[1].axvline(w_gain_margin,linestyle=':',color='teal')
        plots[0].axhline(y_gain_margin,color='teal',label=f"Phase Margin ~ {sp.Abs(y_gain_margin):0.2f}dB")
    plots[2].plot(nyquist_curve_points_ix,nyquist_curve_points_iy,color="green",label="Nyquist Inverted Curve")
    plots[2].plot(nyquist_curve_points_x,nyquist_curve_points_y,color="limegreen",label="Nyquist Direct Curve")
    plots[3].scatter(root_locus_x,root_locus_y,s=1,color="m",label="Root Locus")
    plots[3].scatter(zeros_x,zeros_y,color="m",marker="x",label="Zeros")
    plots[3].scatter(poles_x,poles_y,s=50,facecolors='none', edgecolors='m',marker="o",label="Poles")
    plots[0].set_xscale("log")
    #plots[0].axvline(1,color="black")
    plots[1].set_xscale("log")
    #plots[1].axvline(1,color="black")
    plots[2].set_xscale("linear")
    #plots[2].axvline(0,color="black")
    plots[3].set_xscale("linear")
    #plots[3].axvline(0,color="black")
    fig.legend()
    plt.show()

if __name__ == "__main__":
    G_string = input("G(s)=")
    plot_analisys(G_string)