import sympy as sp
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import matplotlib.cm as cm
import numpy as np
import random
import warnings

W_MIN = -5
W_MAX = +5

K_MIN = -4
K_MAX = +4

S_MIN = -10
S_MAX = +10

NYQUIST_BOUNDS = 1.5

PRECISION = 1000

BODE_AMPLITUDE_COLOR = "r"
BODE_PHASE_COLOR = "b"
BODE_GAIN_MARGIN_COLOR = "teal"
BODE_PHASE_MARGIN_COLOR = "brown"
NYQUIST_DIRECT_COLOR = "limegreen"
NYQUIST_INVERSE_COLOR = "green"
ROOT_LOCUS_COLOR = "m"
INVERSE_ROOT_LOCUS_COLOR = "c"
ROOT_LOCUS_FEATURES_COLOR = "orange"

def plot_analisys(G_string):
    s,w,k,bf = sp.symbols("s,__W__,__K__,__BODE_FUNCTION__")
    G_expr = sp.simplify(sp.parse_expr(G_string,{"e":math.e,"pi":math.pi,"j":sp.I}))

    w_space = np.logspace(W_MIN,W_MAX,PRECISION,base=10)
    k_space = np.logspace(K_MIN,K_MAX,PRECISION,base=10)
    bode_function = G_expr.subs({s:sp.I*w})
    bode_amplitude = 20*sp.log(sp.Abs(bf),10)
    bode_phase = sp.arg(bf)/sp.pi*180
    root_locus_function = sp.simplify(sp.numer(sp.together((1+k*G_expr))))
    zeros_x = []
    zeros_y = []
    poles_x = []
    poles_y = []
    numerator_degree = None
    denominator_degree = None
    breakaway_points = []
    centroid = None
    rl_asymptotes_angles = []
    try:
        zeros = sp.roots(sp.numer(G_expr))
        zeros_x = [sp.re(z) for z in zeros.keys()]
        zeros_y = [sp.im(z) for z in zeros.keys()]
        numerator_degree = sum(mult for _,mult in zeros.items())
    except sp.PolynomialError:
        print("Could not solve for zeroes")
    try:
        poles = sp.roots(sp.denom(G_expr))
        poles_x = [sp.re(p) for p in poles.keys()]
        poles_y = [sp.im(p) for p in poles.keys()]
        denominator_degree = sum(mult for _,mult in poles.items())
    except sp.PolynomialError:
        print("Could not solve for roots")

    if numerator_degree is not None and denominator_degree is not None:
        d = sp.Symbol("d")
        breakaway_points = sp.solve(
            sum(m/(d-p) for p,m in poles.items())
            -sum(m/(d-z) for z,m in zeros.items()),d)
        centroid = sp.re(sum(m*p for p,m in poles.items())-sum(m*z for z,m in zeros.items()))/(denominator_degree-numerator_degree)
        rl_asymptotes_angles = [(180+l*360)/(denominator_degree-numerator_degree) for l in range(denominator_degree-numerator_degree)]

    bode_amplitude_array = []
    bode_phase_array = []
    bode_function_array = []
    nyquist_curve_points_x = []
    nyquist_curve_points_ix = []
    nyquist_curve_points_y = []
    nyquist_curve_points_iy = []
    root_locus_x = []
    root_locus_y = []
    root_locus_ix = []
    root_locus_iy = []

    PASSES = [
        "TF Frequency Evaluation",
        "Bode Plot",
        "Nyquist Inverse Curve",
        "Nyquist Direct Curve",
        "Root Locus"
    ]
    # TF Frequency Evaluation
    current_pass = 0
    bode_function_LAMBDA = sp.lambdify(w,bode_function)
    for i,w_val in enumerate(w_space):
        bode_function_array.append(bode_function_LAMBDA(w_val))
        print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{PRECISION} {(i+1)/PRECISION*100:0.1f}%",end="")
    # Bode Amplitude and Phase
    current_pass+=1
    bode_amplitude_LAMBDA = sp.lambdify(bf,bode_amplitude)
    bode_phase_LAMBDA = sp.lambdify(bf,bode_phase)
    for i,bf_val in enumerate(bode_function_array):
        bode_amplitude_array.append(bode_amplitude_LAMBDA(bf_val))
        bode_phase_array.append(bode_phase_LAMBDA(bf_val))
        print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{PRECISION} {(i+1)/PRECISION*100:0.1f}%",end="")
    # Nyquist Inverse Curve
    current_pass+=1
    for i,bf_val in enumerate(bode_function_array):
        nyquist_curve_points_iy.append(-sp.im(bf_val))
        nyquist_curve_points_ix.append(sp.re(bf_val))
        print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{PRECISION} {(i+1)/PRECISION*100:0.1f}%",end="")
    # Nyquist Direct Curve
    current_pass+=1
    for i,bf_val in enumerate(bode_function_array):
        nyquist_curve_points_y.append(sp.im(bf_val))
        nyquist_curve_points_x.append(sp.re(bf_val))
        print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{PRECISION} {(i+1)/PRECISION*100:0.1f}%",end="")
    # Root Locus
    current_pass+=1
    try:
        roots = sp.solve(root_locus_function,s)
        roots = [sp.lambdify(k,root) for root in roots]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i,k_val in enumerate(k_space):
                for root in roots:
                    root_val = complex(root(k_val+0j))
                    root_locus_x.append(sp.re(root_val))
                    root_locus_y.append(sp.im(root_val))
                for root in roots:
                    root_val = complex(root(-k_val+0j))
                    root_locus_ix.append(sp.re(root_val))
                    root_locus_iy.append(sp.im(root_val))
                print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]}) {i+1}/{PRECISION} {(i+1)/PRECISION*100:0.1f}%",end="")
    except NotImplementedError:
        for i,k_val in enumerate(k_space):
            f = root_locus_function.subs({k:k_val})
            point = random.choice([-1,1])*sp.I+random.choice([-1,1])*random.random()
            point = 2*point.evalf()
            try:
                root = sp.nsolve(f,s,point,prec=10,tol=1e-8)
                root_locus_x.append(sp.re(root))
                root_locus_y.append(sp.im(root))
            except:
                pass
            f = root_locus_function.subs({k:-k_val})
            try:
                root = sp.nsolve(f,s,point,prec=10,tol=1e-8)
                root_locus_ix.append(sp.re(root))
                root_locus_iy.append(sp.im(root))
            except:
                pass
            print(f"\r\033[J{current_pass+1}/{len(PASSES)} ({PASSES[current_pass]} / NUMERICAL) {i+1}/{PRECISION} {(i+1)/PRECISION*100:0.1f}%",end="")
    print("\r\033[JDone!")

    w_phase_margin = None
    w_gain_margin = None
    y_phase_margin = None 
    y_gain_margin = None
    last_y_amplitude = None
    last_y_phase = None
    for x,y_amplitude,y_phase in zip(w_space,bode_amplitude_array,bode_phase_array):
        if w_phase_margin is None and last_y_amplitude is not None and last_y_amplitude*y_amplitude<=0:
            w_phase_margin = x
            y_phase_margin = y_phase
        if w_gain_margin is None and last_y_phase is not None and (sp.Abs(last_y_phase)>179 and sp.Abs(y_phase)>179 and last_y_phase*y_phase<0):
            w_gain_margin = x
            y_gain_margin = y_amplitude
        last_y_phase = y_phase
        last_y_amplitude = y_amplitude
    

    
    fig = plt.figure(figsize=(10,5))
    fig.text(0.5,0.95,"G(s) = "+str(G_expr),{"size":9},ha="center")
    grid_props = {"bottom":0.1,"left":0.05,"right":0.95,"top":0.9,"wspace":0,"hspace":0}
    FIRST_ROW_HEIGHT = 2
    grid = gs.GridSpec(FIRST_ROW_HEIGHT,3,fig,**grid_props)
    plots:list[plt.Axes] = []
    plots.append(fig.add_subplot(grid[0:FIRST_ROW_HEIGHT//2,0]))
    plots.append(fig.add_subplot(grid[FIRST_ROW_HEIGHT//2:FIRST_ROW_HEIGHT,0],sharex=plots[-1]))
    plots.append(fig.add_subplot(grid[0:FIRST_ROW_HEIGHT,1]))
    plots.append(fig.add_subplot(grid[0:FIRST_ROW_HEIGHT,2]))
    for plot in plots:
        plot.set_axisbelow(True)
        plot.grid(True,which="major",color="gray")
    for plot in plots[0:2]:
        plot.set_xscale("log")
        plot.axvline(1,color="black").set_zorder(0)
        plot.axhline(0,color="black").set_zorder(0)
    for plot in plots[2:4]:
        plot.set_xscale("linear")
        plot.axis('equal')
        plot.axvline(0,color="black").set_zorder(0)
        plot.axhline(0,color="black").set_zorder(0)
    plots[0].plot(w_space,bode_amplitude_array,color=BODE_AMPLITUDE_COLOR,label="Bode Amplitude")
    plots[1].plot(w_space,bode_phase_array,color=BODE_PHASE_COLOR,label="Bode Phase")
    if w_phase_margin is not None:
        plots[0].axvline(w_phase_margin,linestyle=':',color=BODE_PHASE_MARGIN_COLOR)
        plots[1].axvline(w_phase_margin,linestyle=':',color=BODE_PHASE_MARGIN_COLOR)
        plots[1].axhline(y_phase_margin,color=BODE_PHASE_MARGIN_COLOR,label=f"Phase Margin ~ {y_phase_margin+180:0.2f}dB")
    if w_gain_margin is not None:
        plots[0].axvline(w_gain_margin,linestyle=':',color=BODE_GAIN_MARGIN_COLOR)
        plots[1].axvline(w_gain_margin,linestyle=':',color=BODE_GAIN_MARGIN_COLOR)
        plots[0].axhline(y_gain_margin,color=BODE_GAIN_MARGIN_COLOR,label=f"Phase Margin ~ {-y_gain_margin:0.2f}dB")
    plots[2].plot(nyquist_curve_points_ix,nyquist_curve_points_iy,color=NYQUIST_INVERSE_COLOR,label="Nyquist Inverted Curve")
    plots[2].plot(nyquist_curve_points_x,nyquist_curve_points_y,color=NYQUIST_DIRECT_COLOR,label="Nyquist Direct Curve")
    plots[2].set_xbound(-NYQUIST_BOUNDS,NYQUIST_BOUNDS)
    plots[2].set_ybound(-NYQUIST_BOUNDS,NYQUIST_BOUNDS)
    plots[3].scatter(root_locus_x,root_locus_y,s=1,color=ROOT_LOCUS_COLOR,label="Root Locus") 
    plots[3].scatter(root_locus_ix,root_locus_iy,s=1,color=INVERSE_ROOT_LOCUS_COLOR,label="Inverse Root Locus") 
    plots[3].scatter(zeros_x,zeros_y,s=50,facecolors='none', edgecolors=ROOT_LOCUS_COLOR,marker="o",label="Zeros") 
    plots[3].scatter(poles_x,poles_y,color=ROOT_LOCUS_COLOR,marker="x",label="Poles") 
    plots[3].scatter(np.array(breakaway_points,dtype=complex).real,np.array(breakaway_points,dtype=complex).imag,color=ROOT_LOCUS_FEATURES_COLOR,marker="d",label="Breakaway Points") 
    plots[3].scatter([centroid],[0],color=ROOT_LOCUS_FEATURES_COLOR,marker="o",label="Centroid") 
    for i,angle in enumerate(rl_asymptotes_angles):
        asymptote_label = "Asymptote" if i==0 else None
        point2 = centroid + sp.exp(-1j*angle/180*sp.pi)
        xy = ((float(centroid.evalf()),0), (float(sp.re(point2).evalf()),float(sp.im(point2).evalf())))
        plots[3].axline(*xy,color=ROOT_LOCUS_FEATURES_COLOR,linestyle="--",label=asymptote_label)

    #fig.legend()
    for plot in plots:
        plot.legend()
    plt.show()

if __name__ == "__main__":
    G_string = input("G(s)=")
    plot_analisys(G_string)