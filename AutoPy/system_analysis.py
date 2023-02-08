import control
import numpy as np
import sympy as sp
import sys
sys.path.append("./")
from AutoPy import utils
from AutoPy import response
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

W_MIN = -5
W_MAX = 5
T_MIN = 0
T_MAX = 100
PRECISION = 10000
SETTLING_MARGIN = 0.05
LEGEND_SETTINGS = {"fontsize":"x-small"}

def overshoot_and_settling_time(error,time):
    settling_index = 0
    for j,e in enumerate(reversed(error)):
        settling_index = j
        if np.abs(e) > SETTLING_MARGIN:
            break
    settling_time = time[-settling_index-1]
    max_error = -min(error)
    overshoot = 0 if max_error < 0 else max_error
    return overshoot, settling_time

def analyse(K,G):
    K_tf = control.tf(*utils.get_coeffs(K))
    G_tf = control.tf(*utils.get_coeffs(G))
    Gcl = control.feedback(control.series(K_tf,G_tf)) if K!=0 else G_tf
    gs = gridspec.GridSpec(2*3,2,hspace=0)
    fig = plt.figure(num="System Analysis")
    plots:list[plt.Axes] = []
    plots.append(fig.add_subplot(gs[0:3,0]))
    plots.append(fig.add_subplot(gs[3:6,0],sharex=plots[-1]))
    plots.append(fig.add_subplot(gs[0:2,1]))
    plots.append(fig.add_subplot(gs[2:4,1],sharex=plots[-1]))
    plots.append(fig.add_subplot(gs[4:6,1],sharex=plots[-1]))
    is_stable = utils.is_stable_sys(Gcl)
    ### Bode ###
    w_space = np.logspace(W_MIN,W_MAX,PRECISION,base=10)
    mag, phase, w_space = control.bode(Gcl,w_space,plot=False)
    gm, pm, wcg, wcp = control.margin(Gcl)
    plots[0].set_xscale("log")
    plots[0].plot(w_space,20*np.log10(mag),color="red")
    plots[0].axhline(-gm,color="steelblue",label=f"G.M.={gm:.4f} dB")
    plots[0].axvline(wcg,color="steelblue",ls=":",label=f"Wcg={wcg:.4f} Hz" if not np.isnan(wcg) else None)
    plots[0].axvline(wcp,color="darkred",ls=":",label=f"Wcp={wcp:.4f} Hz" if not np.isnan(wcp) else None)
    plots[0].grid()
    plots[0].legend(**LEGEND_SETTINGS)
    plots[0].set_ylabel("Magnitude in dB")
    plots[0].set_xlabel("Frequency in Hz")
    plots[1].set_xscale("log")
    plots[1].plot(w_space,phase*180/np.pi,color="blue")
    plots[1].axhline(pm-180,color="darkred",label=f"P.M.={pm:.4f} deg")
    plots[1].axvline(wcg,color="steelblue",ls=":",label=f"Wcg={wcg:.4f} Hz" if not np.isnan(wcg) else None)
    plots[1].axvline(wcp,color="darkred",ls=":",label=f"Wcp={wcp:.4f} Hz" if not np.isnan(wcp) else None)
    plots[1].grid()
    plots[1].legend(**LEGEND_SETTINGS)
    plots[1].set_ylabel("Phase in deg")
    plots[1].set_xlabel("Frequency in Hz")
    ### Response ###
    t_space = np.linspace(T_MIN,T_MAX,PRECISION)
    responses = response.get_response(Gcl,t_space)
    for i,(t,f_t,v_t,e_t) in enumerate(responses):
        overshoot = None
        settling_time = None
        if i == 0 and is_stable:
            # Overshoot and Settling time #
            overshoot, settling_time = overshoot_and_settling_time(e_t,t)
        plots[2+i].plot(t,f_t,color="gray",ls="--",label="Input")
        plots[2+i].plot(t,e_t,color="darkred",alpha=0.5,label=f"Error ({e_t[-1]:.4f})" if is_stable else "Error")
        plots[2+i].plot(t,v_t,label="Output" if i!=0 or not is_stable else f"Output\nP.O.={100*overshoot:.2f}%, Ts={settling_time:.2f}s")
        plots[2+i].grid()
        plots[2+i].legend(**LEGEND_SETTINGS)
        plots[2+i].set_xlabel("Time in s")
    fig.text(0.2,0.95,f"{str(Gcl)}",{"family":"monospace","size":6},va="center")
    if not is_stable:
        fig.text(0.1,0.95,"UNSTABLE",{"family":"monospace","size":7,"weight":"bold","color":"red"})
    else:
        fig.text(0.1,0.95,"STABLE",{"family":"monospace","size":7,"weight":"bold","color":"green"})
    plt.show()
    
if __name__ == "__main__":
    try:
        G = sp.parse_expr(input("G(s)="))
        K = sp.parse_expr(input("K(s)="))
        analyse(K,G)
    except KeyboardInterrupt:
        pass