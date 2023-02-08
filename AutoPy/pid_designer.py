import numpy as np
import control
import random
import sympy as sp
from AutoPy import response
from AutoPy import utils
from AutoPy import system_analysis

def controller_from_gains(Kp,Ki,Kd):
    s = sp.Symbol("s")
    return control.tf(*utils.get_coeffs(Kp+Ki/s+Kd*s))

GENERATION_MULT = 10

def child(
        controller_1:tuple[float,float,float],
        controller_2:tuple[float,float,float],
        PID_type:str,
        mutation_rate:float=0.1):
    child_controller = \
        (random.choice([controller_1[0],controller_2[0]]) + random.choice([(GENERATION_MULT*random.random()*2-1)*mutation_rate,0])) if "P" in PID_type else 0,\
        (random.choice([controller_1[1],controller_2[1]]) + random.choice([(GENERATION_MULT*random.random()*2-1)*mutation_rate,0])) if "I" in PID_type else 0,\
        (random.choice([controller_1[2],controller_2[2]]) + random.choice([(GENERATION_MULT*random.random()*2-1)*mutation_rate,0])) if "D" in PID_type else 0
    return child_controller

def random_controller(PID_type:str):
    controller = \
        ((random.random()*2-1)*GENERATION_MULT) if "P" in PID_type else 0,\
        ((random.random()*2-1)*GENERATION_MULT) if "I" in PID_type else 0,\
        ((random.random()*2-1)*GENERATION_MULT) if "D" in PID_type else 0
    return controller

def loss(Gol:control.TransferFunction,controller:tuple[float,float,float]) -> float:
    """The lowest the better"""
    Gcl = control.feedback(control.series(controller_from_gains(*controller),Gol))
    loss_value = 0
    poles = Gcl.poles()
    for p in poles:
        if np.real(p) >= 0:
            loss_value += np.real(p)*10 + 5
    if loss_value == 0:
        T_MAX = 10
        time = np.linspace(0,T_MAX,100)
        responses = response.get_response(Gcl,time)
        time, _, _, error = responses[0]
        overshoot, settling_time = system_analysis.overshoot_and_settling_time(error,time)
        loss_value += overshoot + settling_time/T_MAX + (np.abs(responses[0][3][-1]) + np.abs(responses[1][3][-1])/T_MAX + np.abs(responses[2][3][-1])/(T_MAX**2))
    return loss_value

POPULATION = 100
EPOCHS = 10

def design(Gol:control.TransferFunction,PID_type:str="PI"):
    population = [random_controller(PID_type) for _ in range(POPULATION)]
    for e in range(EPOCHS):
        fitness = [1/(1+loss(Gol,controller)) for controller in population]
        new_population = []
        for _ in range(POPULATION):
            controllers = random.choices(population,fitness,k=2)
            new_population.append(child(*controllers,PID_type,0.1))
        print(f"\r\033[JEpoch {e+1}/{EPOCHS}",end="")
    population = [(controller,loss(Gol,controller)) for controller in population]
    population.sort(key=lambda x: x[1])
    print("\r\033[JDone!")
    return population
    
if __name__ == "__main__":
    G = sp.parse_expr(input("G(s)="))
    PID = input("PID type [PI]: ").upper()
    if PID == "":
        PID = "PI"
    Gol = control.tf(*utils.get_coeffs(G))
    controllers = design(Gol,PID)
    s = sp.Symbol("s")
    for (Kp,Ki,Kd),loss_value in controllers[:3]:
        K = Kp + Ki/s + Kd*s
        print(f"K(s)={K}, Kp={Kp}, Ki={Ki}, Kd={Kd}, Loss={loss_value}")


