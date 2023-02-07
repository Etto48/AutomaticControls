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

def child(controller_1:tuple[float,float,float],controller_2:tuple[float,float,float],mutation_rate:float=0.1):
    child_controller = \
        (controller_1[0]+controller_2[0])/2 + (random.random()*2-1)*mutation_rate,\
        (controller_1[1]+controller_2[1])/2 + (random.random()*2-1)*mutation_rate,\
        (controller_1[2]+controller_2[2])/2 + (random.random()*2-1)*mutation_rate
    return child_controller

GENERATION_MULT = 1
def random_controller():
    controller = \
        (random.random()*2-1)*GENERATION_MULT,\
        (random.random()*2-1)*GENERATION_MULT,\
        (random.random()*2-1)*GENERATION_MULT
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

def design(Gol:control.TransferFunction):
    population = [random_controller() for _ in range(POPULATION)]
    for e in range(EPOCHS):
        fitness = [1/(1+loss(Gol,controller)) for controller in population]
        new_population = []
        for _ in range(POPULATION):
            controllers = random.choices(population,fitness,k=2)
            new_population.append(child(*controllers,0.1))
        print(f"Epoch {e+1}/{EPOCHS}")
    population = [(controller,loss(Gol,controller)) for controller in population]
    population.sort(key=lambda x: x[1])
    return population
    
if __name__ == "__main__":
    G = sp.parse_expr(input("G(s)="))
    Gol = control.tf(*utils.get_coeffs(G))
    controllers = design(Gol)
    s = sp.Symbol("s")
    for (Kp,Ki,Kd),loss_value in controllers[:3]:
        K = Kp + Ki/s + Kd*s
        print(f"K(s)={K}, Kp={Kp}, Ki={Ki}, Kd={Kd}, Loss={loss_value}")


