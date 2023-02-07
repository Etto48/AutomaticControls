import sympy as sp
import tabulate
import control
import numpy as np
from AutoPy import utils
def rh_table(G):
    s,k = sp.symbols("s,k")
    Gcl = k*G/(1+k*G)
    caracteristic_poly = sp.poly(sp.denom(sp.together(Gcl)),s,domain="C[k]")
    coeffs = caracteristic_poly.all_coeffs()
    coeffs = [sp.simplify(c/coeffs[0]) for c in coeffs]
    rh_matrix:list[list] = []
    cols = (len(coeffs)+1)//2+1
    rows = cols+1
    for _ in range(rows):
        rh_matrix.append([0 for _ in range(cols)])
    for i,c in enumerate(coeffs):
        rh_matrix[i%2][i//2]=c
    for i in range(2,rows):
        for j in range(rows-2):
            first_item_last_row = rh_matrix[i-1][0]
            if first_item_last_row == 0:
                first_item_last_row = 1e-10
            rh_matrix[i][j] = sp.simplify(-(rh_matrix[i-2][0]*rh_matrix[i-1][j+1]-rh_matrix[i-2][j+1]*first_item_last_row)/first_item_last_row)
    if rh_matrix[-1] == [0 for _ in range(len(rh_matrix[-1]))]:
        rh_matrix = rh_matrix[:-1]
    return rh_matrix
    
def k_critical(table):
    k = sp.symbols("k")
    first_col = [l[0] for l in table]
    ret = []
    for e in first_col:
        new_val = sp.solve(e,k)
        for x in new_val:
            ret.append(x)
    return ret

def stability(table):
    k = sp.symbols("k")
    first_col = [l[0]*table[0][0]>0 for l in table]
    k_domain = sp.solve(first_col,k)
    return k_domain

def pid(kp,ki,kd):
    s = sp.symbols("s")
    K = (kp+ki/s+kd*s)
    return K
        

if __name__ == "__main__":
    G = sp.parse_expr(input("G(s)="))
    table = rh_table(G)
    print(tabulate.tabulate(table,maxcolwidths=30,tablefmt="grid"))
    print(f"Kcr={k_critical(table)}")
    stable_k = stability(table)
    if stable_k == False:
        print("Not stabilzable with P controller")
    else:
        print(f"Kp={stable_k}")
        
    
