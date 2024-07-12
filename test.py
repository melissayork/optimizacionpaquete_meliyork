from optimizacionpaquete_meliyork_1.funcionesPrueba.funcionesMultivariables import himmelblau
from optimizacionpaquete_meliyork_1.metodosMultivariables.hookeJeeves import hooke_jeeves

xb_inicial = [-5, -2.5]
delta = [0.5, 0.25]
alpha = 2
epsilon = 0.1
resul = hooke_jeeves(himmelblau, xb_inicial, delta, alpha, epsilon)
print(f"Resultado de booth: x = {resul}, f(x) = {himmelblau(resul)}")