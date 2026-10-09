import numpy as np
#import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

#_________________________________________________________
# Sección 1: En la primera etapa, se busca generar datos sintéticos para
# un HMM Gaussiano de 2 estados conocidos, e implemen-
# tar los algoritmos forward-backward y Viterbi a mano con
# Numpy. Aunque la implementación es opcional, se espera
# hacerlo para comprenderlos a profundidad. La última parte de
# esta primera etapa es verificar que se recupera el estado oculto
# simulado
#_________________________________________________________

"""
Generación de datos sintéticos
"""

A = np.array(
    [
        [0.95, 0.05],
        [0.10, 0.90]
        ]
        )
B = np.array(
    [   # mu   , sigma
        [0.005,  0.01],#Estado 0 (V. Baja)
        [-0.001, 0.03] #Estado 1 (V. Alta)
    ]
)
prob = np.array([0.5, 0.5])

#Genero Q (Estados ocultos, reprensenta si el mercado se encuentra con V.alta o V.baja)
T = 1000
Q = np.zeros(T)
Q[0] = np.random.choice([0,1],p=prob)
for i in range(T-1):
    Q[i+1] = np.random.choice([0,1], p= A[int(Q[i])])


#Genero O (observaciones)

O = np.zeros_like(Q)
for i in range(len(Q)):
    O[i] = np.random.normal(loc=B[int(Q[i])][0], scale= B[int(Q[i])][1])


#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#ax1.plot(Q)
#ax2.plot(O)
#plt.show()


"""
Algoritmo de Forward
"""


#Ahora genero alpha0 inicial para ambos estados 0 y 1
alpha0 = np.zeros(2)
c_t = np.zeros(T)
for i in range(2):
    alpha0[i] = prob[i] * sp.stats.norm.pdf(O[0], loc=B[i][0], scale= B[i][1])

# Con este inicial se forma todo el vector alpha_t(i)
alpha = np.zeros((T,2))
# Ahora normalizo el factor del tiempo inicial
c_t[0] = alpha0.sum()
alpha[0] = alpha0 / c_t[0] 

for t in range(1,T):
    for j in range(2):
        for i in range(2):
            alpha[t, j] += alpha[t-1,i] * A[i,j] *  sp.stats.norm.pdf(O[t], loc=B[j][0], scale= B[j][1])
            c_t[t] = alpha[t].sum()
    alpha[t] = alpha[t] / c_t[t]




# Como alpha_t(i) está normalizado la suma siempre va a ser 1
#  por eso para obtener la prob P(O,\lambda) = \sum log(c_t)

P = np.exp(-np.sum(np.log(c_t))) # Esto realmente es log P = sum_t log(c_t), no es la prob real, pero si se calcula P 
#                        explota numericamente ¿?


"""
Backward
"""

beta = np.zeros((T,2))
beta[T-1] = np.ones((1,2))/2.0 #se normaliza nuevamente

for t in range(T-2,-1,-1):
    for i in range(2):
        for j in range(2):
            beta[t,i] += A[i][j] * sp.stats.norm.pdf(O[t+1], loc=B[j][0], scale= B[j][1]) * beta[t+1,j]
            c_t[t] = beta[t].sum()
    beta[t] = beta[t]/c_t[t]

#print(beta)

"""
Calculo del suavizado gamma_t(i) = P(q_t=s_i|o_1,...,o_T)  (Esto es opcional, pero lo voy a hacer)
"""

gamma = (alpha * beta) / (alpha * beta).sum(axis=1, keepdims=True) 
#print(gamma)


"""
Viterbi
"""

#Primer paso, se define para el tiempo inicial

delta0 = np.zeros(2)
psi = np.zeros((T,2))
delta = np.zeros((T,2))
for i in range(2): 
    delta0[i] = prob[i] * sp.stats.norm.pdf(O[0], loc=B[i][0], scale= B[i][1])

delta[0] = np.log(delta0)
omega = np.zeros(2)

# Ahora, la recursión

for t in range(1,T):
    for j in range(2):
        for i in range(2):
             omega[i] = delta[t-1,i] + np.log(A[i][j])
        delta[t,j] = np.max(omega) + np.log(sp.stats.norm.pdf(O[t], loc=B[j][0], scale= B[j][1]))
        psi[t,j] = np.argmax(omega)


#Terminación: 
P_T = np.max(delta[T-1])
q_T = np.argmax(delta[T-1])

# Reconstrucción (backtracking): 
reconstruction = np.zeros(T)
reconstruction[T-1] = q_T
for t in range(T-2,-1,-1):
    reconstruction[t] = psi[t+1, int(reconstruction[t+1])] 

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(Q)
ax2.plot(reconstruction)
plt.show()