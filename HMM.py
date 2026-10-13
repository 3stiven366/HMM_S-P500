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

#P = np.exp(-np.sum(np.log(c_t))) # Esto realmente es log P = sum_t log(c_t), no es la prob real, pero si se calcula P 
#                        explota numericamente ¿?

print(f'valor incial: {np.sum(np.log(c_t))}')

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

#fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
#ax1.plot(Q)
#ax2.plot(reconstruction)
#plt.show()


"""
Baum-Welch (EM)
"""

chi = np.zeros((T-1,2,2)) #Es un tensor de tamaño Nx2x2

#Calculo de chi
for t in range(T-1):
    for i in range(2):
        for j in range(2):
            chi[t,i,j] = alpha[t,i] * A[i,j] * sp.stats.norm.pdf(O[t+1], loc=B[j][0], scale= B[j][1]) * beta[t+1,j]
    chi[t] = chi[t]/ chi[t].sum()


#Calculo de la probabilidad de estar en el estado s_i en el instante t (gamma_t(i))

gamma_prob = chi.sum(axis=2)

# Ahora se calcula la reestimación de pi, A, mu, simga²

Pi_re = gamma_prob[0]

A_re = chi.sum(axis=0) / gamma_prob.sum(axis=0).reshape(2,1)

mu_re = (gamma_prob * O[:T-1].reshape(T-1,1)).sum(axis=0) / gamma_prob.sum(axis=0)

simga_re = np.sqrt((gamma_prob *(O[:T-1].reshape(T-1,1)-mu_re)**2).sum(axis=0) / gamma_prob.sum(axis=0))





#while ME_CONDITION == False:


def Forward(pi,mu,sigma2,A):
    alpha0 = np.zeros(2)
    c_t = np.zeros(T)
    for i in range(2):
        alpha0[i] = pi[i] * sp.stats.norm.pdf(O[0], loc=mu[i], scale= sigma2[i])
    alpha = np.zeros((T,2))

    c_t[0] = alpha0.sum()
    alpha[0] = alpha0 / c_t[0] 

    for t in range(1,T):
        for j in range(2):
            for i in range(2):
                alpha[t, j] += alpha[t-1,i] * A[i,j] *  sp.stats.norm.pdf(O[t], loc=mu[j], scale= sigma2[j])
        c_t[t] = alpha[t].sum()
        alpha[t] = alpha[t] / c_t[t]

    P = np.sum(np.log(c_t))
    return alpha, P



def Backward(A, mu, sigma2):
    c_t = np.zeros(T)
    beta = np.zeros((T,2))
    beta[T-1] = np.ones((1,2))/2.0 #se normaliza nuevamente

    for t in range(T-2,-1,-1):
        for i in range(2):
            for j in range(2):
                beta[t,i] += A[i][j] * sp.stats.norm.pdf(O[t+1], loc=mu[j], scale= sigma2[j]) * beta[t+1,j]
        c_t[t] = beta[t].sum()
        beta[t] = beta[t]/c_t[t]

    return beta




def Restimation_Fuction( A, mu, sigma2, pi):

    beta = Backward(A, mu, sigma2)
    alpha , _ = Forward(pi,mu,sigma2,A)

    chi = np.zeros((T-1,2,2))
    for t in range(T-1):
        for i in range(2):
            for j in range(2):
                chi[t,i,j] = alpha[t,i] * A[i,j] * sp.stats.norm.pdf(O[t+1], loc=mu[j], scale= sigma2[j]) * beta[t+1,j]
        chi[t] = chi[t]/ chi[t].sum()

    gamma_prob = chi.sum(axis=2)

    # Ahora se calcula la reestimación de pi, A, mu, simga²

    pi_re = gamma_prob[0]

    A_re = chi.sum(axis=0) / gamma_prob.sum(axis=0).reshape(2,1)

    mu_re = (gamma_prob * O[:T-1].reshape(T-1,1)).sum(axis=0) / gamma_prob.sum(axis=0)

    sigma_re = np.sqrt((gamma_prob *(O[:T-1].reshape(T-1,1)-mu_re)**2).sum(axis=0) / gamma_prob.sum(axis=0))

    return pi_re, A_re, mu_re, sigma_re



def Baum_Welch(pi,mu,sigma,A): #Datos iniciales 
    ME_CONDITION = False 
    epsilon = 1e-4

    _, P_0 = Forward(pi,mu,sigma,A)

    while ME_CONDITION == False:
        pi_re, A_re, mu_re, sigma_re = Restimation_Fuction( A, mu, sigma, pi)
        _, P_i = Forward(pi_re,mu_re,sigma_re,A_re)
        if P_i - P_0 <= epsilon:
            ME_CONDITION = True
        else:
            P_0 = P_i
            A = A_re
            mu = mu_re
            sigma = sigma_re
            pi = pi_re
    return P_i, pi_re, A_re, mu_re, sigma_re






mu0 = np.array([0.0, 0.0])
sigma0 = np.array([0.02, 0.02])

#Prueba1,_,_,_,_ = Baum_Welch(prob,mu0,sigma0,A)
print(f'Baum Welch  (P_i, pi_re, A_re, mu_re, sigma_re =): {Baum_Welch(prob,mu0,sigma0,A)}')