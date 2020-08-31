import numpy as np
import random as rd
import matplotlib.pyplot as plt

X_train = []
Y_train = []

X_test = []
Y_test = []

s = 0
C = 0
a = []
b = 0

def lssvm_train(s, C):
    n_train = len(Y_train)
    
    Om = np.ones((n_train + 1,n_train + 1))

    Om[0,0] = 0

    for i in range(n_train):
        for j in range(i,n_train,1):
            temp = np.subtract(X_train[i],X_train[j])
            Om[i + 1,j + 1] = np.exp(-1 * np.dot(temp,temp) / s**2)
            Om[j + 1,i + 1] = Om[i + 1,j + 1]
            if i == j:
                Om[i + 1,j + 1] += C
                
    D = np.zeros(n_train+1)

    for i in range(n_train):
        D[i + 1] = Y_train[i]

    det = np.linalg.det(Om)

    if det != 0.0:
        z = np.linalg.solve(Om,D)

        b = z[0]

        a = np.delete(z, 0)

        ls = 0

        for j in range(n_train):
            predict = 0
            for i in range(n_train):
                temp = np.subtract(X_train[i],X_train[j])
                predict += a[i] * np.exp(-1 * np.dot(temp,temp) / s**2)
            predict += b
            
            ls += (predict - Y_train[j])**2
            
        ls /= n_train
        
        return ls
    else:
        return float('inf')

#=================================================================

def PSO():
    global s,C
    
    n_train = len(Y_train)

    s = 0
    C = 0

    w = 0.9
    c1 = 1.5
    c2 = c1

    n_iteration = 150
    n_particles = 30

    p_pos = np.array([np.array([rd.uniform(1, -1), rd.uniform(10,0.1)]) for _ in range(n_particles)])
    p_best_pos = p_pos.copy()
    p_best_val = np.array([float('inf') for _ in range(n_particles)])
    g_best_val = float('inf')
    g_best_pos = np.array([float('inf'), float('inf')])

    vel = np.array([np.array([0,0]) for _ in range(n_particles)])

    iteration = 0

    while iteration < n_iteration:
        for i in range(n_particles):
            fit_val = lssvm_train(p_pos[i][0],p_pos[i][1])

            if fit_val == float('inf') or p_pos[i][0] > 10 or p_pos[i][0] < 0.1:
                continue
                
            if p_best_val[i] > fit_val:
                p_best_val[i] = fit_val
                p_best_pos[i] = p_pos[i]

            if g_best_val > fit_val:
                g_best_val = fit_val
                g_best_pos = p_pos[i]

        if g_best_val == float('inf'):
            continue

        print("\n#" + str(iteration) + " Train Result\nerror =", g_best_val,"\ns =",g_best_pos[0],"\nC =",g_best_pos[1],"\n")

        for i in range(n_particles):
            new_vel = (w*vel[i]) + (c1*rd.random()*(p_best_pos[i] - p_pos[i])) + (c2*rd.random()*(g_best_pos - p_pos[i]))
            vel[i] = new_vel
            new_pos = new_vel + p_pos[i]
            p_pos[i] = new_pos

        w = 0.9 - (iteration/n_iteration)*0.5
        iteration += 1
        
    s = g_best_pos[0]
    C = g_best_pos[1]

#=================================================================

def set():
    global a,b,s,C
    
    n_train = len(Y_train)
    n_test = len(Y_test)

    Om = np.ones((n_train + 1,n_train + 1))

    Om[0,0] = 0

    for i in range(n_train):
        for j in range(i,n_train,1):
            temp = np.subtract(X_train[i],X_train[j])
            Om[i + 1,j + 1] = np.exp(-1 * np.dot(temp,temp) / s**2)
            Om[j + 1,i + 1] = Om[i + 1,j + 1]
            if i == j:
                Om[i + 1,j + 1] += C

    D = np.zeros(n_train+1)

    for i in range(n_train):
        D[i + 1] = Y_train[i]

    z = np.linalg.solve(Om,D)

    b = z[0]

    a = np.delete(z, 0)

#=================================================================

def test():
    global a,b,s,C
    
    n_train = len(Y_train)
    n_test = len(Y_test)
    
    predicts = []

    for j in range(n_test):
        predict = 0
        for i in range(n_train):
            temp = np.subtract(X_train[i],X_test[j])
            predict += a[i] * np.exp(-1 * np.dot(temp,temp) / s**2)
        predict += b
        
        predicts.append(predict)

    return predicts

