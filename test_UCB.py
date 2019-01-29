import numpy as np
import gain
import math
import matplotlib.pyplot as plt

def UCB(T, J, nb_machines) :
    
    s = [0] * nb_machines #nombre de fois où le bras k a été joué
    regret = [0]  
    moy = [0] * nb_machines
    B = [0] * nb_machines
    a = 0  # On suppose que le gain théorique de la machine ne sera jamais très supérieur 0
    
    # Détermination de b, plus petit k tel que P(X = k) < 10^-9
    
#==============================================================================
#     mu = 1.5
#     b = 0
#     p = math.exp(-mu) * mu**b / math.factorial(b)
#     while p > 1e-10 :
#         b += 1
#         p = math.exp(-mu) * mu**b / math.factorial(b)
#==============================================================================
    b = 15
    
    
    for i in range(nb_machines) :
        moy[i] = gain.testGain(i+1, J)[0]
        s[i] += 1
        regret.append(regret[-1] + (-gain.testGain(i+1, J)[1] + gain.testGain(3, J)[1])/T)
   # print(moy)
    for t in range(nb_machines, T+1) :
        for k in range(nb_machines) :
            B[k] = moy[k] + (b - a) * math.sqrt(3 * np.log(1/0.95) / (2 * s[k]))
        k = np.argmax(np.asarray(B))
        print(B)
        
        moy[k] = gain.testGain(k+1, J)[0]/(s[k] + 1) + (s[k]) / (s[k] + 1) * moy[k]
        s[k] += 1
        regret.append(regret[-1] + (-gain.testGain(k+1, J)[1] + gain.testGain(3, J)[1])/T)

    print(moy)
    return np.asarray(regret[1:-1])


plt.plot(UCB(1000, 0, 5), "green")