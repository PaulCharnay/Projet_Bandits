import numpy as np
import scipy

# Arguments :  res : tab des T1 essais sur les machines ; n : nombre d'essais sur chaque machine(=T1/nb_machines); J : cout du jeton

def calculIC(res, n, J) : 
    
    gain = np.sum(res) 
    moy = np.mean(res)
    var = n/(n-1)*np.var(res)
 
    return moy - 1.96 * var / np.sqrt(n), moy + 1.96 * var / np.sqrt(n)

def calculIC2(res, J, nb_machines) : 
    
    n = len(res)
    moy = np.mean(res)

    Q1 = scipy.stats.chi2.interval(0.05, np.ceil(n/nb_machines) * (moy))[0]
    Q2 = scipy.stats.chi2.interval(0.05, (np.ceil(n/nb_machines)+1) * (moy))[1]
    
    return Q1/(2*n), Q2/(2*n)

def Afficher_IC(res, n, J) :
    out = ""
    a = calculIC2(res, J, 5)#on choisit le bras avec la meilleure moyenne avec une probabilité de 1-E
    out += "[" + str(a[0]) + ", " + str(a[1]) + "]" 
    return out