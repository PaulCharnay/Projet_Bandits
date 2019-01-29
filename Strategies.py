# -*- coding: utf-8 -*-

import numpy as np
import gain
import math

gain_max = gain.testGain(0, 0)[2] #Gain maximum pour le calcul de nos regrets


def Strategie_1(arg) :
    
    T1 = arg[0] #Nombre de jeton en phase d'exploration
    T = arg[1] #Nombre de jetons total
    J = arg[2] #Cout du jeton
    nb_machines = arg[3] #Nombre de machines
    
    n = T1//nb_machines #Nombre d'essais dans chaque machine en phase d'exploration
    essais = [] #Contient un vecteur d'essais par machine pour les T1 premiers lancers
    regret = [0]
    
    #On fait n essais dans chaque machine qu'on range dans essais_machine
    for k in range(nb_machines) :
        essai_machine = []
        for i in range(n) :
            essai_machine.append(gain.testGain(k+1, J)[0])
            regret.append(regret[-1] + (-gain.testGain(k+1, J)[1] + gain_max)/T)
            
        essais.append(essai_machine)
        res = np.asarray(essais)
        
    #On determine la machine avec la meilleure moyenne
    moy = []
    for i in range(nb_machines) :
        moy.append(np.mean(res[i]))
    k = moy.index(max(moy))+1
    
    #On joue le reste de nos essais sur la meilleure machine
    essai_machine = []
    for i in range(T-T1-1) :
        essai_machine.append(gain.testGain(k, J)[0])
        regret.append(regret[-1] + (-gain.testGain(k, J)[1] + gain_max)/T)
    
    return np.asarray(regret), [0]


def Eps_greedy(arg) :
    
    E = arg[0] #Epsilon
    T = arg[1] #Nombre de jetons total
    J = arg[2] #Cout du jeton
    nb_machines = arg[3] #Nombre de machines
    
    s = [0] * nb_machines #Nombre de fois où le bras k a été joue
    regret = [0]
    moy = [0] * nb_machines
    machines = list(range(nb_machines))
    #Initialisation
    for i in range(nb_machines) :
        moy[i] = gain.testGain(i+1, arg[2])[0]
        s[i] += 1
        regret.append(regret[-1] + (-gain.testGain(i+1, J)[1] + gain_max)/T)
         
    for i in range(nb_machines, T+1) :
        if np.random.random() < 1-E :
            if np.where(moy == np.max(moy))[0].shape[0] > 1 : # Si deux bras ont la meilleure moyenne empirique
                # On choisit la machine la plus jouée
                indices = np.take(machines, np.where(moy == np.max(moy))[0])  # indices des meilleures moyennes
                essais = np.take(s, indices)  # nombre d'essais pour ces machines
                k = indices[np.argmax(essais)]  # indice de la machine la plus jouée

            else :
                k = np.argmax(moy)  # on choisit le bras avec la meilleure moyenne avec une probabilité 1-E
        else :
            k = np.random.randint(0,nb_machines) # Ou on choisit le bras aléatoirement parmi les N machines (proba E)
        
        moy[k] = gain.testGain(k+1, J)[0]/(s[k] + 1) + (s[k]) / (s[k] + 1) * moy[k]
        s[k] += 1
        regret.append(regret[-1] + (-gain.testGain(k+1, J)[1] + gain_max)/T)
        
    return np.asarray(regret[1:-1]), [0]


def Eps_greedy_temps(arg) :
    
    E = arg[0] #Constante de temps 
    T = arg[1] #Nombre de jetons total
    J = arg[2] #Cout du jeton
    nb_machines = arg[3] #Nombre de machines
    
    moy = [0] * nb_machines 
    s = [0] * nb_machines #Nombre de fois où le bras k a été joue
    regret = [0]
    
    #Initialisation
    for i in range(nb_machines) :
        moy[i] = gain.testGain(i+1, J)[0]
        s[i] += 1
        regret.append(regret[-1] + (-gain.testGain(i+1, J)[1] + gain_max)/T)
         
    for i in range(nb_machines, T+1) :
        if np.random.random() < 1-E/i**2 : #Meilleure courbe de regret lorsque E decroit en 1/t²
            k = np.argmax(moy) 
        else :
            k = np.random.randint(0,nb_machines)
        moy[k] = gain.testGain(k+1, J)[0]/(s[k] + 1) + (s[k]) / (s[k] + 1) * moy[k]
        s[k] += 1
        regret.append(regret[-1] + (-gain.testGain(k+1, J)[1] + gain_max)/T)
        
    return np.asarray(regret[1:-1]), [0]


def UCB(arg) :
    
    T = arg[0] #Nombre de jetons total
    J = arg[1] #Cout du jeton
    nb_machines = arg[2] #Nombre de machines
    
    s = [0] * nb_machines #Nombre de fois où le bras k a été joue
    regret = [0]  
    moy = [0] * nb_machines
    B = [0] * nb_machines
    a = 0  
    b = 15 
    
    #Initialisation
    for i in range(nb_machines) :
        moy[i] = gain.testGain(i+1, arg[1])[0]
        s[i] += 1
        regret.append(regret[-1] + (-gain.testGain(i+1, J)[1] + gain_max)/T)
        
    for t in range(nb_machines, T+1) :
        for k in range(nb_machines) :
            B[k] = moy[k] + (b - a) * math.sqrt(2 * np.log(1/0.95) / (1 * s[k]))
        k = np.argmax(np.asarray(B))
        moy[k] = gain.testGain(k+1, J)[0]/(s[k] + 1) + (s[k]) / (s[k] + 1) * moy[k]
        s[k] += 1
        regret.append(regret[-1] + (-gain.testGain(k+1, J)[1] + gain_max)/T)
      
    return np.asarray(regret[1:-1]), [0]


def Exp3(arg) :
    
    T = arg[0] #Nombre de jetons total
    J = arg[1] #Cout du jeton
    nb_machines = arg[2] #Nombre de machines
    
    p = [1/nb_machines] * nb_machines
    regret = [0]  
    G = [0] * nb_machines
    gain_estim = 0
    gamma = np.sqrt(nb_machines * np.log(nb_machines)/T)
    eta = np.sqrt(2*np.log(nb_machines)/(T * nb_machines))
    liste =  np.array(list(range(1,nb_machines+1)))
    
    for t in range(T) :
        k = np.random.choice(a=liste,p=p)
        gain_obtenu, gain_obtenu_th = gain.testGain(k, J)[0:2]
        regret.append(regret[-1] + (-gain_obtenu_th + gain_max)/T)
        gain_estim = gain_obtenu / p[k-1]
        G[k-1] += gain_estim
        somme = 0
        for i in range(len(p)) :
            somme += np.sum(np.exp(eta * G[i]))           
        for i in range(len(p)) :
            p[i] = (1-gamma) * np.exp(eta * G[i]) / somme + gamma / nb_machines

    return np.asarray(regret[1:]), [0]
    
    
    
    
    
    

































