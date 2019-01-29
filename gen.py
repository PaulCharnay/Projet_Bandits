# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

J = 2 # Cout du jeton

# Si modif des valeurs, garder la 3 en meilleure machine ou adapter l'algo du regret
def testGain(k) :
    if k == 1 :
        mu = 0.2
    elif k == 2 :
        mu = 1.2
    elif k == 3 :
        mu = 1.5
    elif k == 4 :
        mu = 0.7
    else :
        mu = 1

    return np.random.poisson(mu+J) - J, mu


T=1000 #Nombre total de jetons
nb_machines = 5

def calculIC(res,n) :
    
    gain = np.sum(res) 
    moy = np.mean(res)
    var = n/(n-1)*np.var(res) - J
 
    return moy, moy - 1.96 * var / np.sqrt(n), moy + 1.96 * var / np.sqrt(n), gain

def Afficher_IC(res,n) :
    out = ""
    a = calculIC(res,n)#on choisit le bras avec la meilleure moyenne avec une probabilité de 1-E
    out += "[" + str(a[1]) + ", " + str(a[2]) + "]" 
    return out

def Strategie_1(T1) :
    
    n = T1//nb_machines # nombre d'essais dans chaque machine en phase 1
    
    essais = []  # Contient un vecteur d'essais par machine pour les T1 premiers lancers
    intervalles = []
    regret = [0]
    
    #  On fait T1/nb_machines essais pour chauqe machine qu'on range dans essais_machine
    for k in range(nb_machines) :
        essai_machine = []
        for i in range(n) :
            essai_machine.append(testGain(k+1)[0])
            regret.append(regret[-1] + (-testGain(k+1)[1] + testGain(3)[1])/T)
            
        essais.append(essai_machine)
        res = np.asarray(essais)

    gain = 0
    moy = []
    for i in range(nb_machines) :
        a = calculIC(res[i],n)
        moy.append(a[0])
        gain += a[3]
        intervalles.append(Afficher_IC(res[i],n))
    k = moy.index(max(moy))+1
    essai_machine = []
    for i in range(T-T1-1) :
        essai_machine.append(testGain(k)[0])
        regret.append(regret[-1] + (-testGain(k)[1] + testGain(3)[1])/T)
        
    gain_tot = gain + sum(essai_machine)
      
 #   print("Nombre de jetons pour tester les machines : T1 = ", T1, "\22000nIntervalles de confiance : \n", intervalles, "\n",  "Machine choisie : ", k, "\nNombre de jetons dans la machine choisie : T2 = ", T-T1, "\nGain total obtenu : ", gain_tot, sep = '')
  #  plt.plot(regret)
    return np.asarray(regret)
    
def Eps_greedy(E) :
    gain_tot = [0] * nb_machines #gain total pour chaque machine
    s = [0] * nb_machines #nombre de fois où le bras k a été joué
    regret = [0]
    
    for i in range(nb_machines) :
        gain_tot[i] = testGain(i+1)[0]
        s[i] += 1
        regret.append(regret[-1] + (-testGain(i+1)[1] + testGain(3)[1])/T)
         
    for i in range(nb_machines, T+1) :
        if np.random.random() < 1-E :
            k = np.argmax(np.divide(gain_tot, s))#on choisit le bras avec la meilleure moyenne avec une probabilité 1-E
        else :
            k = np.random.randint(0,nb_machines)#ou on choisit le bras aléatoirement parmi les 5 machines (proba E)
        gain_tot[k] += testGain(k+1)[0]
        s[k] += 1
        regret.append(regret[-1] + (-testGain(k+1)[1] + testGain(3)[1])/T)
        
    return np.asarray(regret[1:-1])

def Eps_greedy_temps(E) :
    gain_tot = [0] * nb_machines #gain total pour chaque machine
    s = [0] * nb_machines #nombre de fois où le bras k a été joué
    regret = [0]
    
    for i in range(nb_machines) :
        gain_tot[i] = testGain(i+1)[0]
        s[i] += 1
        regret.append(regret[-1] + (-testGain(i+1)[1] + testGain(3)[1])/T)
         
    for i in range(nb_machines, T+1) :
        if np.random.random() < 1-E/i**2 : #Meilleure courbe de regret lorsque E décroit en 1/t²
            k = np.argmax(np.divide(gain_tot, s)) 
        else :
            k = np.random.randint(0,nb_machines)
        gain_tot[k] += testGain(k+1)[0]
        s[k] += 1
        regret.append(regret[-1] + (-testGain(k+1)[1] + testGain(3)[1])/T)
        
    return np.asarray(regret[1:-1])

def regret_moyen(N, methode, arg) : #Calcul regret moyen sur N itérations avec T1 premiers lancers
    regret_cumule = [0]*T

    for i in range(N) :
        regret_cumule = np.sum([regret_cumule, methode(arg)], axis=0)
    return np.divide(regret_cumule, N)

def plot_regret(N) :  # Bon truc, à garder ! Calcul regrets moyens sur N itérations en fonction du nombre T1 de premiers lancers
    
    y = []
    
    for t1 in range(2*nb_machines, T//3, 2*nb_machines) :
        y.append(regret_moyen(N, t1))
        
    plt.plot(range(2*nb_machines, T//3, 2*nb_machines), y)
           


# les regrets théoriques correspondent au cas parfait où on choisit la bonne machine après T1 lancers
def regret_moyen_th(T1) :
    regret_th = [0]
    
    for i in range(T1) :
        k = math.ceil(i*nb_machines/T1)
        regret_th.append(regret_th[-1] + (testGain(3)[1] - testGain(k)[1])/T)
    
    for i in range(T-T1) :
        regret_th.append(regret_th[-1])
    return regret_th


#Strategie_1(250)
#plot_regret(400)
plt.plot(regret_moyen(100, Strategie_1, 180), "blue")
plt.plot(regret_moyen(100, Eps_greedy, 0.05), "red")
plt.plot(regret_moyen(100, Eps_greedy_temps, 900), "green")
plt.plot(regret_moyen_th(180))


















