# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import Strategies as S
import regret
import Strategies_labo as SL
import numpy as np


# =============================================================================
# VARIABLES GENERALES
# =============================================================================
T=1000 #Nombre total de jetons
nb_machines = 5
J = 2 # Cout du jeton
N = 10 # Nombre d'itérations pour calculer le regret moyen


# =============================================================================
# VARIABLES SPECIFIQUES AUX METHODES
# =============================================================================
#Stratégie 1
T1 = 150 #Nombre d'essais en phase d'exploration

#Stratégie 2
# plus on prend init petit, plus on doit prendre alpha petit
init = 8
alp = 0.3

#Epsilon-greedy
epsilon = 0.05

#Epsilon-greedy/t²
C = 900  #Constante de temps 


# =============================================================================
# #STRATEGIES MACHINES A SOUS
# =============================================================================

def demo_Strategie() :
    reg1, ch1 = regret.regret_moyen(N, nb_machines, T, S.Strategie_1, [T1, T, J, nb_machines])
    regE, chE = regret.regret_moyen(N, nb_machines, T, S.Eps_greedy, [epsilon, T, J, nb_machines])
    regEt, chEt = regret.regret_moyen(N, nb_machines, T, S.Eps_greedy_temps, [C, T, J, nb_machines])
    regUCB, chUCB = regret.regret_moyen(N, nb_machines, T, S.UCB, [T, J, nb_machines])
    regExp3, chExp3 = regret.regret_moyen(N, nb_machines, T, S.Exp3, [T, J, nb_machines])
    
    plt.plot(reg1, "yellow", label="Stratégie 1")
    plt.plot(regE, "red", label="epsilon-greedy")
    plt.plot(regEt, "green", label="epsilon-greedy /t²")
    plt.plot(regUCB, "brown", label = "UCB")
    plt.plot(regExp3, "pink", label = "Exp3")
    
    plt.xlabel("Nombre de jetons joués")
    plt.ylabel("Regret cumulé moyen")
    
    plt.axis([0, 1000, 0, 0.23])
    
    plt.legend()
    plt.show()


# =============================================================================
# #STRATEGIES ESSAIS CLINIQUES
# =============================================================================
    
def demo_Strategie_Labo() :
    reg1, ch1 = regret.regret_moyen(N, nb_machines, T, SL.Strategie_1, [T1, T, nb_machines])
    reg2, ch2 = regret.regret_moyen(N, nb_machines, T, SL.Strategie_2, [T, nb_machines, init, alp])
    regE, chE = regret.regret_moyen(N, nb_machines, T, SL.Eps_greedy, [epsilon, T, nb_machines])
    regEt, chEt = regret.regret_moyen(N, nb_machines, T, SL.Eps_greedy_temps, [C, T, nb_machines])
    regUCB, chUCB = regret.regret_moyen(N, nb_machines, T, SL.UCB, [T, nb_machines])
    regExp3, chExp3 = regret.regret_moyen(N, nb_machines, T, SL.Exp3, [T, nb_machines])
    
    plt.title("Regrets moyens des différentes stratégies")
    plt.plot(reg1, "yellow", label="Stratégie 1")
    plt.plot(reg2, "blue", label="Stratégie 2")
    plt.plot(regE, "red", label="epsilon-greedy")
    plt.plot(regEt, "green", label="epsilon-greedy /t²")
    plt.plot(regUCB, "brown", label = "UCB")
    plt.plot(regExp3, "pink", label = "Exp3")
    plt.axis([0,1000, 0, 0.09])
    
    plt.xlabel("Nombre d'essais effectués sur les patients")
    plt.ylabel("Regret cumulé moyen")
    
    plt.legend()
    plt.show()
    
    x = list(range(1,nb_machines+1))
    plt.bar(x, ch1/T)
    plt.title("Choix pour Stratégie 1")
    plt.show()
    plt.bar(x, ch2/T)
    plt.title("Choix pour Stratégie 2")
    plt.show()
    plt.bar(x, chE/T)
    plt.title("Choix pour Epsilon-greedy")
    plt.show()
    plt.bar(x, chEt/T)
    plt.title("Choix pour Epsilon-greedy /t²")
    plt.show()
    plt.bar(x, chUCB/T)
    plt.title("Choix pour UCB")
    plt.show()
    plt.bar(x, chExp3/T)
    plt.title("Choix pour Exp3")
    plt.show()
    
    
    plt.title("Comparaison des choix")
    ind = np.arange(nb_machines) + 1
    w = 0.13
    plt.bar(ind-2.5*w, ch1/T,width=w, color="yellow", label="Stratégie 1")
    plt.bar(ind-1.5*w, ch2/T,width=w, color="blue", label="Stratégie 2")
    plt.bar(ind-0.5*w, chE/T,width=w, color="red", label="epsilon-greedy") 
    plt.bar(ind+0.5*w, chEt/T,width=w, color="green", label="epsilon-greedy /t²")
    plt.bar(ind+1.5*w, chUCB/T,width=w, color="brown", label = "UCB")
    plt.bar(ind+2.5*w, chExp3/T,width=w, color="pink", label = "Exp3")
    plt.legend()

    plt.show

# =============================================================================
# demo_Strategie()
# =============================================================================
demo_Strategie_Labo()
