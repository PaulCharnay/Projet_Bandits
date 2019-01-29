# -*- coding: cp1252 -*-

import numpy as np

#N : Nombre d'iterations pour calculer la moyenne
#nb_machines : Nombre de machines(ou de  medicaments)
#T : Nombre de jetons (ou de tests) total
#methode : Strategie utilise 
#arg : Arguments de la strategie utilisee
def regret_moyen(N, nb_machines, T, methode, arg) : #Calcul regret moyen sur N iterations
    regret_somme = [0]*T
    choix_somme = [0]*nb_machines
    for i in range(N) :
        reg, ch = methode(arg)
        regret_somme = np.sum([regret_somme, reg], axis=0) 
        choix_somme = np.sum([choix_somme, ch], axis=0) 
        
    return np.divide(regret_somme, N), np.divide(choix_somme, N)
