# -*- coding: utf-8 -*-

import numpy as np
import gain
import math
import statsmodels.stats.proportion as ssp

gain_max = gain.resMed(0)[2]#Gain maximum pour le calcul de nos regrets


def Strategie_1(arg) :

    T1=arg[0] #Nombre de jeton en phase d'exploration
    T=arg[1] #Nombre de tests total
    N=arg[2] #Nombre de medicaments
    
    n = T1//N #Nombre d'essais de chaque medicaments en phase d'exploration
    essais = [] #Contient un vecteur d'essais par medicament pour les T1 premiers lancers
    regret = [0]
    choix = [0] * N #Compteur du nombre de fois ou on a teste chaque medicament
    
    #On fait n essais pour chaque medicament qu'on range dans essais_machine
    for k in range(N) :
        essai_machine = []
        for i in range(n) :
            essai_machine.append(gain.resMed(k+1)[0])
            choix[k] += 1
            regret.append(regret[-1] + (-gain.resMed(k+1)[1] + gain_max)/T)
        
        essais.append(essai_machine)
        res = np.asarray(essais)
    
    #On determine la medicament avec la meilleure moyenne
    moy = []
    for i in range(N) :
        moy.append(np.mean(res[i]))
    k = np.argmax(moy)+1
    
    #On joue le reste de nos essais avec le meilleur medicament
    essai_machine = []
    for i in range(T-T1-1) :
        essai_machine.append(gain.resMed(k)[0])
        choix[k-1] += 1
        regret.append(regret[-1] + (-gain.resMed(k)[1] + gain_max)/T)

    return np.array(regret), choix

def intDisjSup(intervalles, k) :  # On regarde si le k-ieme intervalle est superieur aux autres
    sup = intervalles[k][0]
    disj = True
    for i in range(len(intervalles)) :
        if sup < intervalles[i][1] :
            disj = False
    return disj

def intDisjInf(intervalles, k) :  # On regarde si le k-ieme intervalle est inferieur a un des autres
    inf = intervalles[k][1]
    disj = False
    for i in range(len(intervalles)) :
        if inf < intervalles[i][0] :
            disj = True
    return disj

def Strategie_2(arg) :

    T = arg[0] #Nombre de tests total
    N = arg[1] #Nombre de medicaments
    init = arg[2]
    alp = arg[3]
    
    intervalles = [[0,1]]
    essais = [[0]]
    for i in range(N -1) :
        intervalles.append([0,1])
        essais.append([0])


    regret = [0]
    int_a_tester = [True] * N
    choix = [0] * N #Compteur du nombre de fois ou on a teste chaque medicament
    trouve = False
    jetons_joues = 0

    e2 = 0
    

    while jetons_joues < init * N :
        for i in range(N) :
            jetons_joues += 1
            essais[i].append(gain.resMed(i+1)[0])
            regret.append(regret[-1] + (-gain.resMed(i+1)[1] + gain_max)/T)
            choix[i] += 1
    for i in range(N) :       
        intervalles[i] = list(ssp.proportion_confint(sum(essais[i]), len(essais[i]) -1, alpha=alp))
    
    while jetons_joues < T and not trouve :
        e2 += 1
        for i in range(len(intervalles)) :
            if int_a_tester[i] :
                if intDisjInf(intervalles, i) :
                    int_a_tester[i] = False
                elif intDisjSup(intervalles, i) :
                    trouve = True
                    meilleur = i
                else :
                    if jetons_joues == T :
                        break
                    jetons_joues += 1
                    choix[i] += 1
                    essais[i].append(gain.resMed(i+1)[0])
                    regret.append(regret[-1] + (-gain.resMed(i+1)[1] + gain_max)/T)
                    intervalles[i] = list(ssp.proportion_confint(sum(essais[i]), len(essais[i]) -1, alpha=alp))


        if int_a_tester.count(True) == 1 :       
            meilleur = int_a_tester.index(True)
            trouve = True
            break
        
    while jetons_joues < T :
        regret.append(regret[-1] + (-gain.resMed(meilleur+1)[1] + gain_max)/T)
        jetons_joues += 1
        choix[meilleur] += 1
        
    return np.asarray(regret[1:]), choix


def Eps_greedy(arg) :
    eps=arg[0] #Epsilon
    T=arg[1] #Nombre de tests total
    N=arg[2] #Nombre de médicaments
    
    
    s = [0] * N #Compteur du nombre de fois ou on a teste chaque medicament
    regret = [0]
    moy = [0] * N
    med = list(range(N))
    #Initialisation
    for i in range(N) :
        moy[i] = gain.resMed(i)[0]
        s[i] += 1
        regret.append(regret[-1] + (-gain.resMed(i)[1] + gain_max)/T)
         
    for i in range(N, arg[1]+1) :
        if np.random.random() < 1-eps :
            if np.where(moy == np.max(moy))[0].shape[0] > 1 : # Si plusieurs médicaments ont la meilleure moyenne empirique
                # On choisit le médicament le plus testé
                indices = np.take(med, np.where(moy == np.max(moy))[0])  # indices des meilleures moyennes
                essais = np.take(s, indices)  # nombre d'essais pour ces médicaments
                k = indices[np.argmax(essais)]  # indice du médicament le plus testé

            else :
                k = np.argmax(moy)  #on choisit le médicament avec la meilleure moyenne avec une probabilite 1-E
        else :
            k = np.random.randint(0,N)#ou on choisit le médicament aleatoirement (proba E)
        
        moy[k] = gain.resMed(k+1)[0]/(s[k] + 1) + (s[k]) / (s[k] + 1) * moy[k]
        s[k] += 1
        regret.append(regret[-1] + (-gain.resMed(k+1)[1] + gain_max)/T)
        
    return np.asarray(regret[1:-1]), s



# Arguments : E : constante à diviser, T : nombre de jetons total, J : cout du jeton, nb_machines
def Eps_greedy_temps(arg) :
    eps=arg[0] #Constante de temps
    T=arg[1] #Nombre de tests total
    N=arg[2] #Nombre de medicaments
       
    moy = [0] * N 
    s = [0] * N #Compteur du nombre de fois ou on a teste chaque medicament
    regret = [0]
    
    #Initialisation
    for i in range(N) :
        moy[i] = gain.resMed(i+1)[0]
        s[i] += 1
        regret.append(regret[-1] + (-gain.resMed(i+1)[1] + gain_max)/T)
         
    for i in range(N, T+1) :
        if np.random.random() < 1-eps/i**2 : #Meilleure courbe de regret lorsque E decroit en 1/t²
            k = np.argmax(moy) 
        else :
            k = np.random.randint(0,N)
        moy[k] = gain.resMed(k+1)[0]/(s[k] + 1) + (s[k]) / (s[k] + 1) * moy[k]
        s[k] += 1
        regret.append(regret[-1] + (-gain.resMed(k+1)[1] + gain_max)/T)
        
    return np.asarray(regret[1:-1]), s


def UCB(arg) :
    T=arg[0] #Nombre de tests total
    N=arg[1] #Nombre de medicaments
    
    s = [0] * N #Compteur du nombre de fois ou on a teste chaque medicament
    regret = [0]  
    moy = [0] * N
    B = [0] * N
    a = 0  
    b = 1

    #Initialisation
    for i in range(N) :
        moy[i] = gain.resMed(i+1)[0]
        s[i] += 1
        regret.append(regret[-1] + (-gain.resMed(i+1)[1] + gain_max)/T)
        
    for t in range(N, T+1) :
        for k in range(N) :
            B[k] = moy[k] + (b - a) * math.sqrt(2 * np.log(1/0.95) / (1 * s[k]))       
        k = np.argmax(np.asarray(B))
        moy[k] = gain.resMed(k+1)[0]/(s[k] + 1) + (s[k]) / (s[k] + 1) * moy[k]
        s[k] += 1
        regret.append(regret[-1] + (-gain.resMed(k+1)[1] + gain_max)/T)
        
    return np.asarray(regret[1:-1]), s


def Exp3(arg) :
    T=arg[0] #Nombre de tests total
    N=arg[1] #Nombre de medicaments
    
    p = [1/N] * N
    regret = [0]  
    G = [0] * N
    gain_estim = 0
    gamma = np.sqrt(N * np.log(N)/T)
    eta = np.sqrt(2*np.log(N)/(T * N))
    liste =  np.array(list(range(1,N+1)))
    choix = [0] * N #Compteur du nombre de fois ou on a teste chaque medicament
    
    for t in range(T) :
        k = np.random.choice(a=liste,p=p)
        choix[k-1] += 1
        gain_obtenu = gain.resMed(k)[1]
        regret.append(regret[-1] + (-gain_obtenu + gain_max)/T)
        gain_estim = gain_obtenu / p[k-1]
        G[k-1] += gain_estim
        somme = 0
        for i in range(len(p)) :
            somme += np.sum(np.exp(eta * G[i]))           
        for i in range(len(p)) :
            p[i] = (1-gamma) * np.exp(eta * G[i]) / somme + gamma / N

    return np.asarray(regret[1:]), choix

