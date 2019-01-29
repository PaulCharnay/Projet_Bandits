# -*- coding: utf-8 -*-

import numpy as np
import gain

# =============================================================================
#KL-UCB, VARIABLES DE DEMO
pt_discr = 10 # Nb de points pour discrétiser pour KL_UCB
c = 3  # reprise de code KL_UCB
# =============================================================================


gain_max = gain.resMed(0)[2]
#==============================================================================
#==============================================================================
# # Go KL-UCB...
#==============================================================================
#==============================================================================



def reseqp(p, V, klMax, max_iterations=50):
    """ Solve ``f(reseqp(p, V, klMax)) = klMax``, using Newton method.

    .. note:: This is a subroutine of :func:`maxEV`.

    - Reference: Eq. (4) in Section 3.2 of [Filippi, Cappé & Garivier - Allerton, 2011](https://arxiv.org/pdf/1004.5229.pdf).

    .. warning:: `np.dot` is very slow!
    """
    MV = np.max(V)
    mV = np.min(V)
    value = MV + 0.1
    tol = 1e-4
    if MV < mV + tol:
        return float('inf')
    u = np.dot(p, (1 / (value - V)))
    y = np.dot(p, np.log(value - V)) + np.log(u) - klMax
    #print("value =", value, ", y = ", y)  # DEBUG
    _count_iteration = 0
    while _count_iteration < max_iterations and np.abs(y) > tol:
        _count_iteration += 1
        yp = u - np.dot(p, (1 / (value - V)**2)) / u  # derivative
        value -= y / yp
        #print("value = ", value)  # DEBUG  # newton iteration
        if value < MV:
            value = (value + y / yp + MV) / 2  # unlikely, but not impossible
        u = np.dot(p, (1 / (value - V)))
        y = np.dot(p, np.log(value - V)) + np.log(u) - klMax
        #print("value = ", value, ", y = ", y)  # DEBUG  # function
    return value


#==============================================================================
# def reseqp(p, V, klMax):
#     """ Solve f(reseqp(p, V, klMax)) = klMax, using a blackbox minimizer, from scipy.optimize.
# 
#     - FIXME it does not work well yet!
# 
#     .. note:: This is a subroutine of :func:`maxEV`.
# 
#     - Reference: Eq. (4) in Section 3.2 of [Filippi, Cappé & Garivier - Allerton, 2011].
# 
#     .. warning:: `np.dot` is very slow!
#     """
#     MV = np.max(V)
#     mV = np.min(V)
#     tol = 1e-4
#     value0 = mV + 0.1
# 
#     def f(value):
#         """ Function fo to minimize."""
#         if MV < mV + tol:
#             y = float('inf')
#         else:
#             u = np.dot(p, (1 / (value - V)))
#             y = np.dot(p, np.log(value - V)) + np.log(u)
#         return np.abs(y - klMax)
# 
#     res = scipy.optimize.minimize(f, value0)
#     #print("scipy.optimize.minimize returned", res)
#     return res.x if hasattr(res, 'x') else res
#==============================================================================

def maxEV(p, V, klMax):
    """Maximize expectation of V wrt. q st. KL(p,q) < klMax.

    Input args.: p, V, klMax.

    Reference: Section 3.2 of [Filippi, Cappé & Garivier - Allerton, 2011].
    """
    Uq = np.zeros(len(p))
    Kb = p>0
    K = p <= 0
    if any(K):
        # Do we need to put some mass on a point where p is zero?
        # If yes, this has to be on one which maximizes V.
        eta = max(V[K])
        J = K & (V == eta)
        if (eta > max(V[Kb])):
            y = np.dot(p[Kb], np.log(eta-V[Kb])) + np.log(np.dot(p[Kb],  (1./(eta-V[Kb]))))
            #print "eta = " + str(eta) + ", y="+str(y);
            if y < klMax:
                rb = np.exp(y-klMax)
                Uqtemp = p[Kb]/(eta - V[Kb])
                Uq[Kb] = rb*Uqtemp/sum(Uqtemp)
                Uq[J] = (1.-rb) / sum(J) # or j=min([j for j in range(k) if J[j]]); Uq[j] = r
                return(Uq)
    # Here, only points where p is strictly positive (in Kb) will receive non-zero mass.
    if any(abs(V[Kb] - V[Kb][0])>1e-8):
        eta = reseqp(p[Kb], V[Kb], klMax) # (eta = nu in the article)
        Uq = p/(eta-V)
        Uq = Uq/sum(Uq)
    else:
        Uq[Kb] = 1/len(Kb); # Case where all values in V(Kb) are almost identical.
    return(Uq)


# nb_bras: nombre de bras
# nb_discr: nombre de points utilisés pour discrétiser l'intervale dans le quel on estime la densité empirique
#arg = [nb_bras,nb_discr, T_end=1000,c=3]
def KL_UCB(arg):

    f = (lambda t: np.log(t)+3.*np.log(np.log(t)))
    s=np.ones(arg[0]) # Nombre de fois ou chaque bras a été joué
    p_space=[0,1]# On suppose que la variable aléatoire peut prendre des valeurs entre 0 et 1, changer les bornes ici si necessaire
    
    mu_estim = np.zeros((arg[0],arg[1])) #ligne=bras, colone=discretisation, densité empirique
    R = np.zeros(arg[2])
    regret = [0]
      # Dictionnaire contenant pour chaque bras, le nombre de fois ou chaque valeur possible à été observé, initialisé à 0 fois vue la valeur 1.
    obs = dict() 
    for arm in range(arg[0]):
      obs[arm]=dict({1.:0})
    idx=np.zeros(arg[0])
   # print(list(obs[1].items()))
      ### Initialisation pour chaque bras
    for t in np.arange(0,arg[0]):
      # p_t = gain du bras t
      R[t] = gain.resMed(t+1)[0]
      regret.append(regret[-1] + (-gain.resMed(t+1)[1] + gain_max)/arg[2])
        
  
      loc = np.argmin(np.abs(p_space-R[t])) # On cherche le point de discretisation le plus proche de la valeur obtenue.
      mu_estim[t,loc]+=1.

        # Ajouté la valeur obsérvé au dictionnaire
      reward=p_space[loc]
      if reward in obs[t] :
          obs[t][reward] += 1
      else:
          obs[t][reward] = 1

      ### Boucle principale
    t=arg[0]
    while t<arg[2]:  

        ## Résolution du problème d'optimisation
        # Pour chaque bras on: 
        for arm in range(arg[0]):
          if s[arm]!=0:
            p = (np.array(list(obs[arm].values())))/float(sum(obs[arm].values())) # normalise la densité estimée
            
            v = np.array(list(obs[arm].keys())) 
            q = maxEV(p, v, f(t/s[arm])/s[arm]) # Résoud le problème d'optimisation discrétisé
            idx[arm]=(np.dot(q,v)) # calcul la valeur associé à ce bras
            #print(arm, obs)
          else:
            idx[arm]=10**10 # Si on a jamais joué se bras, on va vouloir le jouer prochainement donc on met sa valeur au maximum
        
        #print(idx)  # Marche pas, donne les mêmes valeurs pour tous les bras
        
        current_idx=np.argmax(idx) # On cherche l'indice du bras à la plus forte valeur

          #   p_t=p_space[current_idx]
        s[current_idx]+=1

        R[current_idx] = gain.resMed(current_idx+1)[0]
        
        regret.append(regret[-1] + (-gain.resMed(current_idx)[1] + gain_max)/arg[2])
         
        loc = np.argmin(np.abs(p_space-np.array(R[current_idx])))
        
        reward=p_space[loc]
        if reward in obs[current_idx] :
            obs[current_idx][reward] += 1
        else:
            obs[current_idx][reward] = 1
       # print(obs)
        t = t + 1
#==============================================================================
#         if not(np.mod(t,np.round(arg[2]/10.).astype(int))):
#             print('Progress: '+ repr(100.*t/arg[2]) + '%.')
#==============================================================================
        
        
    return np.asarray(regret[1:])

        
        
        