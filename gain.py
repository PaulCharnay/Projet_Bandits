# -*- coding: utf-8 -*-

import numpy as np


#k : Numero de la machine
#J : Cout du jeton
def testGain(k, J) :
    proba = [0.2,1.2,1.5,0.7,1]
    return np.random.poisson(proba[k-1]+J) - J, proba[k-1], np.max(proba)


#k : numero du medicament
def resMed(k) :
    proba = [0.2,0.75,0.6,0.8,0.4]
    return np.random.binomial(1,proba[k-1]), proba[k-1], np.max(proba)