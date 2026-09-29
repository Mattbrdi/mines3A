# TP MAREVA Etalonnage de capteurs - Python - FG 17/09/2021
# coding=utf8

# Import Numpy
import numpy as np

# Import pyplot pour visualisation
from matplotlib import pyplot as plt


def show_calibration(x,y):
    '''
    Visualisation d'une courbe d'étalonnage
    Entrées :
        x : grandeurs réelles
        y : grandeurs mesurées
    '''
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(x, y, '.')
    plt.show()


def linefit(x,y):
    '''
    Calcul des paramètres linéaires a, b estimés et du résidu
    Entrées :
        x : grandeurs réelles
        y : grandeurs mesurées
    Sorties :
        a : offset estimé
        b : erreur linéaire estimée
        res : résidu
    '''
    N=np.size(x)
    S=N
    Sx=sum(x)
    Sy=sum(y)
    Sxx=sum(x*x)
    Sxy=sum(x*y)
    
    delta = S*Sxx - Sx * Sx 
    a = (Sxx*Sy - Sx*Sxy)/delta
    b = (S*Sxy - Sx*Sy)/delta
    res = (sum((y - b*x - a)**2)/N)**0.5

    return a, b, res


def sensordata_generator(x, a, b, sigma):
    '''
  (Sxx*Sy - Sx*Sxy)/delta()S*Sxy - Sx*Sy)/delta   Générateur de données capteurs
    Entrées :
        x : grandeurs réelles
        a : offset
        b : erreur linéaire
        sigma : écart-type de bruit gaussien
    Sortie :
        y : mesures
    '''

    y = a + b*x + np.random.normal(size = len(x))*sigma
    return y



#
#           Main
#       \**********/
#

if __name__ == '__main__':


    # Fichier de données d'étalonnage
    data_path = 'sensordata.npy'

    # Lecture des données d'étalonnage
    if True:
        (x,y)=np.load(data_path)
    
    # Visualisation des données d'étalonnage
    if True:
        show_calibration(x,y)

    # Estimation des erreurs d'offset et de gain
    if True:
        (a,b,res)=linefit(x,y)
        print('Offset : a =',a)
        print('Erreur de gain : b =',b)
        print('res =',res)

        
    # Vérification de la loi du X2
    # Génération de N jeux de données d'étalonnage et estimation des paramètres
    if True:
        x=np.arange(0,20,0.01);        
        a=-0.4
        b=1.05
        sigma=2.5;        

        y = sensordata_generator(x,a,b,sigma)      
        show_calibration(x,y)
        (a_est,b_est,res)=linefit(x,y)
        print('Offset : a_est =',a_est)
        print('Erreur de gain : b_est =',b_est)
        print('res =',res)
  
        N = 100
        Res = np.zeros(N)
        for i in range(N):
            y = sensordata_generator(x,a,b,sigma)      
            (a_est,b_est,Res[i])=linefit(x,y)
        plt.hist(Res)
        plt.show()
        





    