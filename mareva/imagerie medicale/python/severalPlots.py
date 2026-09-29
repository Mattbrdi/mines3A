
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as cl

import random

import numpy as np
def randColormap(SEED = 448):
    random.seed(SEED)
    randarray = np.random.rand(255,3)
    randarray[0] = [0,0,0]
    cmap = cl.ListedColormap(randarray)
    return cmap


# Global colormap used as a unique LUT
randcmap = randColormap(448)


def singlePlot(imInArr,ax,currentLab, norm = True):
    if(np.amax(imInArr)<=1):
        imtmp = imInArr*255
    else:
        imtmp = imInArr
    if(np.amax(imtmp)>255):
        mV=2**16-1
    else:
        mV=255

    if currentLab == False:
        if norm:
            ax.imshow(imtmp,cmap = cm.Greys_r)
        else:
            ax.imshow(imtmp,cmap = cm.Greys_r, vmin=0, vmax=mV)
    else: 
        ax.imshow(imtmp,cmap = randcmap) # cm.Paired

class pClass:
    def __init__(self):
        print("")

def mShow(listIm,labelIm = []):
    nbIm = len(listIm)
    plt.figure()
    gs1 = gridspec.GridSpec(1, nbIm)
    gs1.update(left=0.05, right=1.7, wspace=0.1)
    ax = pClass()
    for i in range(nbIm):
        attributeName = "ax"+str(i)
        setattr(ax,attributeName,plt.subplot(gs1[0,i]))
    #
    attrs = vars(ax)
    for item in attrs.items():
        attributeName = item[0]
        attributeInstance = item[1]
        i = int(attributeName[-1])
        if(i>len(labelIm)-1):
           singlePlot(listIm[i],attributeInstance, False,norm=True)
        else:
           singlePlot(listIm[i],attributeInstance, labelIm[i])

