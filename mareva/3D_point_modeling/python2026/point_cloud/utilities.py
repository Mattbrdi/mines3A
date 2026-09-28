import sys,os
import numpy as np
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go

import skimage as sk
import skimage.morphology as skm
from skimage import io
from point_cloud.skimage_mareva_projection import *
import matplotlib.pyplot as plt


# UTILITIES.
# load_point_cloud(path, has_header=True):
#points = work[xyz_cols]
#gt3DLabels = work['REPLICA_semantic']
# showPC(points, labels=work['intensity'])
# showPC(points, labels=work['REPLICA_semantic'])

def _find_xyz_columns(df):
    lower = {str(c).strip().lower(): c for c in df.columns}
    candidates = [
        ("x", "y", "z"),
        ("posx", "posy", "posz"),
        ("coordx", "coordy", "coordz"),
    ]
    for names in candidates:
        if all(n in lower for n in names):
            return [lower[n] for n in names]

    numeric = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric) >= 3:
        return numeric[:3]
    raise ValueError("Impossible de trouver trois colonnes numériques X, Y et Z.")

def load_point_cloud(path, has_header=True):
    suffix = path.suffix.lower()

    if suffix in {".csv", ".txt", ".xyz"}:
        # sep=None permet à pandas de détecter automatiquement virgule, tabulation,
        # espace, point-viHArgule, etc.
        header = 0 if has_header else None
        df = pd.read_csv(path, sep=None, engine="python", header=header, comment="#")
        if not has_header:
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
        xyz = _find_xyz_columns(df)
        return df, xyz

    if suffix == ".ply":
        from plyfile import PlyData
        ply = PlyData.read(str(path))
        vertex = ply["vertex"].data
        names = vertex.dtype.names
        lower = {n.lower(): n for n in names}
        try:
            xyz = [lower["x"], lower["y"], lower["z"]]
        except KeyError:
            raise ValueError("Le fichier PLY ne contient pas les propriétés x, y et z.")
        df = pd.DataFrame({n: vertex[n] for n in names})
        return df, xyz

    if suffix == ".pcd":
        try:
            import open3d as o3d
        except ImportError as e:
            raise ImportError("Pour lire un PCD, installe Open3D avec : %pip install open3d") from e
        cloud = o3d.io.read_point_cloud(str(path))
        xyz_array = np.asarray(cloud.points)
        df = pd.DataFrame(xyz_array, columns=["x", "y", "z"])
        return df, ["x", "y", "z"]

    raise ValueError(f"Format non pris en charge : {suffix}")

def showPC(points, labels=None,mode = None, point_size=2.5, opacity=0.9):
    """
    Display a 3D point cloud using Plotly.

    Parameters
    ----------
    points : pandas.DataFrame
        DataFrame containing  x, y, z
    labels: the color of each point (pandas.DataFrame also)
    mode : str
        Display mode:
         - default value: None
         - 'rgb' to visualize colored points

    point_size : float
        Size of the displayed points.

    opacity : float
        Point opacity, between 0 and 1.

    Returns
    -------
    plotly.graph_objects.Figure
        The generated Plotly figure.
    """

    # ---------------------------------------------------------
    # 2. Prepare marker colors
    # ---------------------------------------------------------

    marker = {"size": point_size, "opacity": opacity}

    if(mode == 'rgb'):
        marker["color"] = [ f"rgb({r},{g},{b})"   for r, g, b in labels ]
    else:# display_mode == "intensity","x","z","REPLICA_CLASS":
      marker.update(
          color=labels,
          colorscale="Viridis",
          showscale=True,
          colorbar=dict(title="class"),
        )
    # ---------------------------------------------------------
    # 4. Create the 3D Plotly figure
    # ---------------------------------------------------------
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points['x'],
                y=points['y'],
                z=points['z'],
                mode="markers",
                marker=marker,
                customdata=labels
            )
        ]
    )

    # Keep the same scale on X, Y and Z so that the geometry
    # of the point cloud is not distorted.
    fig.update_layout(
        title=f"3D Point Cloud",
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z',
            aspectmode="data"
        ),
        margin=dict(
            l=0,
            r=0,
            t=45,
            b=0
        )
    )

    fig.show()

    #return fig

def show(im):
    plt.clf()
    plt.imshow(im)
    plt.show()


def readBEVImages(bevP,imList):
    bevDir,imName=bevP.dirBEV,bevP.file_i

    bevI=bevImagesClass(bevP)
    for attributeName in imList:

        fn = os.path.join(bevDir,imName+"_"+attributeName+".png")

        attributeInstance=io.imread(fn)
        #attributeInstance=np.swapaxes(attributeInstance,0,1)

        setattr(bevI,attributeName,attributeInstance)
    return bevI
# ..................................................
#    Read and write BEV images (instead of time consuming projection)
# ..................................................
def writeBEVImages(bevI,bevP):
    bevDir,imName=bevP.dirBEV,bevP.file_i

    attrs = vars(bevI)
    for item in attrs.items():
        attributeName = item[0]
        attributeInstance = item[1]
        if( isinstance(attributeInstance,np.ndarray)):

            fn = os.path.join(bevDir,imName+"_"+attributeName+".png")
            if (np.amax(attributeInstance)> 255):
                io.imsave(fn,attributeInstance.astype(np.uint16))
            else:
                io.imsave(fn,attributeInstance.astype(np.uint8))


            if (np.amax(attributeInstance)> 255):
                im8 = np.zeros(attributeInstance.shape,"uint8")

                if(np.amax(attributeInstance)<256):
                    im8=attributeInstance.copy().astype(np.uint8)
                elif(1):#divide by a constant
                    imtmp=attributeInstance/15
                    im8=imtmp.copy().astype(np.uint8)
                else:#clip
                    imtmp=attributeInstance.copy()
                    imtmp[attributeInstance>255]=255
                    im8=imtmp.astype(np.uint8)

                fn8 = os.path.join(bevDir,imName+"_"+attributeName+"_8.png")
                io.imsave(fn8,im8)

    if(0):
        fn8 = os.path.join(bevDir,imName+"_"+"classL_color.png")
        imColor = colorSeg(bevI.imClassL)
        io.imsave(fn8,imColor)

        fn8 = os.path.join(bevDir,imName+"_"+"classH_color.png")
        imColor = colorSeg(bevI.imClassH)
        io.imsave(fn8,imColor)




# ..................................................
#    Bird Eye View Projection
# ..................................................
def BEV(points,bevP):
    """ Bird Eye View Projection"""
    res_x,res_y,res_z = bevP.res_x,bevP.res_y,bevP.res_z

    proj = Projection(proj_type='linear',res_x=res_x, res_y=res_y, res_z=res_z)
    print(bevP.noisePercentage)
    initMinMaxPointValues(proj,points,bevP.noisePercentage)

    if(1):#min,max,acc
        imMin,imMax,imAcc=project(proj,points)#

        print("MAX=",np.amax(imMax))
    return imMin,imMax,imAcc,proj

#####################################################
# EVALUATION
#####################################################

from sklearn.metrics import confusion_matrix,precision_recall_fscore_support

def normalizeCM(conf_mat):
    conf_mat_norm=np.zeros(conf_mat.shape)
    mySum = conf_mat.astype(float).sum(axis=1)
    myLen=mySum.shape[0]

    for i in range(myLen):
        currentSum = mySum[i]
        if(currentSum>0):
            for j in range(myLen):
                conf_mat_norm[i,j]=conf_mat[i,j]/mySum[i]
    conf_mat_norm =    np.around(conf_mat_norm,decimals=2)
    return conf_mat_norm

def evaluate3D(predLabels,labels):
    # GT: 21,23,24 -> road
    gtLabels = np.zeros(predLabels.shape)
    for lab in [21,23,24]:
        gtLabels = np.where((labels==lab),1,gtLabels)

    #Pred: 21 -> road
    pred2Labels = np.zeros(predLabels.shape)
    pred2Labels = np.where((predLabels==21),1,pred2Labels)


    conf_mat = confusion_matrix(gtLabels,pred2Labels)

    conf_mat=conf_mat.astype(int)
    conf_mat_norm=normalizeCM(conf_mat)

    return conf_mat,conf_mat_norm

roadId, sidewalkId,lane_markingId = 21,23,24
facadeId = [63]
groundLikeId=[roadId, sidewalkId,lane_markingId]
groundId = 21    


class bevImagesClass:
    def __init__(self,bevP):
        self.res_x = bevP.res_x
        self.res_y = bevP.res_y
        self.res_z = bevP.res_z

class bevParameters:
    def __init__(self,dirPC,file_i,dirBEV,res_x,res_y,res_z):
        self.file_i = file_i
        self.removeOutliers= False
        self.noisePercentage = 0.2
        self.oriSuffix= "_labeled_bin"
        self.nbSlices = 1
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z

        self.dirPC = dirPC
        self.dirBEV = dirBEV
        self.groundId= 21
class groundParameters:
    def __init__(self,myLambda,bevP):
        self.myLambda = int(myLambda*bevP.res_z)


