
from abc import ABC, abstractmethod
import numpy as np

import skimage.io

import pdb


##################################################
# from projection_david.py
##################################################
class AbstractProjector(ABC):
    """
    Abstract class that represent a general projector
    """
    def __init__(self):
        super(AbstractProjector, self).__init__()

    @abstractmethod
    def project_point(self, points):
        pass

    @abstractmethod
    def get_image_size(self, **kwargs):
        pass
class LinearProjector(AbstractProjector):

    def __init__(self, res_x, res_y,res_z):
        """

        Parameters
        ----------
        res_x: px / mt
        res_y: px / mt
        """
        self.res_x = res_x
        self.res_y = res_y
        self.res_z = res_z
        super(LinearProjector, self).__init__()


    def _init_min_point_values(self,min_x,min_y,min_z):
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z

    def _init_max_point_values(self,max_x,max_y,max_z):
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def project_point(self, points):
        rot=np.eye(3)
        rot_points = np.dot(rot, points.T).T 

        if len(rot_points.shape) < 2:
            rot_points = np.atleast_2d(rot_points)

        height, width = self.get_image_size(points=rot_points)
        
        xmin = self.min_x
        ymin = self.min_y

        x = rot_points[:, 0] - xmin
        y = rot_points[:, 1] - ymin

        i_img_mapping = np.floor(x * self.res_x).astype(int)
        j_img_mapping = np.floor(y * self.res_y).astype(int)

        lidx = ((i_img_mapping) % height) * width + j_img_mapping

        return lidx, i_img_mapping, j_img_mapping

    def get_image_size(self, **kwargs):
        """
        Return the image size
        :param kwargs:
        :return:
        """
        points = kwargs['points']

        height = np.ceil((self.max_x - self.min_x+1) * self.res_x).astype(int)#+1 from BMI
        width  = np.ceil((self.max_y - self.min_y+1) * self.res_y).astype(int)# +1 from BMI

        return height, width

class Projection:
    def __init__(self, proj_type, res_x=0.0, res_y=0.0, res_z=0.0, res_pitch=0.0, res_yaw=0.0, fov_pitch = None, fov_yaw = None):
        """
        Projection class

        Parameters
        ----------
        proj_type: str

        res_x: float
            resolution along the rows of the image
        res_y: float
            resolution along the columns of the image
        """

        self.res_x = res_x
        self.res_y = res_y
        self.res_y = res_z
        self.res_pitch = res_pitch
        self.res_yaw = res_yaw

        if fov_yaw is None:
            fov_yaw = [0.0, 2 * np.pi]

        if fov_pitch is None:
            fov_pitch = [0.0, np.pi]

        if proj_type == 'linear':
            self.projector = self.__initialize_linear_proj(res_x=res_x, res_y=res_y,res_z=res_z)
        elif proj_type == 'spherical':
            self.projector = self.__initialize_spherical_proj(res_yaw=res_yaw,
                                                              res_pitch=res_pitch,
                                                              fov_yaw=fov_yaw,
                                                              fov_pitch=fov_pitch)
        else:
            raise ValueError("proj_type value can be only 'linear' or 'spherical',  you passed {}".format(proj_type))


    def __initialize_linear_proj(self, res_x, res_y,res_z):
        return LinearProjector(res_x=res_x, res_y=res_y,res_z=res_z)

    def _i_nitialize_spherical_proj(self, res_yaw, res_pitch, fov_yaw, fov_pitch):
        return SphericalProjector(res_yaw=res_yaw, res_pitch=res_pitch, fov_yaw=fov_yaw, fov_pitch=fov_pitch)

    def project_points_values(self, points, values, aggregate_func='max', rot=np.eye(3), b=0.0):
        """
        Function that project an array of values to an image

        Parameters
        ----------
        points: ndarray
            Array containing the point cloud

        values: ndarray
            Array containing the values to project

        aggregate_func: optional {'max', 'min', 'mean'}
            Function to use to aggregate the information in case of collision, i.e. when two or more points
            are projected to the same pixel.
            'max': take the maximum value among all the values projected to the same pixel
            'min': take the minimum value among all the values projected to the same pixel
            'mean': take the mean value among all the values projected to the same pixel


        Returns
        -------
        proj_img: ndarray
            Image containing projected values
        """

        rot_points = np.dot(rot, points.T).T + b

        nr, nc = self.projector.get_image_size(points=rot_points)

        if len(values.shape) < 2:
            channel_shape = 1
            values = np.atleast_2d(values).T
        else:
            _, channel_shape = values.shape[:2]

        if channel_shape > 1:
            if type(aggregate_func) is str:
                aggregators = [aggregate_func] * channel_shape
            else:
                assert len(aggregate_func) == channel_shape
                aggregators = aggregate_func
        else:
            aggregators = [aggregate_func]
        # we verify that the length of the two arrays is the same
        # that is for each point we have a corresponding value to project
        assert len(rot_points) == len(values)
        # project points to image
        lidx, i_img_mapping, j_img_mapping = self.projector.project_point(rot_points)

        # initialize binned_feature map
        binned_values = np.zeros((nr * nc, values.shape[1]))

        if 'max' in aggregators or 'min' in aggregators:
            # auxiliary variables to compute minimum and maximum
            sidx = lidx.argsort()
            idx = lidx[sidx]
            # we select the indices of the first time a unique value in lidx appears
            # np.r_[True, idx[:1] != idx[1:]] is true if an element in idx is different than its successive
            # flat non zeros returns indices of values that are non zeros in the array
            m_idx = np.flatnonzero(np.r_[True, idx[:-1] != idx[1:]])

            unq_ids = idx[m_idx]

        if 'mean' in aggregators:
            # auxiliary vector to compute binned count
            count_input = np.ones_like(values[:, 0])
            binned_count = np.bincount(lidx, count_input, minlength=nr * nc)

        for i, func in zip(range(values.shape[1]), aggregators):

            if func == 'max':
                """
                Examples
                --------
                To take the running sum of four successive values:

                >>> np.add.reduceat(np.arange(8),[0,4, 1,5, 2,6, 3,7])[::2]
                array([ 6, 10, 14, 18])

                """

                print("MAX")
                binned_values[unq_ids, i] = np.maximum.reduceat(values[sidx, i], m_idx)

            elif func == 'min':

                print("MIN")
                binned_values[unq_ids, i] = np.minimum.reduceat(values[sidx, i], m_idx)

            elif func == 'sum':
                binned_values[:, i] = np.bincount(lidx, values[:, i], minlength=nr * nc)

            else:  # otherwise we compute mean values
                binned_values[:, i] = np.bincount(lidx, values[:, i], minlength=nr * nc)
                binned_values[:, i] = np.divide(binned_values[:, i], binned_count, out=np.zeros_like(binned_count),
                                                where=binned_count != 0.0)

        # reshape binned_features to feature map
        binned_values_map = binned_values.reshape((nr, nc, values.shape[1]))

        if channel_shape == 1:
            binned_values_map = binned_values_map[:, :, 0]

        return binned_values_map
##################################################
# from David_example.py
##################################################


def max_aggregation_proj(proj, points, labels, aggreg):
    h, w = proj.projector.get_image_size(points=points)
    lidx, i_img_mapping, j_img_mapping = proj.projector.project_point(points)
    max_projection = np.full([h, w], -1)
    aux_height_proj = np.full([h, w], -np.inf)
    aux_idx_proj = np.full([h, w], -np.inf)
    min_val_local = points.min(0)    
    for k in range(len(lidx)):
        h_coor = i_img_mapping[k]
        w_coor = j_img_mapping[k]

        #if h_coor >= h: h_coor = h - 1
        #if w_coor >= w: w_coor = w - 1
            
        if points[k, 2] > aux_height_proj[h_coor, w_coor]:
            max_projection[h_coor, w_coor] = labels[k]
            aux_height_proj[h_coor, w_coor] = points[k, 2]
    return max_projection, aux_idx_proj, [h,w,min_val_local], [lidx, i_img_mapping, j_img_mapping]

def project(proj,points):# semantic Kitti
    pZ = points[:,2]
    moved_z = pZ - proj.projector.min_z
    moved_z = np.clip(moved_z, a_min=0, a_max = np.max(moved_z))
    npZ = (np.floor(moved_z*proj.projector.res_z)+1).astype(int)

    # minimum projections
    npMin = proj.project_points_values(points, npZ, aggregate_func = 'min')

    # maximum projections
    npMax = proj.project_points_values(points, npZ, aggregate_func = 'max')

    # accumulation
    npAcc = proj.project_points_values(points, np.ones_like(points[:,2]), aggregate_func = 'sum')
    npAcc = np.clip(npAcc, 0, 255)

    # project gt
    ###npClass, _,_,_  = max_aggregation_proj(proj, points, labels, 'max')
    if(0):
        npMin=np.swapaxes(npMin,0,1)
        npMax=np.swapaxes(npMax,0,1)
        npAcc=np.swapaxes(npAcc,0,1)

    return npMin,npMax,npAcc#,npClass


def initMinMaxPointValues(proj,points,percent):
    b=0.0
    rot=np.eye(3)
    rot_points = np.dot(rot, points.T).T + b

    min_X, min_Y, min_Z = rot_points.min(0)
    max_X, max_Y, max_Z = rot_points.max(0)
    zL = rot_points[:,2]
    min_Z = np.percentile(zL,percent)# 0.5% ranking

    inv_res_x = 1.0/proj.projector.res_x
    inv_res_y = 1.0/proj.projector.res_y
    min_X = np.floor(min_X/inv_res_x)*inv_res_x
    min_Y = np.floor(min_Y/inv_res_y)*inv_res_y

    proj.projector._init_min_point_values(min_X,min_Y,min_Z)
    proj.projector._init_max_point_values(max_X,max_Y,max_Z)


def backProjection(proj,points,npLabels,predLabels = None):
    lidx, i_img_mapping, j_img_mapping = proj.projector.project_point(points)
    if(predLabels is None):
        predLabels = np.zeros(len(i_img_mapping))
    for n in range(len(i_img_mapping)):
        coor_i = i_img_mapping[n]
        coor_j = j_img_mapping[n]
        
        if(0):#debug
            xpos,ypos=882,265
            if(coor_j==xpos)and(coor_i==ypos):
                print("pos=",n,points[n])
                print("HERE I AM")
                pdb.set_trace()
        myLab =  npLabels[coor_i, coor_j]
        if(myLab>0):
            predLabels[n] = npLabels[coor_i, coor_j]
        if n % 1000000 == 0:
            print(n, 'of', len(i_img_mapping))
    return predLabels



def seeLHPoints(proj,points,imMNT,HSlice):

    # imresL contains the objects in the lower slice, not the ground! MNT seems enough to get the ground at the end

    # Le calcul de npZ (echelle image) a deja ete fait. Voir si on peut le recuperer...
    pZ = points[:,2]
    moved_z = pZ - proj.projector.min_z
    moved_z = np.clip(moved_z, a_min=0, a_max = np.max(moved_z))
    npZ = (np.floor(moved_z*proj.projector.res_z)+1).astype(int)
    pMNTZ=backProjection(proj,points,imMNT)
    pDSMZ = npZ - pMNTZ


    predLabels = np.zeros(len(points))

    pointsL=points[pDSMZ <= HSlice]
    pointsH=points[pDSMZ > HSlice]

    predLabels[pDSMZ <= HSlice]=1
    predLabels[pDSMZ > HSlice]=2

    fn = os.path.join(".\\","toto_resLH.las")
    writeLas(points,fn,labels = predLabels)


def backProjectionGround(proj,points,imMin,imGround,deltaGround,groundId=3,predLabels = None):

    # Le calcul de npZ (echelle image) a deja ete fait. Voir si on peut le recuperer...
    pZ = points[:,2]
    moved_z = pZ - proj.projector.min_z
    moved_z = np.clip(moved_z, a_min=0, a_max = np.max(moved_z))
    npZ = (np.floor(moved_z*proj.projector.res_z)+1).astype(int)

    imtmp = np.zeros(imMin.shape)
    mymax = np.amax(imMin)+1
    imtmp[:,:]=imMin[:,:]
    imtmp[imGround==0]=mymax
    pMNTZ=backProjection(proj,points,imtmp)
    pDSMZ = npZ - pMNTZ


    delta = deltaGround * proj.projector.res_z
    #pixel labelled as ground (<mymax), and point not too far (deltaGround) from min
    idx = ((pMNTZ < mymax) & (pDSMZ <= delta) ) #& (predLabels != carId)


    if(predLabels is None):
        predLabels = np.zeros(len(pZ))

    predLabels[idx]=groundId 

    return predLabels

def backProjectionAll(proj,points,imGround,imMNT,imresL,imresH,deltaGround,HSlice):
    # imresL contains the objects in the lower slice, not the ground! MNT seems enough to get the ground at the end

    # Le calcul de npZ (echelle image) a deja ete fait. Voir si on peut le recuperer...
    pZ = points[:,2]
    moved_z = pZ - proj.projector.min_z
    moved_z = np.clip(moved_z, a_min=0, a_max = np.max(moved_z))
    npZ = (np.floor(moved_z*proj.projector.res_z)+1).astype(int)

    # First imresH is backprojected
    # The label is assigned to all 3D points of each pixel
    predLabels=backProjection(proj,points,imresH)

    if(0):# debug
        fn = os.path.join(".\\","toto_resH.las")
        writeLas(points,fn,labels = predLabels)

    pMNTZ=backProjection(proj,points,imMNT)
    pDSMZ = npZ - pMNTZ

    #HSlice = 2.5*proj.projector.res_z
    pointsL=points[pDSMZ <= HSlice]

    predLabelsL = backProjection(proj,pointsL,imresL)
    predLabels[pDSMZ <= HSlice]=predLabelsL
    
    if(0):# debug
        fn = os.path.join(".\\","toto_resL.las")
        writeLas(points,fn,labels = predLabels)

    if(0):
        predLabels[pDSMZ <= deltaGround]=groundId 
    else:
        idx = (pDSMZ <= deltaGround) & (predLabels != carId)
        predLabels[idx]=groundId 
    if(0):# debug
        fn = os.path.join(".\\","toto_resAll.las")
        writeLas(points,fn,labels = predLabels)
    print("END PROJECT ALL")

    return     predLabels
