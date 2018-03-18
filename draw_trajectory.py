
# Load python packages
import cv2
import numpy as np
import glob
import yaml
import csv
import sys
import re

import scipy
import scipy.linalg 
from scipy.misc import imread

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# Local scripts
import tile_of_func as of
from utils import *
from camera import Camera
from points import *


# Draw world origin (assuming first frame camera origin as world origin)
# Notice it is DIFFERENT from ground truth world origin!
Cam_OXYZ = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
#draw_oxyz(ax, Cam_OXYZ)

# Draw estimation
draw_oxyz_gray(ax, Cam_OXYZ) # Origin
for i in range(len(R_mat)):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')
    # Top-down view?
    #ax.view_init(azim=270, elev=90)
    #
    #ax.view_init(azim=60, elev=50)
    ax.view_init(azim=-90, elev=-45)
    #plt.show()


    # Draw point cloud
    ax.scatter3D(inlier_pts_3D[:,0], inlier_pts_3D[:,1], inlier_pts_3D[:,2], c=inlier_pts_3D[:,2],
            label='Features Point Cloud')


    # set plot limit
    ax.set_xlim([-1.5,1.5])
    ax.set_ylim([-1.5,1.5])
    ax.set_zlim([-1.5,1.5])

    Rotation = np.linalg.inv(R_mat[i])
    Translation = T_vec[i]
    tmp = Rotation.dot((Cam_OXYZ - Translation).T)
    OXYZ1 = tmp.T
    draw_oxyz(ax, OXYZ1)
    
    plt.savefig("draw_traj_"+str(i)+".png")