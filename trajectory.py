##timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z [], v_RS_R_x [m s^-1], v_RS_R_y [m s^-1], v_RS_R_z [m s^-1], b_w_RS_S_x [rad s^-1], b_w_RS_S_y [rad s^-1], b_w_RS_S_z [rad s^-1], b_a_RS_S_x [m s^-2], b_a_RS_S_y [m s^-2], b_a_RS_S_z [m s^-2]
#1403636645813555456,-1.779407,5.202987,0.624628,0.548623,-0.142665,-0.818754,-0.091113,-0.752524,0.671032,-0.062724,-0.003194,0.021295,0.078437,-0.026085,0.137572,0.076266
#1403636645963555584,-1.895176,5.304111,0.630725,0.551714,-0.145586,-0.816311,-0.089752,-0.793350,0.689786,0.125462,-0.003194,0.021295,0.078436,-0.026090,0.137572,0.076268

# Load python packages
import cv2
import numpy as np
import glob
import yaml
import csv
import sys

import scipy
import scipy.linalg 
from scipy.misc import imread

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

#fr1 = 8600
#fr2 = 10000
fr1 = 28600
fr2 = 30000

def draw_oxyz(ax, OXYZ):
    ax.plot([OXYZ[0,0], OXYZ[1,0]], [OXYZ[0,1], OXYZ[1,1]], [OXYZ[0,2], OXYZ[1,2]],c='r')
    ax.plot([OXYZ[0,0], OXYZ[2,0]], [OXYZ[0,1], OXYZ[2,1]], [OXYZ[0,2], OXYZ[2,2]],c='g')
    ax.plot([OXYZ[0,0], OXYZ[3,0]], [OXYZ[0,1], OXYZ[3,1]], [OXYZ[0,2], OXYZ[3,2]],c='b')


def q2R(w, x, y, z):
    """
    Transform a unit quaternion into its corresponding rotation matrix (to
    be applied on the right side).
      
    :returns: transform matrix
    :rtype: numpy array
  
    """
    xx2 = 2 * x * x
    yy2 = 2 * y * y
    zz2 = 2 * z * z
    xy2 = 2 * x * y
    wz2 = 2 * w * z
    zx2 = 2 * z * x
    wy2 = 2 * w * y
    yz2 = 2 * y * z
    wx2 = 2 * w * x
      
    R = np.empty((3, 3), float)
    R[0,0] = 1. - yy2 - zz2
    R[0,1] = xy2 - wz2
    R[0,2] = zx2 + wy2
    R[1,0] = xy2 + wz2
    R[1,1] = 1. - xx2 - zz2
    R[1,2] = yz2 - wx2
    R[2,0] = zx2 - wy2
    R[2,1] = yz2 + wx2
    R[2,2] = 1. - xx2 - yy2
      
    return R

basedir = '/Users/jcli/study/asl_dataset/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/'
# Load cam0 sensor config\n",
with open(basedir+'mav0/cam0/sensor.yaml') as fp:
    cam0_yaml = yaml.load(fp)
#print(cam0_yaml)\n",
fp.close()

# T_BS - [R t] from MAV body coordinates to sensor (cam0) coordinates
T_BS = cam0_yaml['T_BS']['data']
T_BS_cols = cam0_yaml['T_BS']['cols']
T_BS_rows = cam0_yaml['T_BS']['rows']
T_BS = np.array(T_BS).reshape(T_BS_rows, T_BS_cols)
print("cam0 T_BS (MAV body to camera sensor) = \n", T_BS)


# Load leica0 sensor config - position measurement mounted on IMU (ADIS16448)
with open(basedir+'mav0/leica0/sensor.yaml') as fp:
    leica0_yaml = yaml.load(fp)
fp.close()
# T_BL - [R t] from MAV body coordinates to sensor (leica0) coordinates
#   As it is rigidly mounted on IMU, there's only translation from B.
T_BL = leica0_yaml['T_BS']['data']
T_BL_cols = leica0_yaml['T_BS']['cols']
T_BL_rows = leica0_yaml['T_BS']['rows']
T_BL = np.array(T_BL,dtype='float').reshape(T_BL_rows, T_BL_cols)
print("leica0 T_BL (MAV body to leica0 prism marker) = \n", T_BL)

# Trajectory points
T_WL_x = []
T_WL_y = []
T_WL_z = []

T_WL =[]
TS = []

# Load ground truth data of leica0 prism marker
CSV_READ_N_LINES = fr2
with open(basedir+'mav0/state_groundtruth_estimate0/data.csv', newline='') as fp:
    reader = csv.reader(fp)
    # Skip first line, specifying column contents category
    next(reader)
    for i,row in enumerate(reader):
        timestamp, tx, ty, tz, qw, qx, qy, qz = np.array(row[0:8]).astype('float')
        #print(tx, ty, tz, qw, qx, qy, qz)
        R = q2R(qw, qx, qy, qz)
        t = np.array([tx,ty,tz]).reshape(3,1)
        RT = np.hstack([R, t])
        #print("T_WL (world to leica0) = \n", T_WL)
        
        T_WL.append(RT)
        TS.append(timestamp)
        T_WL_x.append(tx)
        T_WL_y.append(ty)
        T_WL_z.append(tz)
        
        if i > CSV_READ_N_LINES:
            break
        
fp.close()

# Convert to array
TS = np.asarray(TS)
T_WL = np.asarray(T_WL)
# Saving into files
np.save("TS.npy", TS)
np.save("T_WL.npy", T_WL)


# Draw the trajectory 
matplotlib.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter3D(T_WL_x[fr1:], T_WL_y[fr1:], T_WL_z[fr1:], c=T_WL_z[fr1:], label='leica0 position')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#ax.set_aspect('equal')
# Top-down view?
#ax.view_init(azim=270, elev=90)
#
ax.view_init(azim=60, elev=50)
#plt.show()

OXYZ = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])

tmp = T_WL[fr1,:,:-1].dot(OXYZ.T) + T_WL[fr1,:,-1].reshape(3,1)
OXYZ1 = tmp.T
tmp = T_WL[fr2,:,:-1].dot(OXYZ.T) + T_WL[fr2,:,-1].reshape(3,1)
OXYZ2 = tmp.T

draw_oxyz(ax, OXYZ)
draw_oxyz(ax, OXYZ1)
draw_oxyz(ax, OXYZ2)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')
plt.show()
T_WL[fr1], T_WL[fr2]
