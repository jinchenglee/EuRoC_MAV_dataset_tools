
# Load python packages
import cv2
import numpy as np
import glob
import yaml
import csv

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

#--------------------------------
# Dataset location
#--------------------------------
# Change this to the directory where you store EuRoC MAV data
#basedir = '/work/asl_dataset/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/'
#basedir = '/Users/jcli/study/asl_dataset/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/'
basedir = '/media/data/EuRoC_MAV_datset/asl_dataset/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/'

#--------------------------------
# Global variables:
#   Unique ID for points and frames
#--------------------------------
pid = point_id() # point ID
#fid = 0 # frame ID

#-------------
# Global Parameters
#-------------
# Tile size
tile_size = 200

# List of camera data
frame_img_list = np.sort(glob.glob(basedir+'mav0/cam0/data/*.png'))
# No of frames to process - !!!process only two frames!!!, last digit is the gap
START_FRAME = 0
STEP = 3
frame_range = range(START_FRAME, START_FRAME+STEP+1, STEP)

#--------------------------------
# Camera distortion correction
#--------------------------------
# Use page: https://hackaday.io/project/12384-autofan-automated-control-of-air-flow/log/41862-correcting-for-lens-distortions

camera_sensor_config_yaml = basedir+'mav0/cam0/sensor.yaml'
camera = Camera(camera_sensor_config_yaml)

## Visualization
## Test camera undistortion below
##
## Remap the original image to a new image
## List of camera data
#img_list = np.sort(glob.glob(basedir+'mav0/cam0/data/*.png'))
## Read 1st frame
#frame0 = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
#
#new_frame0= cv2.remap(frame0, camera.mapx, camera.mapy, cv2.INTER_LINEAR,
#                      cv2.BORDER_TRANSPARENT, 0)
## Display old and new image
#dpi = 50
## Create a figure of the right size with one axes that takes up the full figure
#height, width = frame0.shape
## What size does the figure need to be in inches to fit the image?
#figsize = width / float(dpi), height / float(dpi)
#plt.figure(figsize=figsize)
## Show the gray img from camera 0
#plt.subplot(121)
#plt.imshow(frame0, cmap='gray', extent=[0,width,height,0])
#plt.title('Original image')
#plt.subplot(122)
#plt.imshow(new_frame0, cmap='gray', extent=[0,width,height,0])
#plt.title('Distortion corrected image')
#plt.show()



#--------------------------------
# Monocular Init w/ Tile-based OF 
#--------------------------------

# Read 1st frame
frame0_ori = cv2.imread(frame_img_list[frame_range[0]], cv2.IMREAD_GRAYSCALE)
# Image undistortion
frame0 = cv2.remap(frame0_ori, camera.mapx, camera.mapy, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT, 0)

# Read 2nd frame
fr = frame_range[1]
print("Frame ", fr, ":")
frame1_ori = cv2.imread(frame_img_list[fr], cv2.IMREAD_GRAYSCALE)
# Image undistortion
frame1 = cv2.remap(frame1_ori, camera.mapx, camera.mapy, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT, 0)

# Split image into tile_size to find features
p0 = of.OF_TileFindFeature(frame0, tile_size, of.feature_params)

# Forward OF tracking
p1, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0, None, **of.lk_params)

good_old = p0[st==1]
good_new = p1[st==1]

# Visualization
#cur_rand_color = (np.random.rand(),0.7, np.random.rand(),1.0)
## Draw optic flow on last frame (frame1)
#height, width = frame1.shape
#plt.imshow(frame1, cmap='gray', extent=[0,width,height,0])
#plt.title("Last frame")
#plt.scatter(good_old[:,0], good_old[:,1], linestyle='-', c='g', s=2)
#plt.scatter(good_new[:,0], good_new[:,1], linestyle='-', c='r', s=2)
#
#for idx in range(good_old.shape[0]):
#    plt.plot([good_old[idx,0], good_new[idx,0]],
#                    [good_old[idx,1], good_new[idx,1]],
#                    linestyle='-', color=cur_rand_color, markersize=0.25)
#plt.show()


#-------------------------------------------------------------------------------------
# Start pose estimation process
#-------------------------------------------------------------------------------------
# Construct homogeneous coordinates points
one_col = np.ones_like(good_new[:,0]).reshape(-1,1)
pts_c = np.hstack((good_new[:,:2], one_col))
pts_p = np.hstack((good_old[:,:2], one_col))

# matched image points
img_pts_all = np.zeros([2*pts_c.shape[0], pts_c.shape[1]-1])
img_pts_all[::2,:] = pts_c[:,:-1]
img_pts_all[1::2,:]= pts_p[:,:-1]
img_pts_all = img_pts_all.reshape(-1,2,2)

#-------------------
# Use all points
#-------------------
print("\n--------------------------------")
print("Initial pose estimation using all points:")
print("--------------------------------")
# Calculate fundamental matrix F
F = normalized_eight_point_alg(pts_c, pts_p)

# Essential matrix
E = camera.K.T.dot(F).dot(camera.K)
#
# Estimate RT matrix from E
RT = estimate_RT_from_E(E, img_pts_all, camera.K)

# Reproj erorr
init_err, inliers_cnt, _ = eval_RT_thresh(RT, img_pts_all, camera.K)
print("Mean reproj error: ", init_err, "inliers/total:", inliers_cnt, "/", img_pts_all.shape[0], "\n")

pFp = [pts_p[i].dot(F.dot(pts_c[i]))
    for i in range(pts_c.shape[0])]
print("p'^T F p =", np.abs(pFp).max())
#print("Fundamental Matrix from normalized 8-point algorithm:\n", F)
print("Estimated pose RT:\n", RT)
#plot_points_on_images(pts_c, pts_p, frame0, frame1, F)
plt.imshow(frame1, cmap='gray')
cur_color = (0.1, 0.7, 0.3, 1.0)
for idx in range(pts_c.shape[0]):
    plt.plot([pts_p[idx,0], pts_c[idx,0]], 
                [pts_p[idx,1], pts_c[idx,1]], 
                linestyle='-', color=cur_color, markersize=0.25)
plt.savefig("8pt_alg_allpoints.png")
plt.show()


#-------------------
# Ransac 8 pts
#-------------------
print("\n--------------------------------")
print("Started RANSAC 8pt algorithm:")
print("--------------------------------")

RANSAC_TIMES = 100
INLIER_RATIO_THRESH = 0.8

#min_err = init_err.copy()
min_err = 1e8
min_RT = np.empty((3,4))
min_inliers_list = []

for i in range(RANSAC_TIMES):
    ransac_8 = np.random.randint(0, pts_c.shape[0], size=8)
    rand_pts_c = pts_c[ransac_8]
    rand_pts_p = pts_p[ransac_8]

    # Calculate fundamental matrix F
    F = normalized_eight_point_alg(rand_pts_c, rand_pts_p)

    # Essential matrix
    E = camera.K.T.dot(F).dot(camera.K)

    # matched image points
    img_pts = np.zeros([2*rand_pts_c.shape[0], rand_pts_c.shape[1]-1])
    img_pts[::2,:] = rand_pts_c[:,:-1]
    img_pts[1::2,:]= rand_pts_p[:,:-1]
    img_pts = img_pts.reshape(-1,2,2)

    # Estimate RT matrix from E
    RT = estimate_RT_from_E(E, img_pts, camera.K)

    # Reproj erorr
    err, inliers_cnt, inliers_list = eval_RT_thresh(RT, img_pts_all, camera.K)

    if err < min_err and (inliers_cnt/img_pts_all.shape[0])>INLIER_RATIO_THRESH:
        print("Mean reproj error: ", err, "Inlier/total=", inliers_cnt, "/", img_pts_all.shape[0])

        min_err = err.copy()
        min_F = F
        min_RT = RT
        min_inliers_list = inliers_list


print("\nAfter 8pt algorithm RANSAC pose estimation:")

pFp = [pts_p[i].dot(min_F.dot(pts_c[i]))
            for i in min_inliers_list]
print("p'^T F p =", np.abs(pFp).max())
print("Fundamental Matrix from normalized 8-point algorithm:\n", min_F)
print("Estimated pose RT:\n", min_RT)
# Plotting the remapped points according to epipolar lines
#plot_points_on_images(pts_c, pts_p, frame0, frame1, min_F)
plt.imshow(frame1, cmap='gray')
# Draw all inliers optical flow
cur_color = (0.1, 0.7, 0.3, 1.0)
for idx in min_inliers_list:
    plt.plot([pts_p[idx,0], pts_c[idx,0]], 
                [pts_p[idx,1], pts_c[idx,1]], 
                linestyle='-', color=cur_color, markersize=0.25)
# Draw all points
plt.scatter(pts_p[:,0], pts_p[:,1], linestyle='-', c='g', s=2)
plt.scatter(pts_c[:,0], pts_c[:,1], linestyle='-', c='r', s=2)
plt.savefig("8pt_alg_ransac.png")
plt.show()





