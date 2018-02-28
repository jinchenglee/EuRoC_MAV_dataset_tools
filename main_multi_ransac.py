
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
basedir = '/Users/jinchengli/study/asl_dataset/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/'
#basedir = '/media/data/EuRoC_MAV_datset/asl_dataset/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/'

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
#START_FRAME = 324
START_FRAME = 1321
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
print("1st frame ", frame_range[0], ":", frame_img_list[frame_range[0]])

# Read 2nd frame
fr = frame_range[1]
print("2nd frame ", fr, ":", frame_img_list[fr])
frame1_ori = cv2.imread(frame_img_list[fr], cv2.IMREAD_GRAYSCALE)
# Image undistortion
frame1 = cv2.remap(frame1_ori, camera.mapx, camera.mapy, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT, 0)

# Split image into tile_size to find features
p0 = of.OF_TileFindFeature(frame0, tile_size, of.feature_params)

# Forward OF tracking
p1, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame1, p0, None, **of.lk_params)

# Successfully tracked points
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

##-------------------
## Use all points
##-------------------
#print("\n--------------------------------")
#print("Initial pose estimation using all points:")
#print("--------------------------------")
#
#err, F, RT = ALLPOINTS_estimate_RT(pts_c, pts_p, camera)
#
#pFp = [pts_p[i].dot(F.dot(pts_c[i]))
#    for i in range(pts_c.shape[0])]
#print("p'^T F p =", np.abs(pFp).max())
##print("Fundamental Matrix from normalized 8-point algorithm:\n", F)
#print("Estimated pose RT:\n", RT)
#
### Visualization
###plot_points_on_images(pts_c, pts_p, frame0, frame1, F)
##plt.imshow(frame1, cmap='gray')
##cur_color = (0.1, 0.7, 0.3, 1.0)
##for idx in range(pts_c.shape[0]):
##    plt.plot([pts_p[idx,0], pts_c[idx,0]], 
##                [pts_p[idx,1], pts_c[idx,1]], 
##                linestyle='-', color=cur_color, markersize=0.25)
##plt.savefig("8pt_alg_allpoints.png")
###plt.show()

# Run mutilple times of 8pt ransac algo.
for ransac_times in range(3):
    #-------------------
    # Ransac 8 pts
    #-------------------
    print("\n--------------------------------")
    print("Started RANSAC 8pt algorithm:")
    print("--------------------------------")
    
    RANSAC_TIMES = 200
    INLIER_RATIO_THRESH = 0.6
    
    min_err, min_F, min_RT, min_inliers_list = RANSAC_estimate_RT(pts_c, pts_p, camera, 
                        RANSAC_TIMES, INLIER_RATIO_THRESH)
    
    print("\nAfter 8pt algorithm RANSAC pose estimation:")
    
    pFp = [pts_p[i].dot(min_F.dot(pts_c[i]))
                for i in min_inliers_list]
    print("p'^T F p =", np.abs(pFp).max())
    print("Fundamental Matrix from normalized 8-point algorithm:\n", min_F)
    print("Estimated pose RT:\n", min_RT)
    
    # Visualization
    if ransac_times==0:
        # Plotting the remapped points according to epipolar lines
        #plot_points_on_images(pts_c, pts_p, frame0, frame1, min_F)
        plt.imshow(frame1, cmap='gray')
        # Draw all inliers optical flow
        inlier_color = (0.1, 0.7, 0.3, 1.0)
        for idx in min_inliers_list:
            plt.plot([pts_p[idx,0], pts_c[idx,0]], 
                        [pts_p[idx,1], pts_c[idx,1]], 
                        linestyle='-', color=inlier_color, markersize=0.25)
        # Draw outliers optical flow
        outlier_color = (0.8, 0.1, 0.1, 0.7)
        for idx in range(pts_p.shape[0]):
            if idx not in min_inliers_list:
                plt.plot([pts_p[idx,0], pts_c[idx,0]], 
                        [pts_p[idx,1], pts_c[idx,1]], 
                        linestyle='-', color=outlier_color, markersize=0.25)
        # Draw all points
        plt.scatter(pts_p[:,0], pts_p[:,1], linestyle='-', c='g', s=2)
        plt.scatter(pts_c[:,0], pts_c[:,1], linestyle='-', c='r', s=2)
        plt.savefig("8pt_alg_ransac.png")
        #plt.show()
    
    
    #----------------------
    # Triangulate inlier points
    #----------------------
    # matched image points
    inlier_pts_p = pts_p[min_inliers_list]
    inlier_pts_c = pts_c[min_inliers_list]
    
    inlier_pts_3D = triangulate(inlier_pts_c, inlier_pts_p, camera, min_RT)

#----------------------
# Point cloud visualization
#----------------------
#import pandas as pd
#from pyntcloud import PyntCloud
#pd_dataframe = pd.DataFrame(inlier_pts_3D)
#pd_dataframe.columns = ['x', 'y', 'z']
#test = PyntCloud(pd_dataframe)
#test.plot(lines=inlier_pts_3D.tolist(), line_color=0xFF00FF)

    if ransac_times==0:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    reduced_pts = inlier_pts_3D[abs(inlier_pts_3D[:,2])<100]
    reduced_pts = reduced_pts[reduced_pts[:,2]>0]
    #ax.scatter(inlier_pts_3D[:,0], inlier_pts_3D[:,1], inlier_pts_3D[:,2])
    ax.scatter(reduced_pts[:,0], reduced_pts[:,1], reduced_pts[:,2], marker='x')

ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.view_init(azim=-89, elev=-50)
plt.savefig("pointcloud_multi_8pt_ransac.png")
plt.show()
