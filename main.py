
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
basedir = '/Users/jcli/study/asl_dataset/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/'
#basedir = '/Users/jinchengli/study/asl_dataset/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/'
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
# Average Z threshold - If ave_Z is larger than normalized T=1 (translation between two frames used
#   for R,T estimation, the estimation is deemed fail and we'll use next frame as 2nd frame.)
AVE_Z_THRESH = 70
# RANSAC parameters
RANSAC_TIMES = 200
RANSAC_INLIER_RATIO_THRESH = 0.6


# List of camera data
frame_img_list = np.sort(glob.glob(basedir+'mav0/cam0/data/*.png'))

START_FRAME = 1321 # A good frame to try normal initialization.
#START_FRAME = 465 # Static scene, expect large ave(Z)
#START_FRAME = 893 # From static to move within 5 frames. Expect large ave(Z) then successfully init.
#START_FRAME = 324 # Almost all features on same plane, expect H matrix instead of F/E.
#START_FRAME = 0

#-------------------
# Iterate until successful initialization, OR failure.
#-------------------
for STEP in range(1,5,1):

    print("\n-------------------")
    print("VO init (2D-2D) with STEP = ", STEP)
    print("-------------------")

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
    
    ## Visualization
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
    #pFp = [pts_c[i].dot(F.dot(pts_p[i]))
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
    
    
    #-------------------
    # Ransac 8 pts
    #-------------------
    print("\n\t--------------------------------")
    print("\tStarted RANSAC 8pt algorithm:")
    print("\t--------------------------------")
    
    min_err, min_F, min_RT, min_inliers_list = RANSAC_estimate_RT(pts_c, pts_p, camera, 
                        RANSAC_TIMES, RANSAC_INLIER_RATIO_THRESH)
    
    if len(min_inliers_list) == 0:
        print("WARNING: 8pt RANSAC algorithm failed to find enough inliers!!")
        sys.exit('Initialization failed!')
    
    print("\nAfter 8pt algorithm RANSAC pose estimation:")
    
    pFp = [pts_c[i].dot(min_F.dot(pts_p[i]))
                for i in min_inliers_list]
    print("p'^T F p =", np.abs(pFp).max())
    print("Fundamental Matrix from normalized 8-point algorithm:\n", min_F)
    print("Estimated pose RT:\n", min_RT)
    
    # Visualization
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
    # Sanity check on average depth
    med_Z = np.median(abs(inlier_pts_3D[:,2]))
    ave_Z = np.mean(abs(inlier_pts_3D[:,2]))
    print("Features from Triangulation: median depth(Z):", med_Z, 
            ", average depth(Z):", ave_Z)
    if ave_Z > AVE_Z_THRESH:
        print("Feature average depth(Z) from Triangulation too big, it is:", ave_Z, 
                " threshold = ", AVE_Z_THRESH)
    else:
        break # Successful initialized.

    if STEP==5:
        sys.exit("VO init failed.")

#----------------------
# Point cloud visualization
#----------------------
#import pandas as pd
#from pyntcloud import PyntCloud
#pd_dataframe = pd.DataFrame(inlier_pts_3D)
#pd_dataframe.columns = ['x', 'y', 'z']
#test = PyntCloud(pd_dataframe)
#test.plot(lines=inlier_pts_3D.tolist(), line_color=0xFF00FF)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax_all_pts = fig.add_subplot(122, projection='3d')
reduced_pts = inlier_pts_3D[abs(inlier_pts_3D[:,2])<100]
reduced_pts = reduced_pts[reduced_pts[:,2]>0]
#ax_all_pts.scatter(inlier_pts_3D[:,0], inlier_pts_3D[:,1], inlier_pts_3D[:,2])
ax.scatter(reduced_pts[:,0], reduced_pts[:,1], reduced_pts[:,2])
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.view_init(azim=-90, elev=-55)
plt.savefig("init_pcl.png")
#plt.show()


#-------------------------
# Bundle Adjustment on init point clound
#-------------------------
zero_RT = np.hstack((np.eye(3), np.zeros((3,1))))
M1 = camera.K.dot(zero_RT)
M2 = camera.K.dot(min_RT)

camera_params = np.empty((2,6))
# frame_p
angle_p, direction_p = R2AxisAngle(zero_RT[:,:3])
camera_params[0] = np.concatenate((angle_p*direction_p, zero_RT[:,-1]))
# frame_c
angle_c, direction_c = R2AxisAngle(min_RT[:,:3])
camera_params[1] = np.concatenate((angle_c*direction_c, min_RT[:,-1]))

# Inital value
x0 = np.hstack((camera_params.ravel(), inlier_pts_3D.ravel()))

res = scipy.optimize.least_squares(fun, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf', 
            args=(camera.K, 2, inlier_pts_3D.shape[0], inlier_pts_p[:,:2], inlier_pts_c[:,:2]))

reproj_pts_p = np.matmul(M1, np.hstack((inlier_pts_3D, np.ones((inlier_pts_3D.shape[0],1)))).T).T
reproj_pts_p = reproj_pts_p/reproj_pts_p[:,-1, np.newaxis]
reproj_pts_c = np.matmul(M2, np.hstack((inlier_pts_3D, np.ones((inlier_pts_3D.shape[0],1)))).T).T
reproj_pts_c = reproj_pts_c/reproj_pts_c[:,-1, np.newaxis]
print("pts_p:\n", reproj_pts_p[:3], "\n", inlier_pts_p[:3])
print("pts_c:\n", reproj_pts_c[:3], "\n", inlier_pts_c[:3])


#-----------------------------
# Start 2D-3D PnP procedure
#-----------------------------
print("\n--------------------------------")
print("Start 2D-3D PnP procedure:\n")
print("--------------------------------")

# Use inlier points from 2D-2D pose estimate process above
# Not sure why we must specify it as np.float32, otherwise opencv will assert. See here:
# https://stackoverflow.com/questions/43063320/cv2-calcopticalflowpyrlk-adding-new-points
p0 = inlier_pts_c[:,:-1].reshape(inlier_pts_c.shape[0],1,2).astype(np.float32)

one_col = np.ones_like(inlier_pts_3D[:,0]).reshape(-1,1)
inlier_pts_3D_tmp = np.hstack((inlier_pts_3D, one_col))
pts_3d_last_frame = inlier_pts_3D_tmp.reshape(inlier_pts_3D_tmp.shape[0],1,4)

frame_old = frame1

# Read next frame
for fr in range(START_FRAME+STEP+1, START_FRAME+STEP+8, 1):

    print("\nnext frame ", fr, ":", frame_img_list[fr])
    frame_new_ori = cv2.imread(frame_img_list[fr], cv2.IMREAD_GRAYSCALE)

    # Image undistortion
    frame_new = cv2.remap(frame_new_ori, camera.mapx, camera.mapy, cv2.INTER_LINEAR, cv2.BORDER_TRANSPARENT, 0)

    # Forward OF tracking
    p1, st, err = cv2.calcOpticalFlowPyrLK(frame_old, frame_new, p0, None, **of.lk_params)

    # Successfully tracked points
    good_old = p0[st==1]
    good_new = p1[st==1]
    good_3d_pts = pts_3d_last_frame[st==1]

    # Visualization
    cur_rand_color = (np.random.rand(),0.7, np.random.rand(),1.0)
    # Draw optic flow on last frame (frame1)
    height, width = frame_new.shape
    plt.imshow(frame_new, cmap='gray', extent=[0,width,height,0])
    plt.title("next frame")
    plt.scatter(good_old[:,0], good_old[:,1], linestyle='-', c='g', s=2)
    plt.scatter(good_new[:,0], good_new[:,1], linestyle='-', c='r', s=2)

    for idx in range(good_old.shape[0]):
        plt.plot([good_old[idx,0], good_new[idx,0]],
                        [good_old[idx,1], good_new[idx,1]],
                        linestyle='-', color=cur_rand_color, markersize=0.25)
    plt.savefig("frame"+str(fr)+".png")
    #plt.show()

    # Construct homogeneous coordinates points
    one_col = np.ones_like(good_new[:,0]).reshape(-1,1)
    pts_2d = np.hstack((good_new[:,:2], one_col))
    pts_3d = good_3d_pts

    #-------------------------
    # 2D-3D PnP ransac solver
    #-------------------------
    print("\n\t--------------------------------")
    print("\tStarted 2D-3D PnP algorithm:")
    print("\t--------------------------------")
    
    #R, T = linearPnP(pts_2d, pts_3d)
    #eval_RT_2D_3D(R, T, pts_2d, pts_3d, camera.K)
    min_err_pnp, min_R_pnp, min_T_pnp, min_inliers_list_pnp =  \
            RANSAC_PnP(pts_2d, pts_3d, camera, RANSAC_TIMES=300, INLIER_RATIO_THRESH=0.6)
    print("2D-3D PnP estimated: reproj_err=", min_err_pnp, 
        "R=\n", min_R_pnp, "\nT=", min_T_pnp)
    if min_err_pnp > 3.: # Too big error in reprojection
        sys.exit('VO failed! Exit...')

    # Make current new frame old for next round
    frame_old = frame_new
    p0 = pts_2d[:,:-1].reshape(pts_2d.shape[0],1,2).astype(np.float32)
    pts_3d_last_frame = pts_3d.reshape(pts_3d.shape[0],1,4)

