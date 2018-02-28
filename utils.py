# Load python packages
import cv2
import numpy as np
import scipy
import scipy.linalg
from scipy.misc import imread
import glob
import yaml
import csv

import matplotlib

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

# ### Utility functions

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''


def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    # Epipolar line in second image: l' = F x
    epi_l_p = np.matmul(F, points1.T).T
    # Draw points onto image2
    fig = plt.figure(figsize=(16, 12))
    plt.subplot(121)
    plt.plot(points2[:, 0], points2[:, 1], 'ro')

    # Draw epipolar lines on image2
    num_lines = epi_l_p.shape[0]
    for i in range(num_lines):
        a, b, c = epi_l_p[i]
        x = np.array([0., im2.shape[1]])
        y = -(a * x + c) / b
        plt.plot(x, y, color='b', linewidth=0.5)
        # Plot calculated corresponding points
        plt.plot(points2[i, 0], -(a * points2[i, 0] + c) / b, 'go')

    # plt.imshow(im2, extent=[0, 512, 512, 0], cmap='gray')
    # plt.imshow(im2, cmap='gray', aspect='auto')
    plt.imshow(im2, cmap='gray', extent=[0, im2.shape[1], im2.shape[0], 0])

    # Epipolar line in first image: l = F^T x'
    epi_l = np.matmul(F.T, points2.T).T
    # Draw points onto image1
    plt.subplot(122)
    plt.plot(points1[:, 0], points1[:, 1], 'ro')

    # Draw epipolar lines on image1
    num_lines = epi_l.shape[0]
    for i in range(num_lines):
        a, b, c = epi_l[i]
        x = np.array([0., im1.shape[1]])
        y = -(a * x + c) / b
        plt.plot(x, y, color='b', linewidth=0.5)
        # Plot calculated corresponding points
        plt.plot(points1[i, 0], -(a * points1[i, 0] + c) / b, 'go')

    # plt.imshow(im1, extent=[0, 512, 512, 0], cmap='gray')
    # plt.imshow(im1, cmap='gray', aspect='auto')
    plt.imshow(im1, cmap='gray', extent=[0, im1.shape[1], im1.shape[0], 0])
    plt.show()


def plot_points_on_images(points1, points2, im1, im2, F):
    # Epipolar line in second image: l' = F x
    epi_l_p = np.matmul(F, points1.T).T
    # Draw points onto image2
    fig = plt.figure(figsize=(16, 12))
    plt.subplot(121)
    plt.plot(points2[:, 0], points2[:, 1], 'ro', markersize=5)

    # Draw epipolar lines on image2
    num_lines = epi_l_p.shape[0]
    for i in range(num_lines):
        a, b, c = epi_l_p[i]
        # Plot calculated corresponding points
        plt.plot(points2[i, 0], -(a * points2[i, 0] + c) / b, 'go', markersize=5)

    # plt.imshow(im2, extent=[0, 512, 512, 0], cmap='gray')
    # plt.imshow(im2, cmap='gray', aspect='auto')
    plt.imshow(im2, cmap='gray', extent=[0, im2.shape[1], im2.shape[0], 0])

    # Epipolar line in first image: l = F^T x'
    epi_l = np.matmul(F.T, points2.T).T
    # Draw points onto image1
    plt.subplot(122)
    plt.plot(points1[:, 0], points1[:, 1], 'ro', markersize=5)

    # Draw epipolar lines on image1
    num_lines = epi_l.shape[0]
    for i in range(num_lines):
        a, b, c = epi_l[i]
        # Plot calculated corresponding points
        plt.plot(points1[i, 0], -(a * points1[i, 0] + c) / b, 'go', markersize=5)

    # plt.imshow(im1, extent=[0, 512, 512, 0], cmap='gray')
    # plt.imshow(im1, cmap='gray', aspect='auto')
    plt.imshow(im1, cmap='gray', extent=[0, im1.shape[1], im1.shape[0], 0])
    plt.show()


'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''


def lls_eight_point_alg(points1, points2):
    x1 = points1[:, 0].reshape(-1, 1)
    y1 = points1[:, 1].reshape(-1, 1)
    x1p = points2[:, 0].reshape(-1, 1)
    y1p = points2[:, 1].reshape(-1, 1)

    # Come up with Af=0
    A = np.hstack([x1p * x1, x1p * y1, x1p, y1p * x1, y1p * y1, y1p, x1, y1, points1[:, 2].reshape(-1, 1)])
    # Using SVD to find least-square solution of f
    U, s, V_transpose = np.linalg.svd(A)
    # f is the last column of V
    f = V_transpose[-1, :]
    F = f.reshape(3, 3) / f[-1]

    # Using SVD again to decompose F
    U1, s1, V1_transpose = np.linalg.svd(F)
    # Force rank 2
    D = np.zeros([3, 3])
    D[0, 0] = s1[0]
    D[1, 1] = s1[1]
    # Get F'
    Fp = np.matmul(U1, np.matmul(D, V1_transpose))
    return Fp


'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''


def normalized_eight_point_alg(points1, points2):
    # Find mean of x,y co-ordinates
    ave_pts1 = np.mean(points1, axis=0)
    ave_pts2 = np.mean(points2, axis=0)
    # Calc mean distance to centroid points
    tmp1 = points1 - ave_pts1
    dist1 = np.mean(np.sqrt((tmp1[:, 0]) ** 2 + (tmp1[:, 1]) ** 2))
    scale1 = 2. / dist1
    tmp2 = points2 - ave_pts2
    dist2 = np.mean(np.sqrt((tmp2[:, 0]) ** 2 + (tmp2[:, 1]) ** 2))
    scale2 = 2. / dist2
    # T matrices
    T1 = np.array([[scale1, 0., 0.], [0., scale1, 0.],
                   [-scale1 * ave_pts1[0], -scale1 * ave_pts1[1], 1.]])
    T2 = np.array([[scale2, 0., 0.], [0., scale2, 0.],
                   [-scale2 * ave_pts2[0], -scale2 * ave_pts2[1], 1.]])
    # normalized points
    norm_pts1 = np.matmul(points1, T1)
    norm_pts2 = np.matmul(points2, T2)

    # Now run the linear least squares eight point algorithm
    F_lls = lls_eight_point_alg(norm_pts1, norm_pts2)

    # Un-normalize
    F = np.matmul(T2, np.matmul(F_lls, T1.T))
    F = F / F[2, 2]

    return F


'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''


def compute_distance_to_epipolar_lines(points1, points2, F):
    # Epipolar line in image 1 from F and points2
    epi_l = np.matmul(F.T, points2.T).T
    # line ax+by+c=0
    a = epi_l[:, 0]
    b = epi_l[:, 1]
    c = epi_l[:, 2]
    # Points to be calculated distance
    x0 = points1[:, 0]
    y0 = points1[:, 1]
    # Distance calculated
    d = np.abs(a * x0 + b * y0 + c) / np.sqrt(a ** 2 + b ** 2)

    return np.mean(d)


'''
ESTIMATE_INITIAL_RT from the Essential Matrix, we can compute 4 initial
guesses of the relative RT between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
Returns:
    RT: A 4x3x4 tensor in which the 3x4 matrix RT[i,:,:] is one of the
        four possible transformations
'''


def estimate_initial_RT(E):
    U, s, V_trans = np.linalg.svd(E)
    # Helper matrices
    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    # Calculate R
    R1 = U.dot(W.dot(V_trans))
    R2 = U.dot(W.T.dot(V_trans))
    R1_sign = np.linalg.det(R1)
    R2_sign = np.linalg.det(R2)
    R1 = -R1 if R1_sign < 0 else R1
    R2 = -R2 if R2_sign < 0 else R2
    # T - Here, we actually "normalized" T to be unit vector,
    #   which determines the global scale in monocular VO/SLAM.
    T1 = U[:, 2]
    T2 = -U[:, 2]
    # RT
    RT = np.array([
        np.vstack([R1.T, T1]).T,
        np.vstack([R1.T, T2]).T,
        np.vstack([R2.T, T1]).T,
        np.vstack([R2.T, T2]).T
    ])

    return RT


'''
LINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point is the best linear estimate
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''


def linear_estimate_3d_point(image_points, camera_matrices):
    pts = image_points.copy()
    M = camera_matrices.copy()

    # Construct the 2m x 4 matrix A
    # x coord
    A1 = (M[:, 2, :].T * pts[:, 0]).T - M[:, 0, :]
    # y coord
    A2 = (M[:, 2, :].T * pts[:, 1]).T - M[:, 1, :]

    # Debug
    # print M[:,2,:].shape, pts[:,0].shape

    A = np.vstack([A1, A2])
    U, s, V_trans = np.linalg.svd(A)
    point_3d = V_trans[3, :].copy()
    # print "linear point_3d = ", point_3d
    point_3d /= point_3d[-1]
    # print "linear point_3d (normed) = ", point_3d
    return point_3d[:-1]


'''
REPROJECTION_ERROR given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image (NxMx3?)
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    error - the 2Mx1 reprojection error vector
'''


def reprojection_error(point_3d, image_points, camera_matrices):
    M = image_points.shape[0]
    # Reprojecting points to image
    proj_img_pts = np.matmul(camera_matrices, np.hstack([point_3d, 1.]))
    # Normalize p' = [y1, y2]^T/y3
    proj_img_pts = proj_img_pts.T
    proj_img_pts /= proj_img_pts[-1, :]
    # Re-projection error
    reproj_error = proj_img_pts[:-1, :].T - image_points
    # Make it 2Mx1 vector
    reproj_error = reproj_error.reshape(2 * M, )
    return reproj_error


'''
JACOBIAN given a 3D point and its corresponding points in the image
planes, compute the reprojection error vector and associated Jacobian
Arguments:
    point_3d - the 3D point corresponding to points in the image
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    jacobian - the 2Mx3 Jacobian matrix
'''


def jacobian(point_3d, camera_matrices):
    # Write down element one by one, in vector form

    # homogeneous coordi of 3D point(s) X
    point_3d = np.hstack([point_3d, 1.])

    # derivative of -e + (cx+d)/(ax+b) is (bc-ad)/(ax+b)^2
    # Common denominator
    denom = (np.matmul(camera_matrices[:, 2, :], point_3d)) ** 2

    # Elements of row dx
    d_e_p11 = camera_matrices[:, 0, 0] * np.matmul(camera_matrices[:, 2, [1, 2, 3]],
                                                   point_3d[[1, 2, 3]]) - camera_matrices[:, 2, 0] * np.matmul(
        camera_matrices[:, 0, [1, 2, 3]], point_3d[[1, 2, 3]])
    d_e_p12 = camera_matrices[:, 0, 1] * np.matmul(camera_matrices[:, 2, [0, 2, 3]],
                                                   point_3d[[0, 2, 3]]) - camera_matrices[:, 2, 1] * np.matmul(
        camera_matrices[:, 0, [0, 2, 3]], point_3d[[0, 2, 3]])
    d_e_p13 = camera_matrices[:, 0, 2] * np.matmul(camera_matrices[:, 2, [0, 1, 3]],
                                                   point_3d[[0, 1, 3]]) - camera_matrices[:, 2, 2] * np.matmul(
        camera_matrices[:, 0, [0, 1, 3]], point_3d[[0, 1, 3]])

    # Elements of row dy
    d_e_p21 = camera_matrices[:, 1, 0] * np.matmul(camera_matrices[:, 2, [1, 2, 3]],
                                                   point_3d[[1, 2, 3]]) - camera_matrices[:, 2, 0] * np.matmul(
        camera_matrices[:, 1, [1, 2, 3]], point_3d[[1, 2, 3]])
    d_e_p22 = camera_matrices[:, 1, 1] * np.matmul(camera_matrices[:, 2, [0, 2, 3]],
                                                   point_3d[[0, 2, 3]]) - camera_matrices[:, 2, 1] * np.matmul(
        camera_matrices[:, 1, [0, 2, 3]], point_3d[[0, 2, 3]])
    d_e_p23 = camera_matrices[:, 1, 2] * np.matmul(camera_matrices[:, 2, [0, 1, 3]],
                                                   point_3d[[0, 1, 3]]) - camera_matrices[:, 2, 2] * np.matmul(
        camera_matrices[:, 1, [0, 1, 3]], point_3d[[0, 1, 3]])

    # Divide the denominator
    J1 = np.vstack([[d_e_p11], [d_e_p12], [d_e_p13]])
    J2 = np.vstack([[d_e_p21], [d_e_p22], [d_e_p23]])
    J1 = np.divide(J1, denom).T
    J2 = np.divide(J2, denom).T

    # Interleaving the result to be jacobian matrix
    J = np.zeros((2 * J1.shape[0], J1.shape[1]))
    J[0::2, :] = J1
    J[1::2, :] = J2

    return J


'''
NONLINEAR_ESTIMATE_3D_POINT given a corresponding points in different images,
compute the 3D point that iteratively updates the points
Arguments:
    image_points - the measured points in each of the M images (Mx2 matrix)
    camera_matrices - the camera projective matrices (Mx3x4 tensor)
Returns:
    point_3d - the 3D point
'''


def nonlinear_estimate_3d_point(image_points, camera_matrices):
    pts_im = image_points.copy()
    cam = camera_matrices.copy()

    # Number of iterations
    num_iter = 10

    pts_3d_linear = linear_estimate_3d_point(pts_im, cam)
    pts_3d_nxt = pts_3d_linear.copy()

    for i in range(num_iter):
        J = jacobian(pts_3d_nxt, cam)
        err = reprojection_error(pts_3d_nxt, pts_im, cam)
        # Debug
        #print("Iter", i, " nonlinear 3d_point = ", pts_3d_nxt, "reproj_error = ", err)
        if np.mean(abs(err))<0.3:
            break
        JTdotJ = J.T.dot(J)
        if np.linalg.det(JTdotJ) > 1e-5:
            delta_pts_3d = -np.matmul(np.matmul(np.linalg.inv(JTdotJ), J.T), err)
            pts_3d_nxt += delta_pts_3d
            #print("delta_3d_point = ", delta_pts_3d, "\n")
        else:
            print("Singular JTdotJ met. Nonlinear optimization diverged(?).")
            print("img points: ", pts_im, "\nCam: ", cam)
            print("Iter", i, " nonlinear 3d_point = ", pts_3d_nxt, " linear 3d_point = ", 
                pts_3d_linear, " reproj_error = ", err)
            # Return linear solution instead
            return pts_3d_linear
            break

    return pts_3d_nxt


'''
ESTIMATE_RT_FROM_E from the Essential Matrix, we can compute  the relative RT 
between the two cameras
Arguments:
    E - the Essential Matrix between the two cameras
    img_pts - N measured points in each of the M images (NxMx2 matrix)
    K - the intrinsic camera matrix
Returns:
    RT: The 3x4 matrix which gives the rotation and translation between the 
        two cameras
'''


def estimate_RT_from_E(E, img_pts, K):
    # Estimate the 4 candidate R,T pairs
    init_RT = estimate_initial_RT(E)

    # print("img_pts.shape=", img_pts.shape)

    # Voting counter
    cnt = [0, 0, 0, 0]

    # Projective matrix M1
    M1 = K.dot(np.append(np.eye(3), np.zeros((3, 1)), axis=1))

    # Go through each points
    for i in range(img_pts.shape[0]):
        for j in range(init_RT.shape[0]):
            # Projective matrix M2 candidate i
            M2 = K.dot(init_RT[j])

            M = np.array((M1, M2))

            # Calculate the 3D point

            # Using non-linear method will meet 'singular matrix' error, meaning the 
            # nonlinear optimization has diverged. This most likely is because init_RT
            # estimate provided is not good at all.
            #
            # We should NOT use nonlinear method here for 3d point estimate, as we are
            # not after accuracy, but just ruling out apparent wrong solutions from the
            # 4 provided ones. Linear method is GOOD ENOUGH for this purpose. 
            #
            #X = nonlinear_estimate_3d_point(img_pts[i], M)

            # Linear method instead
            X = linear_estimate_3d_point(img_pts[i], M)

            # print "3d point candidate ", j, " in cam1: ", X
            # Move X to camera2 coordinate system
            X2 = init_RT[j].dot(np.append(X, 1.).T)
            # print "3d point candidate ", j, " in cam2: ", X2.T
            if X2[2] > 0.0 and X[2] > 0.0:
                # print("img_pts[i]=", img_pts[i], ",X=", X, ",X2=", X2)
                cnt[j] += 1

    index = np.argmax(cnt)
    # print("cnt = ", cnt, "index = ", index)
    RT = init_RT[index]

    return RT


def eval_RT(RT, img_pts, K):
    # print("img_pts.shape=", img_pts.shape)

    # Projective matrix M1
    M1 = K.dot(np.append(np.eye(3), np.zeros((3, 1)), axis=1))

    # Projective matrix M2
    M2 = K.dot(RT)

    M = np.array((M1, M2))

    error = 0.

    for i in range(img_pts.shape[0]):
        # Calculate the 3D point i
        X = linear_estimate_3d_point(img_pts[i], M)
        # print("3d point in cam1: ", X)

        err = reprojection_error(X, img_pts[i], M)
        err = np.mean(abs(err))

        error += err

    error /= img_pts.shape[0]

    return error


def eval_RT_thresh(RT, img_pts, K):
    # print("img_pts.shape=", img_pts.shape)

    # Projective matrix M1
    M1 = K.dot(np.append(np.eye(3), np.zeros((3, 1)), axis=1))

    # Projective matrix M2
    M2 = K.dot(RT)

    M = np.array((M1, M2))

    error = 0.
    SKIP_THRESH = 2  # distance threshold - notice there is no absolute scale here!

    # inliers list
    inliers_cnt = 0
    inliers_list = []

    for i in range(img_pts.shape[0]):
        # Calculate the 3D point i
        X = linear_estimate_3d_point(img_pts[i], M)
        #print("Point", i, ", 3d point in cam1: ", X)

        err = reprojection_error(X, img_pts[i], M)
        #print(err)
        #err_dist = np.sqrt(err[0]**2 + err[1]**2) + np.sqrt(err[1]**2 + err[2]**2)
        err_dist = np.mean(abs(err))
        #print("Reprojection error = ", err_dist)

        if err_dist < SKIP_THRESH:
            #print("Inlier added:", i, err_dist)
            error += err_dist
            inliers_cnt += 1
            inliers_list.append(i)

    if inliers_cnt > 0:
        error /= inliers_cnt
    else:
        error = 1e+10

    return error, inliers_cnt, inliers_list



def ALLPOINTS_estimate_RT(pts_c, pts_p, camera):
    """
    Using all point correspondences directly from optical flow in
    current and previous frames to estimate best Fundamental Matrix F.
    Essential Matrix E is recovered from F, and R, T estimated from E.

    pts_c and pts_p are the feature correspondences in current and pre-
    ious frames. 

    Reprojection error is expected to be big. This function is used for
    comparing with RANSAC_estimate_RT() function only.
    """

    # matched image points
    img_pts_all = np.zeros([2*pts_c.shape[0], pts_c.shape[1]-1])
    img_pts_all[::2,:] = pts_c[:,:-1]
    img_pts_all[1::2,:]= pts_p[:,:-1]
    img_pts_all = img_pts_all.reshape(-1,2,2)

    min_err = 1e8
    min_RT = np.empty((3,4))
    min_inliers_list = []

    # Calculate fundamental matrix F
    F = normalized_eight_point_alg(pts_c, pts_p)

    # Essential matrix
    E = camera.K.T.dot(F).dot(camera.K)
    #
    # Estimate RT matrix from E
    RT = estimate_RT_from_E(E, img_pts_all, camera.K)

    # Reproj erorr
    err, inliers_cnt, _ = eval_RT_thresh(RT, img_pts_all, camera.K)
    print("Mean reproj error: ", err, "inliers/total:", inliers_cnt, "/", img_pts_all.shape[0], "\n")

    return err, F, RT



def RANSAC_estimate_RT(pts_c, pts_p, camera, 
                    RANSAC_TIMES=100, INLIER_RATIO_THRESH=0.8):
    """
    Using RANSAC algorithm to randomly pick 8 point correspondences in
    current and previous frames to estimate best Fundamental Matrix F.
    Essential Matrix E is recovered from F, and R, T estimated from E.

    pts_c and pts_p are the feature correspondences in current and pre-
    ious frames. Reprojection error is used to evaluate the candidate
    [F, E, R, T] solutions.
    """

    # matched image points
    img_pts_all = np.zeros([2*pts_c.shape[0], pts_c.shape[1]-1])
    img_pts_all[::2,:] = pts_c[:,:-1]
    img_pts_all[1::2,:]= pts_p[:,:-1]
    img_pts_all = img_pts_all.reshape(-1,2,2)

    min_err = 1e8
    min_inliers_list = []
    min_F = np.empty((3,3))
    min_RT = np.empty((3,4))
    # Use all points to estimate initial value
    min_err, min_F, min_RT = ALLPOINTS_estimate_RT(pts_c, pts_p, camera)

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

    return min_err, min_F, min_RT, min_inliers_list


def triangulate(inlier_pts_c, inlier_pts_p, camera, min_RT):
    """
    Triangulate from image correspondences to get 3D coordindates. 
    """
    inlier_img_pts_all = np.zeros([2*inlier_pts_c.shape[0], inlier_pts_c.shape[1]-1])
    inlier_img_pts_all[::2,:] = inlier_pts_c[:,:-1]
    inlier_img_pts_all[1::2,:]= inlier_pts_p[:,:-1]
    inlier_img_pts_all = inlier_img_pts_all.reshape(-1,2,2)

    # Projective matrix M1
    M1 = camera.K.dot(np.append(np.eye(3), np.zeros((3, 1)), axis=1))
    # Projective matrix M2 candidate i
    M2 = camera.K.dot(min_RT)
    # Combine to matrix M
    M = np.array((M1, M2))

    # Output points array
    inlier_pts_3D = np.empty((inlier_img_pts_all.shape[0], 3))
    # Triangulate each point
    for idx in range(inlier_img_pts_all.shape[0]):
        inlier_pts_3D[idx] = nonlinear_estimate_3d_point(inlier_img_pts_all[idx], M).reshape(-1,3)
        #inlier_pts_3D[idx] = linear_estimate_3d_point(inlier_img_pts_all[idx], M).reshape(-1,3)

    return inlier_pts_3D
