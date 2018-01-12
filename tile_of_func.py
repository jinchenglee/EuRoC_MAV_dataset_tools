import cv2
import numpy as np
import scipy 
import scipy.linalg 
import pykitti
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Local scripts
import tile_of_ransac as r

#-------------
# Optical Flow related setting
#-------------
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.01,
                       minDistance = 3,
                       blockSize = 3)
 
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
 ## OF feature points filter threshold in pixels
OF_FWD_BWD_CONSISTENCY_THRESH = 150 # Set this large as we want to maintain change as much as possible
OF_MIN_OF_DISTANCE_THRESH = 0.5


#-------------
# Utility functions
#-------------
def dist(a,b):   
    '''
    Return Euclidean distance between two sets of points. 
    '''
    assert a.ndim==2
    assert b.ndim==2
    (a1,a2) = a.shape
    (b1,b2) = b.shape
    assert a1==b1
    assert a2==b2
    return np.sqrt((a[:,0]-b[:,0])**2+(a[:,1]-b[:,1])**2)

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''

    # Code to get unique color using this
    ## Generate a color map
    #cmap = get_cmap(len(range(int((cam0_height/tile_size)*(cam0_width/tile_size)))))
    ## Pick current unique color according to i,j
    #cur_color = cmap(int((i/tile_size)*(cam0_height/tile_size)
    #    +(j/tile_size)))
    #print("cur_color = ", cur_color)

    return plt.cm.get_cmap(name, n)

#-------------
# Workhorse processing functions
#-------------

def lidar_proc(cam0, velo, T_v2c0, K_c0, P_v2c0):
    '''
    Processing lidar points and annotate onto cam0 image. 

    Input parameters:
        - cam0:     camera image data 
        - velo:     lidar data, (x,y,z,reflective_strength)
        - T_v2c0:   velodyne to cam0 matrix, shape = 4x4 = (R | t)
        - K_c0:     cam0 intrinsics
        - P_v2c0:   Shape 3x4 = matmul(3x4, 4x4) = K_c0 * T_v2c0
    Reture: None
    '''
    # Reduce number of points - pick one every 5 rows
    velo = velo[::5,:]
    
    # Rule out those invisible points
    velo_visible = []
    for i in range(velo.shape[0]):
        if velo[i,0]>5:
            velo_visible.append(velo[i,:])
    velo_v = np.array(velo_visible)
    
    # Project velo data into image (cam0) coordinates
    tmp = np.matmul(P_v2c0, np.c_[velo_v[:,0:3],np.ones(velo_v.shape[0]).transpose()].transpose()) 
    tmp = tmp.transpose()
    # Normalize to (x,y) coordinates within image
    tmp2 = tmp[:,:-1]/tmp[:,2].reshape((tmp.shape[0],1))
    
    # Grab depth info. 
    #   BUG WAS: velo_v[:,3] - reflective strength, velo_v[:,0] - x depth
    #   Velodyne's coordinates are different from camera
    depth = velo_v[:,0]
    
    # Show the gray img from camera 0
    #plt.imshow(cam0, cmap='gray', interpolation='nearest', aspect='auto')
    plt.imshow(cam0, cmap='gray', interpolation='nearest')
    
    visible_dots = []
    # Draw dots from lidar onto this image
    for i in range(depth.shape[0]):
        # Filter out those dots that are not in FOV
        if tmp2[i][0] >0 and tmp2[i][0]<cam0.shape[1] and tmp2[i][1]>0 and tmp2[i][1]<cam0.shape[0]:
            visible_dots.append([tmp2[i][0], tmp2[i][1], depth[i]])
    
    visible_dots = np.reshape(visible_dots, (len(visible_dots),3))
    
    # Draw the lidar depth onto cam0 image with color mapping 
    #plt.scatter(x=visible_dots[:,0], y=visible_dots[:,1], c=visible_dots[:,2], cmap='viridis_r', marker='+')



def OF_TileFindFeature(cam0, tile_size, feature_params):
    '''
    Split image into tile_size to find features

    Input parameters:
        - cam0:     camera image data 
        - tile_size:Size of tile to split image
        - feature_params: config for cv2.goodFeaturesToTrack()
    Reture: cv2.goodFeaturesToTrack() returned n features list, strange [n,1,2] dim.

    '''
    (cam0_height, cam0_width) = cam0.shape

    corners=[]
    for i in range(0,cam0_height-1,tile_size):
        for j in range(0,cam0_width-1,tile_size):
            #print(i,j)
            cam0_tile_size = cam0[i:i+tile_size-1,j:j+tile_size-1]

            # ----------------
            # Shi-Tomasi Corner
            # ----------------
            
            new_corners = cv2.goodFeaturesToTrack(cam0_tile_size, **feature_params)
            if new_corners is not(None):
                print("Tile (y,x):", i,j)
                # Reverse [i,j] in (x,y) coordinates recording !!!
                new_corners = new_corners + [[j,i]]
                #DEBUG print("new corners", new_corners.shape, new_corners)
                print("New corners detected:", new_corners.shape[0])
                if i==0 and j==0:
                    corners = new_corners
                    #DEBUG print("i = ", i, " j = ", j, " corners", corners.shape, corners)
                else:
                    corners = np.vstack((corners, new_corners))
                    #DEBUG print("i = ", i, " j = ", j, " corners", corners.shape, corners)
                print("Total corners", corners.shape)
            else:
                print("Tile (y,x):", i,j)
                print("No new corner detected.")

    # OpenCV takes CV_32F fp data, while new_corners is 64bit.
    return np.array(corners, dtype=np.float32)



def OF_FeatureListFilter(p0, p1, p2, cam0_width, cam0_height):
    '''
    Split image into tile_size to find features

    Input parameters:
        - feature_list: Merged candidate raw features (TODO: cleanup feature_list data structure)
        - p0,p1,p2: 
            # Merge data into feature_list. Every row has 4 columns:
            #  orig_x orig_y  fwd_x fwd_y  
            #  p0_x   p0_y    p1_x  p1_y   
        - feature_params: config for cv2.goodFeaturesToTrack()
    Reture: cv2.goodFeaturesToTrack() returned n features list, strange [n,1,2] dim.

    '''
    # Merge data into feature_list. 
    feature_list = []
    feature_list = np.hstack((p0.reshape(-1,2),p1.reshape(-1,2)))


    # 1) Remove distance(p0,p2) > OF_FWD_BWD_CONSISTENCY_THRESH pixel
    distance = dist(p0.reshape(-1,2),p2.reshape(-1,2))
    tmp = distance<OF_FWD_BWD_CONSISTENCY_THRESH
    p0 = p0[tmp]
    p1 = p1[tmp]
    p2 = p2[tmp]
    feature_list=feature_list[tmp]
    print("Clean up step 1 (fwd bwd consistency): \tfeature_list.shape:", feature_list.shape)
    # 2) Remove too small OF (static points) distance(p0,p1) < OF_MIN_OF_DISTANCE_THRESH pixels
    distance = dist(p0.reshape(-1,2),p1.reshape(-1,2))
    tmp = distance>OF_MIN_OF_DISTANCE_THRESH
    p0 = p0[tmp]
    p1 = p1[tmp]
    p2 = p2[tmp]
    feature_list=feature_list[tmp]
    print("Clean up step 2 (mininum distance): \tfeature_list.shape:", feature_list.shape)
    # 3) Remove points out of frame boundary 
    tmp = np.logical_or(p1[:,:,0]<0, p1[:,:,0]>cam0_width)
    tmp = np.logical_or(tmp, p1[:,:,1]<0)
    tmp = np.logical_or(tmp, p1[:,:,1]>cam0_height)
    # Inverse selection
    tmp = np.logical_not(tmp)
    tmp = tmp.reshape(tmp.shape[0],)
    p0 = p0[tmp]
    p1 = p1[tmp]
    p2 = p2[tmp]
    feature_list=feature_list[tmp]
    print("Clean up step 3 (img boundary check): \tfeature_list.shape:", feature_list.shape)

    ### Write feature list into csv file
    ##with open('file'+str(fr).zfill(3)+'.csv', 'wb') as f:
    ##    np.savetxt(f, feature_list, delimiter=",")

    return feature_list


def OF_TileAffineSolver(cam0, tile_size, feature_list):
    '''
    Scan over each tile, find affine parameter for that tile.

    Feature_list contains the mapped optical flow src and destination points, 
    which consists formula as below. 

    #   - Assuming only scaling and translation for now
    #
    #    | x 0 1 0 | | sx |    | x' |
    #    |         |*| sy |  = |    |
    #    | 0 y 0 1 | | tx |    | y' |
    #                | tx |          
    #
    #   - Over-determined system, using least-square solver
    #
    '''
    (cam0_height, cam0_width) = cam0.shape

    # Draw grid over image
    for i in range(0,cam0_height,tile_size):
        plt.plot([0, cam0_width], [i, i], linestyle='-', color='g')
    for i in range(0,cam0_width,tile_size):
        plt.plot([i,i], [0, cam0_height], linestyle='-', color='g')

    print("feature_list.shape: ", feature_list.shape)
    for i in range(0,cam0_height,tile_size):
        for j in range(0,cam0_width,tile_size):

            # Select src points inside current tile
            tmp = [] # Reset points of current tile
            tmp = np.logical_and(feature_list[:,0]<j+tile_size-1, feature_list[:,0]>j)
            tmp = np.logical_and(tmp, feature_list[:,1]<i+tile_size-1)
            tmp = np.logical_and(tmp, feature_list[:,1]>i)
            tmp = tmp.reshape(tmp.shape[0],)
            #DEBUG print("Shape tmp:", tmp.shape)
            #DEBUG print("tmp =", tmp)
 
            # Prepare Bx=C
            A = feature_list[tmp]
            seleted_feature_list_idx = [i for i, x in enumerate(tmp) if x]
            if A.shape[0]>2:    # Skip empty tile or less than 3 points for ransac
                print("TileAffineSolver at tile: (", i,j, ")")
                print("Shape A:", A.shape)
                B = np.zeros_like(A)
                B = np.vstack((B,B))
                C = np.zeros_like(B[:,0])
                print("Shape B, C for Bx=C:", B.shape, C.shape)
                # Prepare data for B*x=C
                for k in range(A.shape[0]):
                    B[2*k][0]=A[k][0]
                    B[2*k][2]=1
                    B[2*k+1][1]=A[k][1]
                    B[2*k+1][3]=1
                    C[2*k]=A[k][2]
                    C[2*k+1]=A[k][3]

                # Run ransac
                all_data = np.hstack( (B,C.reshape(-1,1)) )
                n_inputs = B.shape[1]
                n_outputs = 1
                input_columns = range(n_inputs) # the first columns of the array
                output_columns = [n_inputs+i for i in range(n_outputs)] # the last columns of the array
                debug = False
                # Least-square solver - One outlier will ruin least-square
                model = r.LinearLeastSquaresModel(input_columns,output_columns,debug=debug)
                # run RANSAC algorithm
                #   n - the minimum number of data values required to fit the model
                #   k - the maximum number of iterations allowed in the algorithm
                #   t - a threshold value for determining when a data point fits a model
                #   d - the number of close data values required to assert that a model fits well to data
                ransac_fit, ransac_min_err, ransac_data = r.ransac(all_data,model,
                                     10, 500, 10, 5, # misc. parameters
                                     debug=debug,return_all=True)
                inliers_idx = ransac_data['inliers']
                num_inliers = len(inliers_idx)
                inlier_ratio = num_inliers/B.shape[0]
                if inlier_ratio > 0.3:
                    print("Ransac inliers_idx of B = ", inliers_idx)
                    print(ransac_fit, "min_err = ", ransac_min_err, "inliers = ", 
                            num_inliers, " ", 100*inlier_ratio, "%")

                #-----------------------
                # Visualization
                #-----------------------
                #
                # Pick a rand color for current tile
                cur_rand_color = (np.random.rand(),0.7,
                                np.random.rand(),1.0)

                for idx in inliers_idx:
                    cur_inlier = feature_list[seleted_feature_list_idx[int(idx/2)]]
                    #DEBUG print("feature:", cur_inlier)
                    #DEBUG print("B:", B[idx])

                # 1. Draw ransac-inlier feature points and associated OF of each tile
                    plt.plot([cur_inlier[0], cur_inlier[2]], 
                             [cur_inlier[1], cur_inlier[3]],
                             linestyle='-', color=cur_rand_color)

#                # 2. Draw rectangle from a (5,5)-sized square located at center 
#                # of each tile. Scaled by (sx, sy) and offset by (tx, ty)
#                tile_center_x = j + tile_size/2
#                tile_center_y = i + tile_size/2
#                square_width = 15
#                # 4 points of the square
#                D = np.zeros((4,2))
#                D[0] = [tile_center_x, tile_center_y]
#                D[1] = [tile_center_x+square_width, tile_center_y]
#                D[2] = [tile_center_x, tile_center_y+square_width]
#                D[3] = [tile_center_x+square_width, tile_center_y+square_width]
#                E = np.zeros((4*2,4))
#                # Prepare data for E*x=F
#                for k in range(4):
#                    E[2*k][0]=D[k][0]
#                    E[2*k][2]=1
#                    E[2*k+1][1]=D[k][1]
#                    E[2*k+1][3]=1
#                # 4 points of the affine-transformed rectangle
#                F = np.matmul(E,ransac_fit)
#                # Draw a thick line between squares
#                plt.plot([D[0][0], F[0]], 
#                         [D[0][1], F[1]],
#                         linestyle='-', linewidth=3, color=cur_rand_color)
#                # Draw the two squares
#                # - Original square first
#                plt.plot([D[0][0], D[1][0]], 
#                         [D[0][1], D[1][1]],
#                         linestyle='-', color=cur_rand_color)
#                plt.plot([D[0][0], D[2][0]], 
#                         [D[0][1], D[2][1]],
#                         linestyle='-', color=cur_rand_color)
#                plt.plot([D[2][0], D[3][0]], 
#                         [D[2][1], D[3][1]],
#                         linestyle='-', color=cur_rand_color)
#                plt.plot([D[3][0], D[1][0]], 
#                         [D[3][1], D[1][1]],
#                         linestyle='-', color=cur_rand_color)
#                # - Affine transformed square first
#                plt.plot([F[0], F[2]], 
#                         [F[1], F[3]],
#                         linestyle='-', color=cur_rand_color)
#                plt.plot([F[0], F[4]], 
#                         [F[1], F[5]],
#                         linestyle='-', color=cur_rand_color)
#                plt.plot([F[2], F[6]], 
#                         [F[3], F[7]],
#                         linestyle='-', color=cur_rand_color)
#                plt.plot([F[4], F[6]], 
#                         [F[5], F[7]],
#                         linestyle='-', color=cur_rand_color)
#


