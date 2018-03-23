import cv2
import numpy as np
import scipy 
import scipy.linalg 
#import pykitti
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
                #print("Tile (y,x):", i,j)
                # Reverse [i,j] in (x,y) coordinates recording !!!
                new_corners = new_corners + [[j,i]]
                #DEBUG print("new corners", new_corners.shape, new_corners)
                #print("New corners detected:", new_corners.shape[0])
                if i==0 and j==0:
                    corners = new_corners
                    #DEBUG print("i = ", i, " j = ", j, " corners", corners.shape, corners)
                else:
                    corners = np.vstack((corners, new_corners))
                    #DEBUG print("i = ", i, " j = ", j, " corners", corners.shape, corners)
                #print("Total corners", corners.shape)
            else:
                print("Tile (y,x):", i,j)
                print("No new corner detected.")

    # OpenCV takes CV_32F fp data, while new_corners is 64bit.
    return np.array(corners, dtype=np.float32)



