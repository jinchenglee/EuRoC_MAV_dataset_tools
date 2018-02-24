import numpy as np
import cv2
import yaml

class Camera:
    """
    Camera class contains camera configuration.

    K           - intrinsics
    w, h        - width and height of camera image
    mapx, mapy  - LUT to remap camera image for undistortion.
    """
    def __init__(self, sensor_yaml_file=''):

        # Load cam0 sensor config
        with open(sensor_yaml_file) as fp:
            cam_yaml = yaml.load(fp)
        #print(cam_yaml)
        fp.close()

        # K - intrinsics
        K_params = cam_yaml['intrinsics']
        self.K = np.array([
            [K_params[0], 0., K_params[2]],
            [0., K_params[1], K_params[3]],
            [0., 0., 1.]
        ])
        print("camera intrinsics = \n", self.K)
        
        # D - distortion coefficients
        D_params = cam_yaml['distortion_coefficients']
        D = np.hstack([D_params[:], 0.])
        print("cam0 distortion_coeffs = \n", D)
        
        self.w, self.h = cam_yaml['resolution']
        
        # Generate new camera matrix from parameters
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(self.K, D, (self.w,self.h), 0)
        
        # Generate LUT (look-up tables) for remapping the camera image
        self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.K, D, None,
                                        newcameramatrix, (self.w, self.h), 5)
