from argparse import ArgumentParser
from ast import arg
import cv2
import os
import numpy as np
import pandas as pd

LEFT = 0
RIGHT = 1

def load_image_paths(path):
    left_paths, right_paths = [], []
    l_path, r_path = os.path.join(path, 'left'), os.path.join(path, 'right')

    for idx, f in enumerate(os.listdir(l_path)):
        l_file = os.path.join(l_path, f)
        r_file = os.path.join(r_path, f)

        # check file existance on the right folder
        if not os.path.exists(r_file):
            print("%s exists in the left folder but not in the right folder!" % (f))

        left_paths.append(l_file)
        right_paths.append(r_file)

    return left_paths, right_paths


def get_obj_pnts(row, col):
    # obj_pnts = np.zeros((row*col,3), np.float32)
    # obj_pnts[:,:2] = np.mgrid[0:row,0:col].T.reshape(-1,2)
    obj_pnts = np.zeros((row * col, 3), np.float32)

    x, y = 0, 0
    for i in range(col):
        x = i
        for j in range(row):
            y = j
            pnt = np.array([x, y, 0])
            obj_pnts[x * row + y] = pnt

    return obj_pnts

def get_img_pnts(img, rows, cols, winSize):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# step 1, apply chessboard corner detection
    ret, corners = cv2.findChessboardCorners(gray_img, (rows, cols), None)
    if ret is False:
        return None, img

# step 2, apply refinement to the output from step 1
    # (criteria type, max num iterations, epsilon)
    subpix_refine_term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.001)
    refine_corner = cv2.cornerSubPix(gray_img, corners, winSize, (-1, -1), subpix_refine_term_criteria) 

# step 3, show the chessboard corners to user for sanity check
    image_with_corners = np.copy(img)
    cv2.drawChessboardCorners(image_with_corners, (rows, cols), refine_corner, ret)
    cv2.putText(image_with_corners, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv2.FONT_HERSHEY_COMPLEX, 0.52, (33,100,255), 1)

    return refine_corner, image_with_corners

def do_stereo_calibration(args):
    l_path, r_path = load_image_paths(args.img_folder)
    print("%d images loaded for each camera" % (len(l_path)))

    calib_dict = {"cmtx": [], "dist": [], "img_p": [], "R": [], "T": []}
    sample = cv2.imread(l_path[0])
    width = sample.shape[1]
    height = sample.shape[0]

    print("calibrate left camera")
    cmtx1, dist1, img_p1, obj_p = single_camera_calibration(args, l_path)
    calib_dict["cmtx"].append(cmtx1)
    calib_dict["dist"].append(dist1)
    calib_dict["img_p"].append(img_p1)

    print("calibrate right camera")
    cmtx2, dist2, img_p2, _ = single_camera_calibration(args, r_path)
    calib_dict["cmtx"].append(cmtx2)
    calib_dict["dist"].append(dist2)
    calib_dict["img_p"].append(img_p2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv2.stereoCalibrate(obj_p, calib_dict["img_p"][LEFT], calib_dict["img_p"][RIGHT], calib_dict["cmtx"][LEFT], calib_dict["dist"][LEFT],
                                                                 calib_dict["cmtx"][RIGHT], calib_dict["dist"][RIGHT], (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    print('rotaion:\n', R)
    print('translation:\n', T)

    calib_dict["R"].append(np.eye(3, dtype=np.float32))
    calib_dict["T"].append(np.array([[0], [0], [0]]))
    calib_dict["R"].append(R)
    calib_dict["T"].append(T)

    return calib_dict

def single_camera_calibration(args, path):
    img_size = cv2.imread(path[0]).shape
    print("image size:", img_size)

    img_p, obj_p = [], []
    for idx, l_p in enumerate(path):
        obj_pnts = get_obj_pnts(args.row, args.col) * args.tile_size
        l_im = cv2.imread(l_p)
        img_pnts, img_with_img_pnts = get_img_pnts(l_im, args.row, args.col, (5, 5))

        cv2.imshow('corners', img_with_img_pnts)
        if cv2.waitKey(0) == 115: # press 's'
            print('skipped image sample')
            
        if img_pnts is None:
            print("find corner failed for this image")

        img_p.append(img_pnts)
        obj_p.append(obj_pnts)

    cv2.destroyAllWindows()
    print("obj points shape:", obj_p[0].shape, "img points shape:", img_p[0].shape)
    ret, cmtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_p, img_p, (img_size[1], img_size[0]), None, None)

    print("\n#####################################################################\nRMS re-projection error: %.8f\n" % (ret))
    print("intrinsic camera matrix:\n", cmtx)
    print('#####################################################################\n')

    return cmtx, dist, img_p, obj_p

if __name__ == "__main__":
    parser = ArgumentParser(description="stereo calibration")

    parser.add_argument('--img_folder', type=str, required=True, dest='img_folder')
    parser.add_argument('-r', type=int, required=True, dest='row')
    parser.add_argument('-c', type=int, required=True, dest='col')
    parser.add_argument('--tile_size', type=float, required=True, dest='tile_size') # in cm
    parser.add_argument('--save_dir', type=str, required=True, dest='save_dir')
    parser.add_argument('--camera_name', type=str, required=True, dest='camera_name')
    parser.add_argument('--npy_file', type=str, required=False, dest='npy_file', default="")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        print("creating folder", args.save_dir)
        os.makedirs(args.save_dir)

    i = 0
    while os.path.exists(os.path.join(args.save_dir, args.camera_name + '_' + str(i) + '.npy')):
        i += 1
    
    args.npy_file = os.path.join(args.save_dir, args.camera_name + '_' + str(i) + '.npy')
    print("will save calibration results to:", args.npy_file)

    result_dict = do_stereo_calibration(args)
    result_dict.pop("img_p")

    print("writing calibration results ...")
    np.save(args.npy_file, result_dict)

    # test load
    back = np.load(args.npy_file, allow_pickle=True)
    for k in back.item().keys():
        print(k)
        print(back.item().get(k))
    