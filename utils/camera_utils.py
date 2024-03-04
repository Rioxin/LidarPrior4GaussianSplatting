#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import os
import random
from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)
    # 创建保存目录 gt（如果目录不存在）
    save_directory = 'gt'
    os.makedirs(save_directory, exist_ok=True)

    # 获取 gt 目录下已有文件的数量，用于确定新文件的编号
    existing_files = [f for f in os.listdir(save_directory) if os.path.isfile(os.path.join(save_directory, f))]
    file_number = len(existing_files) + 1

    # 构建保存文件的完整路径，命名方式为 saved_image_1.jpg、saved_image_2.jpg，依此类推
    save_path = os.path.join(save_directory, f"saved_image_{file_number}.jpg")

    # 假设 cam_info.image 是一个 PIL Image 对象
    #cam_info.image.save(save_path)

    #print(f"Saved to: {save_path}")

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  #FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  cx=cam_info.cx, cy=cam_info.cy,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt = np.eye(4)
    Rt[:3, :3] = camera.R.T
    Rt[:3, 3] = camera.T
    RR = np.array([  [0, 0, 1],
                        [-1, 0, 0],
                        [0, -1, 0]])

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    pos = np.dot(-camera.R,camera.T)
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
