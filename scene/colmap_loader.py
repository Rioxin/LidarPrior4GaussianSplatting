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

import numpy as np
import collections
import struct
import json
from scipy.spatial.transform import Rotation
from plyfile import PlyData, PlyElement

CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    xyzs = None
    rgbs = None
    errors = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1


    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    errors = np.empty((num_points, 1))
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = np.array(float(elems[7]))
                xyzs[count] = xyz
                rgbs[count] = rgb
                errors[count] = error
                count += 1

    return xyzs, rgbs, errors

def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """


    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]

        xyzs = np.empty((num_points, 3))
        rgbs = np.empty((num_points, 3))
        errors = np.empty((num_points, 1))

        for p_id in range(num_points):
            binary_point_line_properties = read_next_bytes(
                fid, num_bytes=43, format_char_sequence="QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(
                fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(
                fid, num_bytes=8*track_length,
                format_char_sequence="ii"*track_length)
            xyzs[p_id] = xyz
            rgbs[p_id] = rgb
            errors[p_id] = error
    return xyzs, rgbs, errors

def read_intrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                assert model == "PINHOLE", "While the loader support other types, the rest of the code assumes PINHOLE"
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_bytes=8*num_params,
                                     format_char_sequence="d"*num_params)
            cameras[camera_id] = Camera(id=camera_id,
                                        model=model_name,
                                        width=width,
                                        height=height,
                                        params=np.array(params))
        assert len(cameras) == num_cameras
    return cameras

def pose_rotation(pose):

    euler_angles = [pose["yaw"], pose["pitch"], pose["roll"]]
    rotation = Rotation.from_euler('zyx', euler_angles)

    return rotation

def pose_translation(pose):

    translation = np.array([pose["x"], pose["y"], pose["z"]])

    return translation

def create_transform_matrix(rotation_matrix, translation_vector):
    """
    创建变换矩阵

    参数：
    - rotation_matrix: 3x3 的旋转矩阵
    - translation_vector: 3 维的位移向量

    返回：
    - transform_matrix: 4x4 的变换矩阵
    """
    # 创建一个单位矩阵作为变换矩阵的基础
    transform_matrix = np.eye(4)

    # 将旋转矩阵的内容复制到变换矩阵的左上角（3x3 子矩阵）
    transform_matrix[:3, :3] = rotation_matrix

    # 将位移向量复制到变换矩阵的右侧（前三个元素）
    transform_matrix[:3, 3] = translation_vector

    return transform_matrix

def decompose_transform_matrix(transform_matrix):
    """
    将变换矩阵分离为旋转矩阵和位移向量

    参数：
    - transform_matrix: 4x4 的变换矩阵

    返回：
    - rotation_matrix: 3x3 的旋转矩阵
    - translation_vector: 3 维的位移向量
    """
    # 提取旋转矩阵（左上角的3x3子矩阵）
    rotation_matrix = transform_matrix[:3, :3]

    # 提取位移向量（右侧的前三个元素）
    translation_vector = transform_matrix[:3, 3]

    return rotation_matrix, translation_vector

def pose_to_transform(pose):
    rotation = pose_rotation(pose).as_matrix()
    translation = pose_translation(pose)
    transform = create_transform_matrix(rotation,translation)
    return transform

def read_intrinsics_Json(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    # 假设文件路径为 path_to_json_file
    with open(path_to_model_file, 'r') as json_file: 
        data = json.load(json_file)
        # 获取 run_params 列表中的第一个元素
        run_params = data["run_params"][0]
        # 遍历 camera_params 列表
        for camera_params in run_params["camera_params"]:
            # 获取 camera参数
            camera_id = camera_params["camera_id"]
            width = camera_params["width"]
            height = camera_params["height"]
            model = "PINHOLE"
            camera_matrix = camera_params["camera_matrix"]
            #Colmap的内参是： fy": 686.5524553956232, "fx": 826.1663097380575
            params = [camera_matrix["fx"],camera_matrix["fy"],camera_matrix["cx"],camera_matrix["cy"]]
            #params = [826.1663097380575,686.5524553956232,camera_matrix["cx"],camera_matrix["cy"]]
            
            
            # 在这里可以使用 camera_id 进行其他操作 686.552455395623  
            '''
            print("Camera ID:", camera_id)
            print("Camera width:", width)
            print("Camera height:", height)
            print("Camera model:", model)
            print("Camera params:", params)
            '''
            #if(camera_id == "CAM_PBQ_FRONT_RIGHT" or camera_id == "CAM_PBQ_FRONT_LEFT" ):
            if(1):
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                            width=width, height=height,
                                            params=params)
    return cameras

def save_ply(points, filename):
    vertices = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    plydata = PlyData([PlyElement.describe(vertices, 'vertex')])
    plydata.write(filename)

def read_extrinsics_Json(path_to_model_file):
     # 读取 JSON 数据
    with open(path_to_model_file, 'r') as json_file: 
        data = json.load(json_file)
        images = {}
        # 获取第一个轨迹
        trajectory = data["trajectories"][0]
        # 在循环之前初始化计数器
        image_id = 0
        # 初始化一个列表，用于存储所有tvec
        all_tvec_points = []

        # 使用字典保存 camera_id 对应的 camera_to_vehicle
        camera_to_vehicle_pose = {}
        for run_param in data["run_params"]:
            for camera_param in run_param["camera_params"]:
                    camera_id = camera_param["camera_id"]
                    camera_to_vehicle = camera_param["camera_to_vehicle"]
                    camera_to_vehicle_pose[camera_id] = camera_to_vehicle

            # 打印保存的字典
            #print(camera_to_vehicle_pose['CAM_PBQ_FRONT_TELE']['x'])
            # 遍历轨迹中的所有视图节点
        for view_node in trajectory["view_nodes"]:
            #timestamp = view_node["timestamp"]
            timestamp = '{:.3f}'.format(view_node["timestamp"])

            # 获取与之相关的相机对齐信息
            align_images_info = view_node["align_images_info"]
            vehicle_pose =  view_node["pose"]
            for align_info in align_images_info:
                camera_id = align_info["camera_id"]
                camera_pose = align_info["pose"]
                
                RR = np.array([ [ 0,  0, 1],
                                [-1,  0, 0],
                                [ 0, -1, 0]])
                # 计算变换
                transform_vehicle_to_world = pose_to_transform(vehicle_pose)
                transform_camera_align = pose_to_transform(camera_pose)
                transform_camera_to_vehicle = pose_to_transform(camera_to_vehicle_pose[camera_id])

                transform_vehicle_to_world_plus = np.dot(transform_vehicle_to_world,transform_camera_align)
                transform_camera_to_world = np.dot(transform_vehicle_to_world_plus,transform_camera_to_vehicle)
                transform_camera_to_world[:3, :3] = np.dot(transform_camera_to_world[:3, :3],RR)
                transform_world_to_camera = np.linalg.inv(transform_camera_to_world)
                #print("inverse_matrix",inverse_matrix)


                rotation_world_to_camera , translation_world_to_camera = decompose_transform_matrix(transform_world_to_camera)
                #print("translation_world_to_camera",translation_world_to_camera)

                
                #rotation_world_to_camera = np.dot(np.linalg.inv(RR),rotation_world_to_camera)
                qvec = Rotation.from_matrix(rotation_world_to_camera).as_quat()
                #print("q1",qvec)
                qvec = rotmat2qvec(rotation_world_to_camera)
                #print("q2",qvec)
                tvec = translation_world_to_camera
                
                image_name = "/" +str(timestamp) + "/" + camera_id + ".png"
                #if(camera_id == "CAM_PBQ_FRONT_LEFT"  and image_id<=1000000):
                #if(camera_id == "CAM_PBQ_FRONT_RIGHT" or camera_id == "CAM_PBQ_FRONT_LEFT" ):
                if(image_id<=1000000):
                    images[image_id] = Image(
                        id=image_id, qvec=qvec, tvec=tvec,
                        camera_id=camera_id, name=image_name,
                        xys=None, point3D_ids=None)
                    #print("Camera camera_id:", camera_id)
                    #r1 , t1 = decompose_transform_matrix(transform_camera_to_world)
                    #print(tvec)
                    # 将当前tvec添加到列表中
                    all_tvec_points.append(tuple(tvec))
                    #print("rotation_world_to_camera",rotation_world_to_camera)
                    # 每次循环后递增计数器
                    image_id += 1
    # 保存所有tvec到PLY文件
    save_ply(all_tvec_points, 'output_point_cloud.ply')
    return images
def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                       tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=None, 
                    point3D_ids=None
                    )
    return images


def read_colmap_bin_array(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_dense.py

    :param path: path to the colmap binary file.
    :return: nd array with the floating point values in the value
    """
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()
