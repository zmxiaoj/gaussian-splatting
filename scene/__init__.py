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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    # 类型注释
    gaussians : GaussianModel

    # 构造函数
    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], init=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        # 将参数赋值给实例变量
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        # 获取当前model的迭代次数
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        # 根据数据格式为colmap还是blender，加载场景信息
        # 读取colmap格式数据
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            # 调用函数字典，返回一个scene_info命名元组
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        # 读取blender格式数据
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        # 如果是第一次加载模型
        if not self.loaded_iter:
            # 将点云数据写入input.ply文件
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            # 创建2个保存相机信息的空列表
            json_cams = []
            camlist = []
            # 将训练和测试相机信息合并到camlist
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            # 遍历camlist，将相机信息转换为json格式
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            # 保存相机信息到cameras.json文件
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        # 打乱相机顺序，默认为True
        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        # 取出相机的最大范围，保存到cameras_extent
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # 对不同分辨率的相机进行加载
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            # 将相机信息转换为Camera对象，保存到camera_list，赋值给train_cameras字典
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # 加载已保存的高斯模型
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        # 创建新的高斯模型
        else:
            # 仅初始化高斯模型
            if init:
                self.gaussians.create_from_pcd_init(scene_info.point_cloud, self.cameras_extent)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    # 保存当前迭代的高斯模型
    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # 获取对应分辨率的train camera_list
    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    # 获取对应分辨率的test camera_list
    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]