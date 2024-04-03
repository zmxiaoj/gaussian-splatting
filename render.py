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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import numpy as np

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    # 增加深度图的渲染
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")
    depth_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_color")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(depth_color_path, exist_ok=True)

    torch.set_printoptions(profile="full", precision=10, sci_mode=False, threshold=100, edgeitems=10)
    
    # 初始化一个list保存各视角下的深度图渲染结果
    depth_images = {}
    depth_max = -1.0
    depth_min = 100000.0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depth_image = render_pkg["depth_image"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # 统计深度图像的最大值和最小值
        depth_max = max(depth_max, torch.max(depth_image))
        depth_min = min(depth_min, torch.min(depth_image))
        depth_images[idx] = depth_image[0].detach().cpu().numpy().copy()
    
    # 判断depth_max和depth_min是否合理
    if depth_max < 0 or depth_min > 100000:
        # 结束程序
        print("Invalid depth range: ", depth_max, depth_min)
        return
    # 将depth_max和depth_min转换为numpy
    depth_max = depth_max.detach().cpu().numpy()
    depth_min = depth_min.detach().cpu().numpy()
    inv_depth_max = 1 / (depth_min + 1e-6)
    inv_depth_min = 1 / (depth_max + 1e-6)
    print("depth max: ", depth_max, "depth min: ", depth_min)
    # 遍历深度图
    for idx, depth_image in enumerate(depth_images.values()):
        # # 输出depth_image的维度
        # print("depth_image shape: ", depth_image.shape)
        # 将depth_image的内容输出到文件
        with open(model_path + '/depth_image.txt', 'a') as f:
            f.write("Depth image " + str(idx) + ":\n")
            f.write("max: " + str(np.max(depth_image)) + " min: " + str(np.min(depth_image)))
            f.write("\n")
            f.write("mean: " + str(np.mean(depth_image)))
            f.write("\n")
        # 将depth_image每个像素取倒数
        depth_image = 1 / (depth_image + 1e-6)
        # inv_depth_max = min(inv_depth_max, np.max(depth_image))
        # inv_depth_min = max(inv_depth_min, np.min(depth_image))
        inv_depth_max = np.max(depth_image)
        inv_depth_min = np.min(depth_image)
        with open(model_path + '/depth_image.txt', 'a') as f:
            f.write("Inverse Depth image: \n")
            f.write("max: " + str(np.max(depth_image)) + " min: " + str(np.min(depth_image)))
            f.write("\n")
            f.write("mean: " + str(np.mean(depth_image)))
            f.write("\n")
        # 输出depth_image的范围和均值
        print("max: ", np.max(depth_image), " min: ", np.min(depth_image))
        print("mean: ", np.mean(depth_image))
        # 将depth_image归一化到0-1之间
        depth_image = (depth_image - inv_depth_min) / (inv_depth_max - inv_depth_min + 1e-6)

        with open(model_path + '/depth_image.txt', 'a') as f:
            f.write("Normalized Inverse Depth image: \n")
            f.write("max: " + str(np.max(depth_image)) + " min: " + str(np.min(depth_image)))
            f.write("\n")
            f.write("mean: " + str(np.mean(depth_image)))
            f.write("\n") 

        # 使用cv2保存深度图，先转换为0-255的整数
        depth_image *= 255
        depth_image = depth_image.astype(np.uint8)
        cv2.imwrite(os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"), depth_image)
        
        # 使用调色板将深度图像转换为伪彩色图像
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        with open(model_path + '/depth_image.txt', 'a') as f:
            f.write("Colormap Depth image: \n")
            f.write("max: " + str(np.max(depth_image)) + " min: " + str(np.min(depth_image)))
            f.write("\n")
            f.write("mean: " + str(np.mean(depth_image)))
            f.write("\n")
        # 将深度图像保存到文件
        cv2.imwrite(os.path.join(depth_color_path, '{0:05d}'.format(idx) + ".png"), depth_image)
        
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)