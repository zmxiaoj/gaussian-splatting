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

    torch.set_printoptions(profile="full", precision=10, sci_mode=False, threshold=10000, edgeitems=10)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        rendering = render_pkg["render"]
        depth_image = render_pkg["depth_image"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # 将depth_image的内容输出到文件
        with open(model_path + '/depth_image.txt', 'w') as f:
            f.write("Depth image: \n")
            f.write(str(depth_image))
        print("depth_image shape: ", depth_image.shape, "depth_image type: ", depth_image.dtype)
        print("depth_image: ", depth_image)
        # 只进行单张图像的归一化
        # 将depth_image这个tensor归一化到0-1之间
        depth_image = (depth_image - torch.min(depth_image)) / (torch.max(depth_image) - torch.min(depth_image))
        # 颜色取反，深度越大的地方颜色越深
        depth_image = (1 - depth_image) 
        torchvision.utils.save_image(depth_image, os.path.join(depth_path, '{0:05d}'.format(idx) + ".png"))
        # 使用调色板将深度图像转换为伪彩色图像
        # 将tensor转换为二维numpy数组
        depth_image *= 255
        depth_image = depth_image[0].cpu().numpy().astype(np.uint8)
        # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_INFERNO)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
        print("depth_image: ", depth_image)
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