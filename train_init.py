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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import math
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, exclude_vars, sh_degree_up, clamp = False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # 初始化gs
    gaussians = GaussianModel(dataset.sh_degree)
    # 初始化场景
    scene = Scene(dataset, gaussians, init=True)
    # 初始化训练参数
    gaussians.training_setup(opt)

    # 初始化背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 设置torch输出精度和最大变量显示数量
    torch.set_printoptions(precision=8, sci_mode=False, threshold=1000)

    print("Initial Gaussians: ")
    # 输出gaussians的_xyz，最大值，最小值，均值
    print("xyz: ", gaussians._xyz)
    print("xyz max: ", torch.max(gaussians._xyz))
    print("xyz min: ", torch.min(gaussians._xyz))
    print("xyz mean: ", torch.mean(gaussians._xyz))
    # 输出gaussians的_rotation，最大值，最小值，均值
    print("rotation: ", gaussians._rotation)
    print("rotation max: ", torch.max(gaussians._rotation))
    print("rotation min: ", torch.min(gaussians._rotation))
    print("rotation mean: ", torch.mean(gaussians._rotation))
    # 输出gaussians的_scaling，最大值，最小值，均值
    print("scaling: ", gaussians._scaling)
    print("scaling max: ", torch.max(gaussians._scaling))
    print("scaling min: ", torch.min(gaussians._scaling))
    print("scaling mean: ", torch.mean(gaussians._scaling))
    # 输出gaussians的_opacity，最大值，最小值，均值
    print("opacity: ", gaussians._opacity)
    print("opacity max: ", torch.max(gaussians._opacity))
    print("opacity min: ", torch.min(gaussians._opacity))
    print("opacity mean: ", torch.mean(gaussians._opacity))
    # 输出gaussians的_features_dc，最大值，最小值，均值
    print("features_dc: ", gaussians._features_dc)
    print("features_dc max: ", torch.max(gaussians._features_dc))
    print("features_dc min: ", torch.min(gaussians._features_dc))
    print("features_dc mean: ", torch.mean(gaussians._features_dc))
    # 输出gaussians的_features_rest，最大值，最小值，均值
    print("features_rest: ", gaussians._features_rest) 
    print("features_rest max: ", torch.max(gaussians._features_rest))
    print("features_rest min: ", torch.min(gaussians._features_rest))
    print("features_rest mean: ", torch.mean(gaussians._features_rest))

    with open(scene.model_path + '/record.txt', 'w') as f:
        f.write("Initial Gaussians: \n")
        # 记录初始gaussians的xyz，最大值，最小值，均值
        f.write("xyz: " + str(gaussians._xyz) + "\n")
        f.write("xyz max: " + str(torch.max(gaussians._xyz)) + "\n")
        f.write("xyz min: " + str(torch.min(gaussians._xyz)) + "\n")
        f.write("xyz mean: " + str(torch.mean(gaussians._xyz)) + "\n")
        # 记录初始gaussians的rotation，最大值，最小值，均值
        f.write("rotation: " + str(gaussians._rotation) + "\n")
        f.write("rotation max: " + str(torch.max(gaussians._rotation)) + "\n")
        f.write("rotation min: " + str(torch.min(gaussians._rotation)) + "\n")
        f.write("rotation mean: " + str(torch.mean(gaussians._rotation)) + "\n")
        # 记录初始gaussians的scaling，最大值，最小值，均值
        f.write("scaling: " + str(gaussians._scaling) + "\n")
        f.write("scaling max: " + str(torch.max(gaussians._scaling)) + "\n")
        f.write("scaling min: " + str(torch.min(gaussians._scaling)) + "\n")
        f.write("scaling mean: " + str(torch.mean(gaussians._scaling)) + "\n")
        # 记录初始gaussians的opacity，最大值，最小值，均值
        f.write("opacity: " + str(gaussians._opacity) + "\n")
        f.write("opacity max: " + str(torch.max(gaussians._opacity)) + "\n")
        f.write("opacity min: " + str(torch.min(gaussians._opacity)) + "\n")
        f.write("opacity mean: " + str(torch.mean(gaussians._opacity)) + "\n")
        # 记录初始gaussians的features_dc，最大值，最小值，均值
        f.write("features_dc: " + str(gaussians._features_dc) + "\n")
        f.write("features_dc max: " + str(torch.max(gaussians._features_dc)) + "\n")
        f.write("features_dc min: " + str(torch.min(gaussians._features_dc)) + "\n")
        f.write("features_dc mean: " + str(torch.mean(gaussians._features_dc)) + "\n")
        # 记录初始gaussians的features_rest，最大值，最小值，均值
        f.write("features_rest: " + str(gaussians._features_rest) + "\n")
        f.write("features_rest max: " + str(torch.max(gaussians._features_rest)) + "\n")
        f.write("features_rest min: " + str(torch.min(gaussians._features_rest)) + "\n")
        f.write("features_rest mean: " + str(torch.mean(gaussians._features_rest)) + "\n")
    
    scene.save(0)
    print("Saved initial scene to ", scene.model_path)


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    # 增加输入参数，设置不参与优化的变量
    parser.add_argument("--exclude_vars", nargs="*", type=str, default=[])
    # 增加输入参数，设置是否进行提高sh_degree
    parser.add_argument("--sh_degree_up", action="store_false", default=True)
    # 增加输入参数，设置是否进行clamp
    parser.add_argument("--clamp", action="store_true", default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.exclude_vars, args.sh_degree_up, args.clamp)

    # All done
    print("\nTraining complete.")
