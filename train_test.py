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

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, exclude_vars, sh_degree_up):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # 初始化gs
    gaussians = GaussianModel(dataset.sh_degree)
    # 初始化场景
    scene = Scene(dataset, gaussians)
    # 初始化训练参数
    gaussians.training_setup(opt)

    # 根据输入参数检查哪些参数需要梯度更新
    all_vars = ['_rotation', '_scaling', '_opacity', '_features_dc', '_features_rest', '_xyz']
    for var in exclude_vars:
        if var in all_vars:
            all_vars.remove(var)
            getattr(gaussians, var).requires_grad = False
    for var in all_vars:
        getattr(gaussians, var).requires_grad = True

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    # 初始化背景颜色
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    # 记录迭代执行时间
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

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
    # 迭代循环
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # 更新学习率
        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # 提高active_sh_degree
        if sh_degree_up is True:
            if iteration % 1000 == 0:
                gaussians.oneupSHdegree()

        # Pick a random Camera
        # 随机选择视点相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        # 渲染图像
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        # 增加了渲染深度图的返回值
        image, viewspace_point_tensor, visibility_filter, radii, depth = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["depth_image"]

        # Loss
        
        # 将gt图像复制到gpu上
        gt_image = viewpoint_cam.original_image.cuda()
        # 在gpu上计算loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # 反向传播计算梯度
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            # 记录训练结果
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                # 保存训练结果
                scene.save(iteration)

            # Densification
            # 密度控制
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            iterationCheck = 200
            # Optimizer step
            # 优化器更新(在gpu上进行)
            if iteration < opt.iterations:
                # 根据梯度更新参数
                gaussians.optimizer.step()
                # 每迭代100次更新梯度和状态信息相关
                if iteration % iterationCheck == 0:
                    # 检验gaussians的_xyz梯度更新情况
                    if gaussians._xyz.grad is not None:
                        print("xyz gradiant", gaussians._xyz.grad)
                    else:
                        print("xyz dont have gradient")
                    # # 输出gaussians的_xyz
                    # print("xyz: ", gaussians._xyz)

                    # 检查gaussians的_rotation梯度更新情况
                    if gaussians._rotation.grad is not None:
                        print("rotation gradiant", gaussians._rotation.grad)
                    else:
                        print("rotation dont have gradient")
                    # # 输出gaussians的_rotation
                    # print("rotation: ", gaussians._rotation)

                    # 检查gaussians的_scaling梯度更新情况
                    if gaussians._scaling.grad is not None:
                        print("scaling gradiant", gaussians._scaling.grad)
                    else:
                        print("scaling dont have gradient")
                    # # 输出gaussians的_scaling
                    print("scaling: ", gaussians._scaling)
                    # # 输出gaussians的_scaling的最大值
                    print("scaling max: ", torch.max(gaussians._scaling))
                    
                    # 检查gaussians的_opacity梯度更新情况
                    if gaussians._opacity.grad is not None:
                        print("opacity gradiant", gaussians._opacity.grad)
                    else:
                        print("opacity dont have gradient")
                    # # 输出gaussians的_opacity
                    # print("opacity: ", gaussians._opacity)

                    # 检查gaussians的_features_dc梯度更新情况
                    if gaussians._features_dc.grad is not None:
                        print("features_dc gradiant", gaussians._features_dc.grad)
                    else:   
                        print("features_dc dont have gradient")
                    # # 输出gaussians的_features_dc
                    # print("features_dc: ", gaussians._features_dc)

                    # 检查gaussians的_features_rest梯度更新情况
                    if gaussians._features_rest.grad is not None:
                        print("features_rest gradiant", gaussians._features_rest.grad)
                    else:
                        print("features_rest dont have gradient")
                    # # 输出gaussians的_features_rest
                    # print("features_rest: ", gaussians._features_rest)     

                    # 记录训练信息
                    with open(scene.model_path + '/record.txt', 'a') as f:
                        f.write("Iteration: " + str(iteration) + "\n")
                        f.write("xyz: " + str(gaussians._xyz) + "\n")
                        f.write("rotation: " + str(gaussians._rotation) + "\n")
                        f.write("scaling: " + str(gaussians._scaling) + "\n")
                        f.write("opacity: " + str(gaussians._opacity) + "\n")
                        f.write("features_dc: " + str(gaussians._features_dc) + "\n")
                        f.write("features_rest: " + str(gaussians._features_rest) + "\n")
                        f.write("xyz gradiant: " + str(gaussians._xyz.grad) + "\n")
                        f.write("rotation gradiant: " + str(gaussians._rotation.grad) + "\n")
                        f.write("scaling gradiant: " + str(gaussians._scaling.grad) + "\n")
                        f.write("opacity gradiant: " + str(gaussians._opacity.grad) + "\n")
                        f.write("features_dc gradiant: " + str(gaussians._features_dc.grad) + "\n")
                        f.write("features_rest gradiant: " + str(gaussians._features_rest.grad) + "\n")

                # 清空梯度
                gaussians.optimizer.zero_grad(set_to_none = True)

                # 对高斯点的scaling进行clamp 20cm 
                gaussians._scaling.data = torch.clamp(gaussians._scaling.data, math.log(0.002), math.log(0.2))
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

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
    parser.add_argument("--sh_degree_up", type=bool, default=True)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.exclude_vars, args.sh_degree_up)

    # All done
    print("\nTraining complete.")
