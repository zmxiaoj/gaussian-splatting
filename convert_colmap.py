import json
import argparse
import sys
import os
import cv2
from plyfile import PlyData, PlyElement
import numpy as np
from typing import NamedTuple
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

# from scene/colmap_loader.py
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

# 对3dgs中的fetchPly函数进行修改
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    # 检查是否存在 'red', 'green', 'blue' 字段
    if {'red', 'green', 'blue'}.issubset(vertices.data.dtype.names):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    else:
        # 如果不存在，随机初始化颜色，rgb均是[0,255]之间随机整数
        colors = (np.random.rand(positions.shape[0], 3) * 255).astype(np.uint8)
    # 检查是否存在 'nx', 'ny', 'nz' 字段
    if {'nx', 'ny', 'nz'}.issubset(vertices.data.dtype.names):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        # 如果不存在，为法线赋予一个默认值
        normals = np.zeros_like(positions)    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

# 3dgs
def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)
    colors = (pcd.colors * 255).astype(np.uint8)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This is a script to convert data into colmap format.")
    parser.add_argument("path2database", help="Path to the database.")
    parser.add_argument("--camera", default="all", help="First camera to process.")
    parser.add_argument("--number", default=100, help="Total number of images to process.")
    parser.add_argument("--pointcloud", default="sfm", help="Pointcloud to process.")
    parser.add_argument("--plottraj", default=False, help="Plot trajectory of camera")
    args = parser.parse_args()

    path_database = args.path2database
    # print('type: ', type(path_database), ' path_database: ', path_database)
    # 检验args.camera是否属于{'front_left', 'front_right', 'rear_left', 'rear_right'}
    if args.camera not in ['all', 'front_left', 'front_right', 'rear_left', 'rear_right']:
        print('camera:' + args.camera + ' does not match')
        sys.exit(0)
    cameraProcess = args.camera
    # print('type: ', type(cameraProcess), ' cameraProcess: ', cameraProcess)
    imageNumber = int(args.number)
    # print('type: ', type(imageNumber), ' imageNumber: ', imageNumber)
    pointcloutType = args.pointcloud
    # print('type: ', type(pointcloutType), ' pointcloutType: ', pointcloutType)
    plotTraj = args.plottraj
    # print('type: ', type(plotTraj), 'plotTraj', plotTraj)

    # 检查path_database是否以sfm结尾
    if path_database.endswith('sfm'):
        print('path_database:' + path_database)
    else:
        print('path_database:' + path_database + ' does not match')
        sys.exit(0)

    path_json_file = path_database + '/licam_result.json'
    path_sfm_pointcloud = path_database + '/cloud_optim.ply'
    path_lidar_pointcloud = path_database + '/lidarCloud.ply'
    path_image_folder = path_database + '/image'
    # 检查path_json_file是否存在
    if os.path.exists(path_json_file):
        print('path_json_file:' + path_json_file)
    else:
        print('path_json_file:' + path_json_file + ' does not exist')
        sys.exit(0)
    if os.path.exists(path_sfm_pointcloud):
        print('path_sfm_pointcloud:' + path_sfm_pointcloud)
    else:
        print('path_sfm_pointcloud:' + path_sfm_pointcloud + ' does not exist')
        sys.exit(0)
    if os.path.exists(path_lidar_pointcloud):
        print('path_lidar_pointcloud:' + path_lidar_pointcloud)
    else:
        print('path_lidar_pointcloud:' + path_lidar_pointcloud + ' does not exist')
        sys.exit(0)
    if os.path.exists(path_image_folder):
        print('path_image_folder:' + path_image_folder)
    else:
        print('path_image_folder:' + path_image_folder + ' does not exist')
        sys.exit(0)

    # 创建保存结果的位置，默认是/3dgs
    savepath = '/3dgs_' + cameraProcess + '_' + str(imageNumber) + '_' + pointcloutType
    path_colmap_file = path_database + savepath
    if not os.path.exists(path_colmap_file):
        os.makedirs(path_colmap_file)
        print('create folder:' + path_colmap_file)
    else:
        print('folder already exists:' + path_colmap_file)
        sys.exit(0)
   
    # 创建sparse/0/ 文件夹
    path_colmap_file_info = path_colmap_file + '/sparse/0'
    if not os.path.exists(path_colmap_file_info):
        os.makedirs(path_colmap_file_info)
        print('create folder:' + path_colmap_file_info)
    else:
        print('folder already exists:' + path_colmap_file_info)
    # 创建images文件夹
    path_colmap_file_image = path_colmap_file + '/images'
    if not os.path.exists(path_colmap_file_image):
        os.makedirs(path_colmap_file_image)
        print('create folder:' + path_colmap_file_image)
    else:
        print('folder already exists:' + path_colmap_file_image)

    # 读取json文件
    with open(path_json_file, 'r') as f:
        data = json.load(f)
    # 取出json中的cameras, images, trajectory_opt
    cameras_json = data['cameras']
    images_json = data['images']
    traj_opt_json = data['trajectory_opt']

    # 初始化相机参数字典
    camera_params = {}
    # 在循环中增加一个枚举变量idx，idx从1开始
    for idx, camera in enumerate(cameras_json, 1):
        Tbc_opt = camera['Tbc_opt']
        Tbc_raw = camera['Tbc_raw']
        intrinsic_opt = camera['intrinsic_opt']
        camera_name = camera['camera_name']
        # 用新的数据结构将相机参数组织起来，用camera_name作为key
        camera_params[camera_name] = {
            'model': "PINHOLE",
            'camera_id': idx,
            'camera_name': camera['camera_name'],
            'width': camera['width'],
            'height': camera['height'],
            # params中的参数fx, fy, cx, cy 按照固定顺序保存
            'params': [intrinsic_opt['fx'], intrinsic_opt['fy'], intrinsic_opt['cx'], intrinsic_opt['cy']],
            # q_xyzw
            'q_bc_opt': [Tbc_opt['rotation']['x'], Tbc_opt['rotation']['y'], Tbc_opt['rotation']['z'], Tbc_opt['rotation']['w']],
            't_bc_opt': [Tbc_opt['translation']['x'], Tbc_opt['translation']['y'], Tbc_opt['translation']['z']],
            # q_xyzw
            'q_bc_raw': [Tbc_raw['rotation']['x'], Tbc_raw['rotation']['y'], Tbc_raw['rotation']['z'], Tbc_raw['rotation']['w']],
            't_bc_raw': [Tbc_raw['translation']['x'], Tbc_raw['translation']['y'], Tbc_raw['translation']['z']],
            # 相机曝光时间
            'td': intrinsic_opt['td'],
            # rolling shutter每行相机曝光延时
            'rs_per_row': camera['rs_per_row']
        }
    # # 遍历camera_params
    # for key, value in camera_params.items():
    #     print(key, value)
    # 输出camera_params的大小
    # print(len(camera_params))

    # 初始化车体位姿字典
    traj_opt = {}
    for traj in traj_opt_json:
        # 将时间戳作为key，将位姿作为value
        traj_opt[traj['timestamp']] = {
            # q_xyzw
            'rotation': [traj['rotation']['x'], traj['rotation']['y'], traj['rotation']['z'], traj['rotation']['w']],
            'translation': [traj['translation']['x'], traj['translation']['y'], traj['translation']['z']],
            'vb': [traj['vb']['x'], traj['vb']['y'], traj['vb']['z']],
            'wb': [traj['wb']['x'], traj['wb']['y'], traj['wb']['z']]
        }
    # # 遍历traj_opt
    # for key, value in traj_opt.items():
    #     print(key, value)
    # # 输出traj_opt的大小 *4为图像的数量
    # print(len(traj_opt))

    # 初始化图像信息字典
    images_info = {}
    for image in images_json:
        # 将相机+时间戳作为key，将图像路径作为value
        images_info[image['camera_name'] + '_' + str(image['real_timestamp'])] = {
            'image_path' : image['image_path'],
            'camera_name' : image['camera_name'],
            'timestamp' : image['timestamp'],
            'real_timestamp' : image['real_timestamp']
        }
    # 将images_info按照key升序排列
    images_info = dict(sorted(images_info.items()))
    # # 遍历images_info
    # for key, value in images_info.items():
    #         print(key)
    # 输出images_info的大小 全部图像数目
    # print(len(images_info))
    # sys.exit(0)

    # 创建cameras.txt文件
    with open(path_colmap_file_info + '/cameras.txt', 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for key, value in camera_params.items():
            f.write('%d %s %d %d %s\n' % (value['camera_id'], value['model'], value['width'], value['height'], ' '.join(map(str, value['params']))))

    # 检验位姿结果
    # 创建一个新的3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴的标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 遍历images信息，创建images.txt文件
    with open(path_colmap_file_info + '/images.txt', 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        # 遍历images_info
        # 增加枚举变量idx，idx从1开始
        idx = 0
        for key, value in images_info.items():
            # 选择cameraProcess相机的前imageNumber张图
            if cameraProcess != 'all':
                if value['camera_name'] != (cameraProcess + '_camera'):
                    continue
            idx += 1
            if idx >= imageNumber + 1:
                break

            # 根据名字读取图像
            image = cv2.imread(path_image_folder + '/' + value['camera_name'] + '_undistort/' + str(value['timestamp']) + '.jpg')

            # 以real_timestamp命名保存到images文件夹中
            cv2.imwrite(path_colmap_file_image + '/' + key + '.jpg', image)
            # 输出图像保存信息
            print('Image saved: ' + path_colmap_file_image + '/' + key + '.jpg')
            
            # 获取标称时间戳作为key，获取车体位姿及速度
            timestamp = value['timestamp']
            # 获取车体系位姿
            pose_wb_q = traj_opt[timestamp]['rotation']
            # q_xyzw
            pose_wb_R = R.from_quat([pose_wb_q[0], pose_wb_q[1], pose_wb_q[2], pose_wb_q[3]]).as_matrix()
            pose_wb_R_array = np.array(pose_wb_R)
            print("type: ", type(pose_wb_R_array), "pose_wb_R_array: ", pose_wb_R_array)
            pose_wb_t = traj_opt[timestamp]['translation']
            pose_wb_t_array = np.array([pose_wb_t[0], pose_wb_t[1], pose_wb_t[2]])
            print("type: ", type(pose_wb_t_array), "pose_wb_t_array: ", pose_wb_t_array)
            # 取出车体系线速度
            vb = np.array(traj_opt[timestamp]['vb'])
            print("type: ", type(vb), "vb: ", vb)
            # 取出车体系角速度
            wb = np.array(traj_opt[timestamp]['wb'])
            print("type: ", type(wb), "wb: ", wb)

            # 获取相机的名字
            camera_name = value['camera_name']
            # 获取相机真实时间戳
            real_timestamp = value['real_timestamp']
            # 获取相机的id
            camera_id = camera_params[camera_name]['camera_id']
            # 获取相机的高度
            camera_height = camera_params[camera_name]['height']
            # 获取相机的宽度
            camera_width = camera_params[camera_name]['width']
            # 计算相机曝光延时和rolling shutter每行相机曝光延时
            td = camera_params[camera_name]['td']
            rs_per_row = camera_params[camera_name]['rs_per_row']
            # print('type: ', type(rs_per_row), ' rs_per_row: ', rs_per_row)
            # 计算1/2图像处的时间
            # 0.5 * 1080 * 1.90476e-05 = 0.0102852704(s)
            deltat_camera = td + rs_per_row * camera_height * 0.5
            print('type: ', type(deltat_camera), ' deltat_camera: ', deltat_camera)
            # 计算相机真实时间相对车体时间的时间差
            # front_left:0.033(s) front_right:-0.033(s) rear_left:0.000001(s) rear_right:0.00025(s)
            groupTimeDiff = (real_timestamp - timestamp) * 1e-6
            print('type: ', type(groupTimeDiff), ' groupTimeDiff: ', groupTimeDiff)
            # 整体时间延时
            deltat = deltat_camera + groupTimeDiff
            print('type: ', type(deltat), ' deltat: ', deltat)
            # 更新车体位姿
            pose_wb_t_array_opt = pose_wb_t_array + np.dot(pose_wb_R_array, vb * deltat) 
            print("type: ", type(pose_wb_t_array_opt), "pose_wb_t_array_opt: ", pose_wb_t_array_opt)
            pose_wb_R_opt = pose_wb_R_array @ np.array(R.from_rotvec(wb * deltat).as_matrix())
            print("type: ", type(pose_wb_R_opt), "pose_wb_R_opt: ", pose_wb_R_opt)
            
            # # 取消位姿优化
            # pose_wb_R_opt = pose_wb_R_array
            # pose_wb_t_array_opt = pose_wb_t_array

            # 获取车体系到相机的外参
            pose_bc_q = camera_params[camera_name]['q_bc_opt']       
            pose_bc_R = R.from_quat([pose_bc_q[0], pose_bc_q[1], pose_bc_q[2], pose_bc_q[3]]).as_matrix()
            pose_bc_R_array = np.array(pose_bc_R)
            pose_bc_t = camera_params[camera_name]['t_bc_opt']
            pose_bc_t_array = np.array([pose_bc_t[0], pose_bc_t[1], pose_bc_t[2]])
            # 计算世界系到相机的变换
            pose_wc_R = pose_wb_R_opt @ pose_bc_R_array
            # 输出类型和值
            print("type: ", type(pose_wc_R), " pose_wc_R: ", pose_wc_R)
            pose_wc_t = np.dot(pose_wb_R_opt, pose_bc_t_array) + pose_wb_t_array_opt
            # 输出类型和值
            print("type: ", type(pose_wc_t), " pose_wc_t: ", pose_wc_t)
            # 逆变换
            pose_cw_R = pose_wc_R.transpose()
            print("type: ", type(pose_cw_R), " pose_cw_R: ", pose_cw_R)
            pose_cw_t = -np.dot(pose_cw_R, pose_wc_t)
            print("type: ", type(pose_cw_t), " pose_cw_t: ", pose_cw_t)
            
            pose_cw_q_xyzw = R.from_matrix(pose_cw_R).as_quat()
            if (pose_cw_q_xyzw[3] < 0):
                pose_cw_q_xyzw = -pose_cw_q_xyzw
            print("type: ", type(pose_cw_q_xyzw), "pose_wc_q_xyzw: ", pose_cw_q_xyzw)
            pose_cw_q_wxyz = np.array([pose_cw_q_xyzw[3], pose_cw_q_xyzw[0], pose_cw_q_xyzw[1], pose_cw_q_xyzw[2]])
            print("type: ", type(pose_cw_q_wxyz), "pose_wc_q_wxyz: ", pose_cw_q_wxyz)
            # 绘制pose_wc_t的点
            camera_position = pose_wc_t
            ax.scatter(*camera_position, c='r', marker='o')
            camera_direction = pose_wc_R[:3, :3] @ np.array([1, 1, 1])
            ax.quiver(*camera_position, *camera_direction, color='b', length=0.025)

            # 写入images.txt文件
            f.write('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %d %s\n' % (idx, pose_cw_q_wxyz[0], pose_cw_q_wxyz[1], pose_cw_q_wxyz[2], pose_cw_q_wxyz[3], pose_cw_t[0], pose_cw_t[1], pose_cw_t[2], camera_id, key + '.jpg'))
            # 在images.txt文件中写入第二行数据，内容为(1, 1, -1)
            f.write('1 1 -1\n')
            # 将上面两行内容输出到命令行
            print('%d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %d %s' % (idx, pose_cw_q_wxyz[0], pose_cw_q_wxyz[1], pose_cw_q_wxyz[2], pose_cw_q_wxyz[3], pose_cw_t[0], pose_cw_t[1], pose_cw_t[2], camera_id, key + '.jpg'))
            
        # 输出处理照片数量
        print('Processed images: ', max(0, idx - 1))
        if plotTraj:
            plt.show()
        
    # 将path_sfm_pointcloud拷贝重命名为points3D.ply
    # 读入path_sfm_pointcloud点云文件
    # pcd = fetchPly(path_sfm_pointcloud)
    # 读入path_lidar_pointcloud点云文件
    if pointcloutType == 'sfm':
        pcd = fetchPly(path_sfm_pointcloud)
        print('Input sfm pointcloud: ' + path_sfm_pointcloud)
    elif pointcloutType == 'lidar':
        pcd = fetchPly(path_lidar_pointcloud)
        print('Input lidar pointcloud: ' + path_lidar_pointcloud)
    else:
        print('pointcloutType:' + pointcloutType + ' does not match')
        sys.exit(0)
    
    # 如果颜色值为[0,1]，则将颜色值转换为[0,255]
    if np.max(pcd.colors) <= 1:
        colors = (pcd.colors * 255).astype(np.uint8)
    else:
        colors = pcd.colors

    # 将pcd点云文件存储为points3D.ply文件
    storePly(path_colmap_file_info + '/points3D.ply', pcd.points, colors)
    # 输出points3D.txt点云文件保存信息
    print('Point cloud saved: ' + path_colmap_file_info + '/points3D.ply')

    # 将ply点云文件转换为point3D.txt文件
    with open(path_colmap_file_info + '/points3D.txt', 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write('# Number of points: %d\n' % pcd.points.shape[0]) 
        for i in range(pcd.points.shape[0]):
            f.write('%d %.12f %.12f %.12f %d %d %d 0' % (i, pcd.points[i, 0], pcd.points[i, 1], pcd.points[i, 2], colors[i, 0], colors[i, 1], colors[i, 2]))
            f.write('\n')
    # 输出points3D.txt点云文件保存信息
    print('Point cloud saved: ' + path_colmap_file_info + '/points3D.txt')
