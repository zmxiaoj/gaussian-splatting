import json
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

# 对3dgs中的fetchPly函数进行修改
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    # 检查是否存在 'nx', 'ny', 'nz' 字段
    if {'nx', 'ny', 'nz'}.issubset(vertices.data.dtype.names):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        # 如果不存在，为法线赋予一个默认值
        normals = np.zeros_like(positions)    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

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

    if (len(sys.argv) > 1):
        path_database = sys.argv[1]

    # Specify the paths to the JSON and COLMAP files
    path_json_file = path_database + '/licam_result.json'
    path_sfm_pointcloud = path_database + '/cloud_optim.ply'
    path_lidar_pointcloud = path_database + '/lidarCloud.ply'
    path_image_folder = path_database + '/image'

    print(path_image_folder)
    # path_colmap_file = path_database + '/3dgs_front_right'
    path_colmap_file = path_database + '/3dgs_front_right_500'

    if not os.path.exists(path_colmap_file):
        os.makedirs(path_colmap_file)
   
    # 创建sparse/0/ 文件夹
    path_colmap_file_info = path_colmap_file + '/sparse/0'
    if not os.path.exists(path_colmap_file_info):
        os.makedirs(path_colmap_file_info)
    print(path_colmap_file_info)
    # 创建images文件夹
    path_colmap_file_image = path_colmap_file + '/images'
    if not os.path.exists(path_colmap_file_image):
        os.makedirs(path_colmap_file_image)
    print(path_colmap_file_image)

    # 读取JSON文件
    with open(path_json_file, 'r') as f:
        data = json.load(f)
    # 输出JSON文件中的数据dist的key
    cameras_json = data['cameras']
    images_json = data['images']
    traj_opt_json = data['trajectory_opt']

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
            'q_bc_opt': [Tbc_opt['rotation']['x'], Tbc_opt['rotation']['y'], Tbc_opt['rotation']['z'], Tbc_opt['rotation']['w']],
            't_bc_opt': [Tbc_opt['translation']['x'], Tbc_opt['translation']['y'], Tbc_opt['translation']['z']],
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

    traj_opt = {}
    for traj in traj_opt_json:
        # 将时间戳作为key，将位姿作为value
        traj_opt[traj['timestamp']] = {
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
    #     idx += 1
    #     if idx%25 == 0:
    #         print(key)
    # 输出images_info的大小 全部图像数目
    print(len(images_info))
    # sys.exit(0)

    # 创建cameras.txt文件
    with open(path_colmap_file_info + '/cameras.txt', 'w') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        for key, value in camera_params.items():
            f.write('%d %s %d %d %s\n' % (value['camera_id'], value['model'], value['width'], value['height'], ' '.join(map(str, value['params']))))

    # check 位姿信息
    # 创建一个新的3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 设置坐标轴的标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 遍历images信息，穿件images.txt文件
    with open(path_colmap_file_info + '/images.txt', 'w') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        f.write('#   POINTS2D[] as (X, Y, POINT3D_ID)\n')
        # 遍历images_info
        # 增加枚举变量idx，idx从1开始
        idx = 0
        for key, value in images_info.items():
            # 只选择front_right相机的前300张图
            if value['camera_name'] != 'front_right_camera':
                continue
            idx += 1
            if idx == 500:
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
            pose_wb_R = R.from_quat([pose_wb_q[0], pose_wb_q[1], pose_wb_q[2], pose_wb_q[3]]).as_matrix()
            pose_wb_t = traj_opt[timestamp]['translation']
            pose_wb_t_array = np.array([pose_wb_t[0], pose_wb_t[1], pose_wb_t[2]])
            # 取出车体系线速度
            vb = np.array(traj_opt[timestamp]['vb'])
            # 取出车体系角速度
            wb = np.array(traj_opt[timestamp]['wb'])

            # 获取相机的名字
            camera_name = value['camera_name']
            # 获取相机真实时间戳
            real_timestamp = value['real_timestamp']
            # 获取相机的id
            camera_id = camera_params[camera_name]['camera_id']
            camera_height = camera_params[camera_name]['height']
            camera_width = camera_params[camera_name]['width']
            # 计算相机曝光延时和rolling shutter每行相机曝光延时
            td = camera_params[camera_name]['td']
            rs_per_row = camera_params[camera_name]['rs_per_row']
            # 计算1/2图像处的时间
            deltat_camera = td + rs_per_row * camera_height * 0.5
            # print(deltat_camera)
            # 计算相机真实时间相对车体时间的时间差
            groupTimeDiff = (real_timestamp - timestamp) * 1e-6
            # print(groupTimeDiff)
            # 整体时间延时
            deltat = deltat_camera + groupTimeDiff
            # 更新车体位姿
            pose_wb_t_array_opt = pose_wb_t_array + np.dot(pose_wb_R, vb) * deltat
            pose_wb_R_opt = pose_wb_R @ R.from_rotvec(wb * deltat).as_matrix()
            # 获取车体系到相机的外参
            pose_bc_q = camera_params[camera_name]['q_bc_opt']       
            pose_bc_R = R.from_quat([pose_bc_q[0], pose_bc_q[1], pose_bc_q[2], pose_bc_q[3]]).as_matrix()
            pose_bc_R_array = np.array(pose_bc_R)
            pose_bc_t = camera_params[camera_name]['t_bc_opt']
            pose_bc_t_array = np.array([pose_bc_t[0], pose_bc_t[1], pose_bc_t[2]])
            # 计算世界系到相机的变换
            pose_wc_R = pose_wb_R_opt @ pose_bc_R_array
            pose_wc_t = np.dot(pose_wb_R_opt, pose_bc_t_array) + pose_wb_t_array_opt
        
            pose_wc_q_xyzw = R.from_matrix(pose_wc_R).as_quat()
            pose_wc_q_wxyz = [pose_wc_q_xyzw[3], pose_wc_q_xyzw[0], pose_wc_q_xyzw[1], pose_wc_q_xyzw[2]]
            
            pltR = R.from_quat(pose_wc_q_xyzw).as_matrix()

            pltt = pose_wc_t.flatten()
            # 绘制pose_wc_t的点
            ax.scatter(pltt[0], pltt[1], pltt[2], c='r', marker='o')
            # 绘制pose_wc_t的线
            # ax.plot([0, pltt[0]], [0, pltt[1]], [0, pltt[2]], c='b')


            # ax.quiver(pltt[0], pltt[1], pltt[2], pltR[0, 0], pltR[1, 0], pltR[2, 0], color='r', length=1)
            # ax.quiver(pltt[0], pltt[1], pltt[2], pltR[0, 1], pltR[1, 1], pltR[2, 1], color='g', length=1)
            # ax.quiver(pltt[0], pltt[1], pltt[2], pltR[0, 2], pltR[1, 2], pltR[2, 2], color='b', length=1)

            # 写入images.txt文件
            f.write('%d %f %f %f %f %f %f %f %d %s\n' % (idx, pose_wc_q_wxyz[0], pose_wc_q_wxyz[1], pose_wc_q_wxyz[2], pose_wc_q_wxyz[3], pose_wc_t[0], pose_wc_t[1], pose_wc_t[2], camera_id, key + '.jpg'))
            # 在images.txt文件中写入第二行数据，内容为(1, 1, -1)
            f.write('1 1 -1\n')
            
        # 输出处理照片数量
        print('Processed images: ' + str(idx))
        plt.show()
        
    # 将path_sfm_pointcloud拷贝重命名为points3D.ply
    # 读入path_sfm_pointcloud点云文件
    pcd = fetchPly(path_sfm_pointcloud)
    # print(pcd)
    # 如果颜色值为[0,1]，则将颜色值转换为[0,255]
    if np.max(pcd.colors) <= 1:
        colors = (pcd.colors * 255).astype(np.uint8)
    else:
        colors = pcd.colors
    # 将pcd点云文件存储为points3D.ply文件
    storePly(path_colmap_file_info + '/points3D.ply', pcd.points, colors)
    
    pcd2 = fetchPly(path_colmap_file_info + '/points3D.ply')
    # # 验证pcd和pcd2是否一致
    # print(pcd.points[321] == pcd2.points[321])
    # # 颜色变成了[0,0,0]
    # print(pcd.colors[321] == pcd2.colors[321])
    # print(pcd.normals[321] == pcd2.normals[321])

    # with open(path_sfm_pointcloud, 'r') as f:
    #     lines = f.readlines()
    #     # print(lines)
    # # 创建points3D.ply文件
    # with open(path_colmap_file_info + '/points3D.ply', 'w') as f:
    #     for line in lines[:]:
    #         f.write(line)

