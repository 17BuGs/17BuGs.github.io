---
redirect_from: /_posts/2024-12-12-nuscenes_radar
title: OpenPCDet配置NuScenes数据集Radar模态
tags: 经验分享
---


##### 背景

OpenPCDet框架中给出的NuScenes数据集的生成代码是基于LiDAR（或LiDAR + Camera）模态的，本文给出了在OpenPCDet框架中配置NuScenes数据集Radar模态的代码。

##### 方法

1. 修改`nuscenes_utils.py`：
   
   ```python
    """
    The NuScenes data pre-processing and evaluation is modified from
    https://github.com/traveller59/second.pytorch and https://github.com/poodarchu/Det3D
    """

    import operator
    from functools import reduce
    from pathlib import Path

    import numpy as np
    import tqdm
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import transform_matrix
    from pyquaternion import Quaternion

    map_name_from_general_to_detection = {
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.wheelchair': 'ignore',
        'human.pedestrian.stroller': 'ignore',
        'human.pedestrian.personal_mobility': 'ignore',
        'human.pedestrian.police_officer': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'animal': 'ignore',
        'vehicle.car': 'car',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.emergency.ambulance': 'ignore',
        'vehicle.emergency.police': 'ignore',
        'vehicle.trailer': 'trailer',
        'movable_object.barrier': 'barrier',
        'movable_object.trafficcone': 'traffic_cone',
        'movable_object.pushable_pullable': 'ignore',
        'movable_object.debris': 'ignore',
        'static_object.bicycle_rack': 'ignore',
    }


    cls_attr_dist = {
        'barrier': {
            'cycle.with_rider': 0,
            'cycle.without_rider': 0,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 0,
            'vehicle.parked': 0,
            'vehicle.stopped': 0,
        },
        'bicycle': {
            'cycle.with_rider': 2791,
            'cycle.without_rider': 8946,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 0,
            'vehicle.parked': 0,
            'vehicle.stopped': 0,
        },
        'bus': {
            'cycle.with_rider': 0,
            'cycle.without_rider': 0,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 9092,
            'vehicle.parked': 3294,
            'vehicle.stopped': 3881,
        },
        'car': {
            'cycle.with_rider': 0,
            'cycle.without_rider': 0,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 114304,
            'vehicle.parked': 330133,
            'vehicle.stopped': 46898,
        },
        'construction_vehicle': {
            'cycle.with_rider': 0,
            'cycle.without_rider': 0,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 882,
            'vehicle.parked': 11549,
            'vehicle.stopped': 2102,
        },
        'ignore': {
            'cycle.with_rider': 307,
            'cycle.without_rider': 73,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 165,
            'vehicle.parked': 400,
            'vehicle.stopped': 102,
        },
        'motorcycle': {
            'cycle.with_rider': 4233,
            'cycle.without_rider': 8326,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 0,
            'vehicle.parked': 0,
            'vehicle.stopped': 0,
        },
        'pedestrian': {
            'cycle.with_rider': 0,
            'cycle.without_rider': 0,
            'pedestrian.moving': 157444,
            'pedestrian.sitting_lying_down': 13939,
            'pedestrian.standing': 46530,
            'vehicle.moving': 0,
            'vehicle.parked': 0,
            'vehicle.stopped': 0,
        },
        'traffic_cone': {
            'cycle.with_rider': 0,
            'cycle.without_rider': 0,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 0,
            'vehicle.parked': 0,
            'vehicle.stopped': 0,
        },
        'trailer': {
            'cycle.with_rider': 0,
            'cycle.without_rider': 0,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 3421,
            'vehicle.parked': 19224,
            'vehicle.stopped': 1895,
        },
        'truck': {
            'cycle.with_rider': 0,
            'cycle.without_rider': 0,
            'pedestrian.moving': 0,
            'pedestrian.sitting_lying_down': 0,
            'pedestrian.standing': 0,
            'vehicle.moving': 21339,
            'vehicle.parked': 55626,
            'vehicle.stopped': 11097,
        },
    }


    def get_available_scenes(nusc):
        '''
        获取所有的scenes（场景）。
        scene是20s的视频数据，一个scene中有若干个sample（关键帧），
        sample_data是关键帧的数据，包括lidar、camera等各个传感器的数据。
        '''
        available_scenes = []
        print('total scene num:', len(nusc.scene))
        for scene in nusc.scene:
            scene_token = scene['token']
            scene_rec = nusc.get('scene', scene_token)
            sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break
                # if not sd_rec['next'] == '':
                #     sd_rec = nusc.get('sample_data', sd_rec['next'])
                # else:
                #     has_more_frames = False
            if scene_not_exist:
                continue
            available_scenes.append(scene)
        print('exist scene num:', len(available_scenes))
        return available_scenes


    def get_sample_data(nusc, sample_data_token, selected_anntokens=None):
        """
        Returns the data path as well as all annotations related to that sample_data.
        Note that the boxes are transformed into the current sensor's coordinate frame.
        Args:
            nusc:
            sample_data_token: Sample_data token.
            selected_anntokens: If provided only return the selected annotation.

        Returns:

        """
        # Retrieve sensor & pose records
        sd_record = nusc.get('sample_data', sample_data_token)  # 关键帧数据token
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token']) # 校准的传感器记录
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])   # 传感器记录
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token']) # 车辆姿态记录

        data_path = nusc.get_sample_data_path(sample_data_token)

        # 如果是相机数据，获取相机内参
        if sensor_record['modality'] == 'camera':
            cam_intrinsic = np.array(cs_record['camera_intrinsic'])
            imsize = (sd_record['width'], sd_record['height'])
        else:
            cam_intrinsic = imsize = None

        # Retrieve all sample annotations and map to sensor coordinate system.
        # 获取所有的sample annotations，并映射到传感器坐标系
        if selected_anntokens is not None:
            boxes = list(map(nusc.get_box, selected_anntokens))
        else:
            boxes = nusc.get_boxes(sample_data_token)

        # Make list of Box objects including coord system transforms.
        # 生成Box对象列表，包括坐标系变换
        box_list = []
        for box in boxes:
            box.velocity = nusc.box_velocity(box.token)
            # Move box to ego vehicle coord system
            box.translate(-np.array(pose_record['translation']))
            box.rotate(Quaternion(pose_record['rotation']).inverse)

            #  Move box to sensor coord system
            box.translate(-np.array(cs_record['translation']))
            box.rotate(Quaternion(cs_record['rotation']).inverse)

            box_list.append(box)

        return data_path, box_list, cam_intrinsic


    def quaternion_yaw(q: Quaternion) -> float:
        """
        Calculate the yaw angle from a quaternion.
        Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
        It does not work for a box in the camera frame.
        :param q: Quaternion of interest.
        :return: Yaw angle in radians.
        """

        # Project into xy plane.
        v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

        # Measure yaw using arctan.
        yaw = np.arctan2(v[1], v[0])

        return yaw
        

    def obtain_sensor2top(
        nusc, sensor_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, sensor_type="lidar"
    ):
        """Obtain the info with RT matric from general sensor to Top LiDAR.

        Args:
            nusc (class): Dataset class in the nuScenes dataset.
            sensor_token (str): Sample data token corresponding to the
                specific sensor type.
            l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
            l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
                in shape (3, 3).
            e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
            e2g_r_mat (np.ndarray): Rotation matrix from ego to global
                in shape (3, 3).
            sensor_type (str): Sensor to calibrate. Default: 'lidar'.

        Returns:
            sweep (dict): Sweep information after transformation.
        """
        sd_rec = nusc.get("sample_data", sensor_token)
        cs_record = nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = nusc.get("ego_pose", sd_rec["ego_pose_token"])
        data_path = str(nusc.get_sample_data_path(sd_rec["token"]))
        # if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        #     data_path = data_path.split(f"{os.getcwd()}/")[-1]  # relative path
        sweep = {
            "data_path": data_path,
            "type": sensor_type,
            "sample_data_token": sd_rec["token"],
            "sensor2ego_translation": cs_record["translation"],
            "sensor2ego_rotation": cs_record["rotation"],
            "ego2global_translation": pose_record["translation"],
            "ego2global_rotation": pose_record["rotation"],
            "timestamp": sd_rec["timestamp"],
        }
        l2e_r_s = sweep["sensor2ego_rotation"]
        l2e_t_s = sweep["sensor2ego_translation"]
        e2g_r_s = sweep["ego2global_rotation"]
        e2g_t_s = sweep["ego2global_translation"]

        # obtain the RT from sensor to Top LiDAR
        # sweep->ego->global->ego'->lidar
        l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
        e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
        R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        )
        T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
            np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
        )
        T -= (
            e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            + l2e_t @ np.linalg.inv(l2e_r_mat).T
        ).squeeze(0)
        sweep["sensor2lidar_rotation"] = R.T  # points @ R.T + T
        sweep["sensor2lidar_translation"] = T
        return sweep


    def fill_trainval_infos(data_path, nusc, train_scenes, val_scenes, test=False, max_sweeps=10, with_cam=False):
        train_nusc_infos = []
        val_nusc_infos = []
        progress_bar = tqdm.tqdm(total=len(nusc.sample), desc='create_info', dynamic_ncols=True)

        radar_channels = [
            'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT'
        ]

        for index, sample in enumerate(nusc.sample):
            progress_bar.update()

            # 参考通道
            ref_chan = 'RADAR_FRONT'
            chan = radar_channels  # 这里使用所有Radar通道

            ref_sd_token = sample['data'][ref_chan]
            ref_sd_rec = nusc.get('sample_data', ref_sd_token)
            ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
            ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
            ref_time = 1e-6 * ref_sd_rec['timestamp']

            ref_lidar_path, ref_boxes, _ = get_sample_data(nusc, ref_sd_token)

            ref_cam_front_token = sample['data']['CAM_FRONT']
            ref_cam_path, _, ref_cam_intrinsic = nusc.get_sample_data(ref_cam_front_token)

            # Homogeneous transform from ego car frame to reference frame
            ref_from_car = transform_matrix(
                ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True
            )

            # Homogeneous transformation matrix from global to _current_ ego car frame
            car_from_global = transform_matrix(
                ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True,
            )

            info = {
                'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                'cam_front_path': Path(ref_cam_path).relative_to(data_path).__str__(),
                'cam_intrinsic': ref_cam_intrinsic,
                'token': sample['token'],
                'sweeps': [],
                'ref_from_car': ref_from_car,
                'car_from_global': car_from_global,
                'timestamp': ref_time,
            }

            if with_cam:
                info['cams'] = dict()
                l2e_r = ref_cs_rec["rotation"]
                l2e_t = ref_cs_rec["translation"],
                e2g_r = ref_pose_rec["rotation"]
                e2g_t = ref_pose_rec["translation"]
                l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                # obtain 6 image's information per frame
                camera_types = [
                    "CAM_FRONT",
                    "CAM_FRONT_RIGHT",
                    "CAM_FRONT_LEFT",
                    "CAM_BACK",
                    "CAM_BACK_LEFT",
                    "CAM_BACK_RIGHT",
                ]
                for cam in camera_types:
                    cam_token = sample["data"][cam]
                    cam_path, _, camera_intrinsics = nusc.get_sample_data(cam_token)
                    cam_info = obtain_sensor2top(
                        nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam
                    )
                    cam_info['data_path'] = Path(cam_info['data_path']).relative_to(data_path).__str__()
                    cam_info.update(camera_intrinsics=camera_intrinsics)
                    info["cams"].update({cam: cam_info})

            # 获取每个雷达通道的点云特征
            radars_sweeps = []
            for chan in radar_channels:
                sample_data_token = sample['data'][chan]
                curr_sd_rec = nusc.get('sample_data', sample_data_token)
                sweeps = []
                # radar_path, _, _ = get_sample_data(nusc, sample_data_token)
                # radar_path = nusc.get_sample_data_path(curr_sd_rec['token'])
                while len(sweeps) < max_sweeps - 1:
                    if curr_sd_rec['prev'] == '':
                        if len(sweeps) == 0:
                            sweep = {
                                'lidar_path': Path(ref_lidar_path).relative_to(data_path).__str__(),
                                'sample_data_token': curr_sd_rec['token'],
                                'transform_matrix': None,
                                'time_lag': curr_sd_rec['timestamp'] * 0,
                            }
                            sweeps.append(sweep)
                        else:
                            sweeps.append(sweeps[-1])
                    else:
                        curr_sd_rec = nusc.get('sample_data', curr_sd_rec['prev'])

                        # Get past pose
                        current_pose_rec = nusc.get('ego_pose', curr_sd_rec['ego_pose_token'])
                        global_from_car = transform_matrix(
                            current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False,
                        )

                        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
                        current_cs_rec = nusc.get(
                            'calibrated_sensor', curr_sd_rec['calibrated_sensor_token']
                        )
                        car_from_current = transform_matrix(
                            current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False,
                        )

                        tm = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])

                        lidar_path = nusc.get_sample_data_path(curr_sd_rec['token'])

                        time_lag = ref_time - 1e-6 * curr_sd_rec['timestamp']

                        sweep = {
                            'lidar_path': Path(lidar_path).relative_to(data_path).__str__(),
                            'sample_data_token': curr_sd_rec['token'],
                            'transform_matrix': tm,
                            'global_from_car': global_from_car,
                            'car_from_current': car_from_current,
                            'time_lag': time_lag,
                        }
                        sweeps.append(sweep)

                # info['sweeps'] = sweeps
                radars_sweeps.append(sweeps)

                # assert len(info['sweeps']) == max_sweeps - 1, \
                #     f"sweep {curr_sd_rec['token']} only has {len(info['sweeps'])} sweeps, " \
                #     f"you should duplicate to sweep num {max_sweeps - 1}"
                assert len(sweeps) == max_sweeps - 1, \
                    f"sweep {curr_sd_rec['token']} only has {len(sweeps)} sweeps, " \
                    f"you should duplicate to sweep num {max_sweeps - 1}"

            info['sweeps'] = radars_sweeps

            if not test:
                annotations = [nusc.get('sample_annotation', token) for token in sample['anns']]

                # the filtering gives 0.5~1 map improvement
                num_lidar_pts = np.array([anno['num_lidar_pts'] for anno in annotations])
                num_radar_pts = np.array([anno['num_radar_pts'] for anno in annotations])
                mask = (num_lidar_pts + num_radar_pts > 0)

                locs = np.array([b.center for b in ref_boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in ref_boxes]).reshape(-1, 3)[:, [1, 0, 2]]  # wlh == > dxdydz (lwh)
                velocity = np.array([b.velocity for b in ref_boxes]).reshape(-1, 3)
                rots = np.array([quaternion_yaw(b.orientation) for b in ref_boxes]).reshape(-1, 1)
                names = np.array([b.name for b in ref_boxes])
                tokens = np.array([b.token for b in ref_boxes])
                gt_boxes = np.concatenate([locs, dims, rots, velocity[:, :2]], axis=1)

                assert len(annotations) == len(gt_boxes) == len(velocity)

                info['gt_boxes'] = gt_boxes[mask, :]
                info['gt_boxes_velocity'] = velocity[mask, :]
                info['gt_names'] = np.array([map_name_from_general_to_detection[name] for name in names])[mask]
                info['gt_boxes_token'] = tokens[mask]
                info['num_lidar_pts'] = num_lidar_pts[mask]
                info['num_radar_pts'] = num_radar_pts[mask]

            if sample['scene_token'] in train_scenes:
                train_nusc_infos.append(info)
            else:
                val_nusc_infos.append(info)

        progress_bar.close()
        return train_nusc_infos, val_nusc_infos


    def boxes_lidar_to_nusenes(det_info):
        boxes3d = det_info['boxes_lidar']
        scores = det_info['score']
        labels = det_info['pred_labels']

        box_list = []
        for k in range(boxes3d.shape[0]):
            quat = Quaternion(axis=[0, 0, 1], radians=boxes3d[k, 6])
            velocity = (*boxes3d[k, 7:9], 0.0) if boxes3d.shape[1] == 9 else (0.0, 0.0, 0.0)
            box = Box(
                boxes3d[k, :3],
                boxes3d[k, [4, 3, 5]],  # wlh
                quat, label=labels[k], score=scores[k], velocity=velocity,
            )
            box_list.append(box)
        return box_list


    def lidar_nusc_box_to_global(nusc, boxes, sample_token):
        s_record = nusc.get('sample', sample_token)
        sample_data_token = s_record['data']['LIDAR_TOP']

        sd_record = nusc.get('sample_data', sample_data_token)
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        sensor_record = nusc.get('sensor', cs_record['sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        data_path = nusc.get_sample_data_path(sample_data_token)
        box_list = []
        for box in boxes:
            # Move box to ego vehicle coord system
            box.rotate(Quaternion(cs_record['rotation']))
            box.translate(np.array(cs_record['translation']))
            # Move box to global coord system
            box.rotate(Quaternion(pose_record['rotation']))
            box.translate(np.array(pose_record['translation']))
            box_list.append(box)
        return box_list


    def transform_det_annos_to_nusc_annos(det_annos, nusc):
        nusc_annos = {
            'results': {},
            'meta': None,
        }

        for det in det_annos:
            annos = []
            box_list = boxes_lidar_to_nusenes(det)
            box_list = lidar_nusc_box_to_global(
                nusc=nusc, boxes=box_list, sample_token=det['metadata']['token']
            )

            for k, box in enumerate(box_list):
                name = det['name'][k]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in ['car', 'construction_vehicle', 'bus', 'truck', 'trailer']:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = None
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = None
                attr = attr if attr is not None else max(
                    cls_attr_dist[name].items(), key=operator.itemgetter(1))[0]
                nusc_anno = {
                    'sample_token': det['metadata']['token'],
                    'translation': box.center.tolist(),
                    'size': box.wlh.tolist(),
                    'rotation': box.orientation.elements.tolist(),
                    'velocity': box.velocity[:2].tolist(),
                    'detection_name': name,
                    'detection_score': box.score,
                    'attribute_name': attr
                }
                annos.append(nusc_anno)

            nusc_annos['results'].update({det["metadata"]["token"]: annos})

        return nusc_annos


    def format_nuscene_results(metrics, class_names, version='default'):
        result = '----------------Nuscene %s results-----------------\n' % version
        for name in class_names:
            threshs = ', '.join(list(metrics['label_aps'][name].keys()))
            ap_list = list(metrics['label_aps'][name].values())

            err_name =', '.join([x.split('_')[0] for x in list(metrics['label_tp_errors'][name].keys())])
            error_list = list(metrics['label_tp_errors'][name].values())

            result += f'***{name} error@{err_name} | AP@{threshs}\n'
            result += ', '.join(['%.2f' % x for x in error_list]) + ' | '
            result += ', '.join(['%.2f' % (x * 100) for x in ap_list])
            result += f" | mean AP: {metrics['mean_dist_aps'][name]}"
            result += '\n'

        result += '--------------average performance-------------\n'
        details = {}
        for key, val in metrics['tp_errors'].items():
            result += '%s:\t %.4f\n' % (key, val)
            details[key] = val

        result += 'mAP:\t %.4f\n' % metrics['mean_ap']
        result += 'NDS:\t %.4f\n' % metrics['nd_score']

        details.update({
            'mAP': metrics['mean_ap'],
            'NDS': metrics['nd_score'],
        })

        return result, details

   ```
2. 修改`nuscenes_dataset.py`：
   
   ```python
    import copy
    import pickle
    import struct
    from pathlib import Path

    import numpy as np
    from tqdm import tqdm

    from ...ops.roiaware_pool3d import roiaware_pool3d_utils
    from ...utils import common_utils
    from ..dataset import DatasetTemplate
    from pyquaternion import Quaternion
    from PIL import Image


    class NuScenesDataset(DatasetTemplate):
        def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
            root_path = (root_path if root_path is not None else Path(dataset_cfg.DATA_PATH)) / dataset_cfg.VERSION
            super().__init__(
                dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
            )
            self.infos = []
            self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
            if self.camera_config is not None:
                self.use_camera = self.camera_config.get('USE_CAMERA', True)
                self.camera_image_config = self.camera_config.IMAGE
            else:
                self.use_camera = False

            self.include_nuscenes_data(self.mode)
            if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING', False):
                self.infos = self.balanced_infos_resampling(self.infos)

        def include_nuscenes_data(self, mode):
            self.logger.info('Loading NuScenes dataset')
            nuscenes_infos = []

            for info_path in self.dataset_cfg.INFO_PATH[mode]:
                info_path = self.root_path / info_path
                if not info_path.exists():
                    continue
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    nuscenes_infos.extend(infos)

            self.infos.extend(nuscenes_infos)
            self.logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

        def balanced_infos_resampling(self, infos):
            """
            Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
            """
            if self.class_names is None:
                return infos

            cls_infos = {name: [] for name in self.class_names}
            for info in infos:
                for name in set(info['gt_names']):
                    if name in self.class_names:
                        cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
            cls_dist = {k: len(v) / duplicated_samples for k, v in cls_infos.items()}

            sampled_infos = []

            frac = 1.0 / len(self.class_names)
            ratios = [frac / v for v in cls_dist.values()]

            for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
                sampled_infos += np.random.choice(
                    cur_cls_infos, int(len(cur_cls_infos) * ratio)
                ).tolist()
            self.logger.info('Total samples after balanced resampling: %s' % (len(sampled_infos)))

            cls_infos_new = {name: [] for name in self.class_names}
            for info in sampled_infos:
                for name in set(info['gt_names']):
                    if name in self.class_names:
                        cls_infos_new[name].append(info)

            cls_dist_new = {k: len(v) / len(sampled_infos) for k, v in cls_infos_new.items()}

            return sampled_infos

        def get_radar_points_sweep(self, lidar_path):
            ############# specified for Radar data #################
            assert str(lidar_path).endswith('.pcd'), 'Unsupported filetype {}'.format(str(lidar_path))
            meta = []
            with open(str(lidar_path), 'rb') as f:
                for line in f:
                    line = line.strip().decode('utf-8')
                    meta.append(line)
                    if line.startswith('DATA'):
                        break

                data_binary = f.read()

            # Get the header rows and check if they appear as expected.
            assert meta[0].startswith('#'), 'First line must be comment'
            assert meta[1].startswith('VERSION'), 'Second line must be VERSION'
            sizes = meta[3].split(' ')[1:]
            types = meta[4].split(' ')[1:]
            counts = meta[5].split(' ')[1:]
            width = int(meta[6].split(' ')[1])
            height = int(meta[7].split(' ')[1])
            data = meta[10].split(' ')[1]
            feature_count = len(types)
            assert width > 0
            assert len([c for c in counts if c != c]) == 0, 'Error: COUNT not supported!'
            assert height == 1, 'Error: height != 0 not supported!'
            assert data == 'binary'

            # Lookup table for how to decode the binaries.
            unpacking_lut = {'F': {2: 'e', 4: 'f', 8: 'd'},
                            'I': {1: 'b', 2: 'h', 4: 'i', 8: 'q'},
                            'U': {1: 'B', 2: 'H', 4: 'I', 8: 'Q'}}
            types_str = ''.join([unpacking_lut[t][int(s)] for t, s in zip(types, sizes)])

            # Decode each point.
            offset = 0
            point_count = width
            points = []
            for i in range(point_count):
                point = []
                for p in range(feature_count):
                    start_p = offset
                    end_p = start_p + int(sizes[p])
                    assert end_p < len(data_binary)
                    point_p = struct.unpack(types_str[p], data_binary[start_p:end_p])[0]
                    point.append(point_p)
                    offset = end_p
                points.append(point)

            # A NaN in the first point indicates an empty pointcloud.
            point = np.array(points[0])
            if np.any(np.isnan(point)):
                points = np.zeros((0, feature_count))

            # Convert to numpy matrix.
            points = np.array(points).transpose()

            invalid_states = [0]
            dynprop_states = range(7)
            ambig_states = [3]

            # Filter points with an invalid state.
            valid = [p in invalid_states for p in points[-4, :]]
            points = points[:, valid]

            # Filter by dynProp.
            valid = [p in dynprop_states for p in points[3, :]]
            points = points[:, valid]

            # Filter by ambig_state.
            valid = [p in ambig_states for p in points[11, :]]
            points = points[:, valid]

            return points
            ###################################################

        def get_sweep(self, sweep_info):
            # def remove_ego_points(points, center_radius=1.0):
            #     mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
            #     return points[mask]

            def remove_ego_points(points, center_radius=1.0):
                mask = ~((np.abs(points[0, :]) < center_radius) & (np.abs(points[1, :]) < center_radius))
                return points.T[mask].T

            lidar_path = self.root_path / sweep_info['lidar_path']
            # points_sweep = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

            points_sweep = self.get_radar_points_sweep(lidar_path)

            points_sweep = remove_ego_points(points_sweep)
            if sweep_info['transform_matrix'] is not None:
                num_points = points_sweep.shape[1]
                points_sweep[:3, :] = sweep_info['transform_matrix'].dot(
                    np.vstack((points_sweep[:3, :], np.ones(num_points))))[:3, :]

            cur_times = sweep_info['time_lag'] * np.ones((1, points_sweep.shape[1]))
            return points_sweep.T, cur_times.T

        def get_lidar_with_sweeps(self, index, max_sweeps=1):
            info = self.infos[index]
            lidar_path = self.root_path / info['lidar_path']
            # points = np.fromfile(str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :4]

            points = self.get_radar_points_sweep(lidar_path).T
            # print(points.shape)

            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]

            for radar_sweeps in info['sweeps']:
                for k in np.random.choice(len(radar_sweeps), max_sweeps - 1, replace=False):
                    points_sweep, times_sweep = self.get_sweep(radar_sweeps[k])
                    # print(points_sweep.shape)
                    sweep_points_list.append(points_sweep)
                    sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            points = np.concatenate((points, times), axis=1)
            return points

        def crop_image(self, input_dict):
            W, H = input_dict["ori_shape"]
            imgs = input_dict["camera_imgs"]
            img_process_infos = []
            crop_images = []
            for img in imgs:
                if self.training == True:
                    fH, fW = self.camera_image_config.FINAL_DIM
                    resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                    resize = np.random.uniform(*resize_lim)
                    resize_dims = (int(W * resize), int(H * resize))
                    newW, newH = resize_dims
                    crop_h = newH - fH
                    crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
                else:
                    fH, fW = self.camera_image_config.FINAL_DIM
                    resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                    resize = np.mean(resize_lim)
                    resize_dims = (int(W * resize), int(H * resize))
                    newW, newH = resize_dims
                    crop_h = newH - fH
                    crop_w = int(max(0, newW - fW) / 2)
                    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
                
                # reisze and crop image
                img = img.resize(resize_dims)
                img = img.crop(crop)
                crop_images.append(img)
                img_process_infos.append([resize, crop, False, 0])
            
            input_dict['img_process_infos'] = img_process_infos
            input_dict['camera_imgs'] = crop_images
            return input_dict
        
        def load_camera_info(self, input_dict, info):
            input_dict["image_paths"] = []
            input_dict["lidar2camera"] = []
            input_dict["lidar2image"] = []
            input_dict["camera2ego"] = []
            input_dict["camera_intrinsics"] = []
            input_dict["camera2lidar"] = []

            for _, camera_info in info["cams"].items():
                input_dict["image_paths"].append(camera_info["data_path"])

                # lidar to camera transform
                lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
                lidar2camera_t = (
                    camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
                )
                lidar2camera_rt = np.eye(4).astype(np.float32)
                lidar2camera_rt[:3, :3] = lidar2camera_r.T
                lidar2camera_rt[3, :3] = -lidar2camera_t
                input_dict["lidar2camera"].append(lidar2camera_rt.T)

                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
                input_dict["camera_intrinsics"].append(camera_intrinsics)

                # lidar to image transform
                lidar2image = camera_intrinsics @ lidar2camera_rt.T
                input_dict["lidar2image"].append(lidar2image)

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(
                    camera_info["sensor2ego_rotation"]
                ).rotation_matrix
                camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
                input_dict["camera2ego"].append(camera2ego)

                # camera to lidar transform
                camera2lidar = np.eye(4).astype(np.float32)
                camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
                camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
                input_dict["camera2lidar"].append(camera2lidar)
            # read image
            filename = input_dict["image_paths"]
            images = []
            for name in filename:
                images.append(Image.open(str(self.root_path / name)))
            
            input_dict["camera_imgs"] = images
            input_dict["ori_shape"] = images[0].size
            
            # resize and crop image
            input_dict = self.crop_image(input_dict)

            return input_dict

        def __len__(self):
            if self._merge_all_iters_to_one_epoch:
                return len(self.infos) * self.total_epochs

            return len(self.infos)

        def __getitem__(self, index):
            if self._merge_all_iters_to_one_epoch:
                index = index % len(self.infos)

            info = copy.deepcopy(self.infos[index])
            points = self.get_lidar_with_sweeps(index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)

            input_dict = {
                'points': points,
                'frame_id': Path(info['lidar_path']).stem,
                'metadata': {'token': info['token']}
            }

            if 'gt_boxes' in info:
                if self.dataset_cfg.get('FILTER_MIN_POINTS_IN_GT', False):
                    mask = (info['num_lidar_pts'] > self.dataset_cfg.FILTER_MIN_POINTS_IN_GT - 1)
                else:
                    mask = None

                input_dict.update({
                    'gt_names': info['gt_names'] if mask is None else info['gt_names'][mask],
                    'gt_boxes': info['gt_boxes'] if mask is None else info['gt_boxes'][mask]
                })
            if self.use_camera:
                input_dict = self.load_camera_info(input_dict, info)

            data_dict = self.prepare_data(data_dict=input_dict)

            if self.dataset_cfg.get('SET_NAN_VELOCITY_TO_ZEROS', False) and 'gt_boxes' in info:
                gt_boxes = data_dict['gt_boxes']
                gt_boxes[np.isnan(gt_boxes)] = 0
                data_dict['gt_boxes'] = gt_boxes

            if not self.dataset_cfg.PRED_VELOCITY and 'gt_boxes' in data_dict:
                data_dict['gt_boxes'] = data_dict['gt_boxes'][:, [0, 1, 2, 3, 4, 5, 6, -1]]

            return data_dict

        def evaluation(self, det_annos, class_names, **kwargs):
            import json
            from nuscenes.nuscenes import NuScenes
            from . import nuscenes_utils
            nusc = NuScenes(version=self.dataset_cfg.VERSION, dataroot=str(self.root_path), verbose=True)
            nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
            nusc_annos['meta'] = {
                'use_camera': False,
                'use_lidar': False,
                'use_radar': True,
                'use_map': False,
                'use_external': False,
            }

            output_path = Path(kwargs['output_path'])
            output_path.mkdir(exist_ok=True, parents=True)
            res_path = str(output_path / 'results_nusc.json')
            with open(res_path, 'w') as f:
                json.dump(nusc_annos, f)

            self.logger.info(f'The predictions of NuScenes have been saved to {res_path}')

            if self.dataset_cfg.VERSION == 'v1.0-test':
                return 'No ground-truth annotations for evaluation', {}

            from nuscenes.eval.detection.config import config_factory
            from nuscenes.eval.detection.evaluate import NuScenesEval

            eval_set_map = {
                'v1.0-mini': 'mini_val',
                'v1.0-trainval': 'val',
                'v1.0-test': 'test'
            }
            try:
                eval_version = 'detection_cvpr_2019'
                eval_config = config_factory(eval_version)
            except:
                eval_version = 'cvpr_2019'
                eval_config = config_factory(eval_version)

            nusc_eval = NuScenesEval(
                nusc,
                config=eval_config,
                result_path=res_path,
                eval_set=eval_set_map[self.dataset_cfg.VERSION],
                output_dir=str(output_path),
                verbose=True,
            )
            metrics_summary = nusc_eval.main(plot_examples=0, render_curves=False)

            with open(output_path / 'metrics_summary.json', 'r') as f:
                metrics = json.load(f)

            result_str, result_dict = nuscenes_utils.format_nuscene_results(metrics, self.class_names, version=eval_version)
            return result_str, result_dict

        def create_groundtruth_database(self, used_classes=None, max_sweeps=10):
            import torch

            database_save_path = self.root_path / f'gt_database_{max_sweeps}sweeps_withvelo'
            db_info_save_path = self.root_path / f'nuscenes_dbinfos_{max_sweeps}sweeps_withvelo.pkl'

            database_save_path.mkdir(parents=True, exist_ok=True)
            all_db_infos = {}

            for idx in tqdm(range(len(self.infos))):
                sample_idx = idx
                info = self.infos[idx]
                points = self.get_lidar_with_sweeps(idx, max_sweeps=max_sweeps)
                gt_boxes = info['gt_boxes']
                gt_names = info['gt_names']

                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()

                for i in range(gt_boxes.shape[0]):
                    filename = '%s_%s_%d.bin' % (sample_idx, gt_names[i], i)
                    filepath = database_save_path / filename
                    gt_points = points[box_idxs_of_pts == i]

                    gt_points[:, :3] -= gt_boxes[i, :3]
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    if (used_classes is None) or gt_names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        db_info = {'name': gt_names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
                        if gt_names[i] in all_db_infos:
                            all_db_infos[gt_names[i]].append(db_info)
                        else:
                            all_db_infos[gt_names[i]] = [db_info]
            for k, v in all_db_infos.items():
                print('Database %s: %d' % (k, len(v)))

            with open(db_info_save_path, 'wb') as f:
                pickle.dump(all_db_infos, f)


    def create_nuscenes_info(version, data_path, save_path, max_sweeps=10, with_cam=False):
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils import splits
        from . import nuscenes_utils
        data_path = data_path / version
        save_path = save_path / version

        assert version in ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
        if version == 'v1.0-trainval':
            train_scenes = splits.train
            val_scenes = splits.val
        elif version == 'v1.0-test':
            train_scenes = splits.test
            val_scenes = []
        elif version == 'v1.0-mini':
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise NotImplementedError

        nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
        available_scenes = nuscenes_utils.get_available_scenes(nusc)
        available_scene_names = [s['name'] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in train_scenes])
        val_scenes = set([available_scenes[available_scene_names.index(s)]['token'] for s in val_scenes])

        print('%s: train scene(%d), val scene(%d)' % (version, len(train_scenes), len(val_scenes)))

        train_nusc_infos, val_nusc_infos = nuscenes_utils.fill_trainval_infos(
            data_path=data_path, nusc=nusc, train_scenes=train_scenes, val_scenes=val_scenes,
            test='test' in version, max_sweeps=max_sweeps, with_cam=with_cam
        )

        if version == 'v1.0-test':
            print('test sample: %d' % len(train_nusc_infos))
            with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_test.pkl', 'wb') as f:
                pickle.dump(train_nusc_infos, f)
        else:
            print('train sample: %d, val sample: %d' % (len(train_nusc_infos), len(val_nusc_infos)))
            with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_train.pkl', 'wb') as f:
                pickle.dump(train_nusc_infos, f)
            with open(save_path / f'nuscenes_infos_{max_sweeps}sweeps_val.pkl', 'wb') as f:
                pickle.dump(val_nusc_infos, f)


    if __name__ == '__main__':
        import yaml
        import argparse
        from pathlib import Path
        from easydict import EasyDict

        parser = argparse.ArgumentParser(description='arg parser')
        parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
        parser.add_argument('--func', type=str, default='create_nuscenes_infos', help='')
        parser.add_argument('--version', type=str, default='v1.0-trainval', help='')
        parser.add_argument('--with_cam', action='store_true', default=False, help='use camera or not')
        args = parser.parse_args()

        if args.func == 'create_nuscenes_infos':
            dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))
            ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
            dataset_cfg.VERSION = args.version
            create_nuscenes_info(
                version=dataset_cfg.VERSION,
                data_path=ROOT_DIR / 'data' / 'nuscenes',
                save_path=ROOT_DIR / 'data' / 'nuscenes',
                max_sweeps=dataset_cfg.MAX_SWEEPS,
                with_cam=args.with_cam
            )

            nuscenes_dataset = NuScenesDataset(
                dataset_cfg=dataset_cfg, class_names=None,
                root_path=ROOT_DIR / 'data' / 'nuscenes',
                logger=common_utils.create_logger(), training=True
            )
            nuscenes_dataset.create_groundtruth_database(max_sweeps=dataset_cfg.MAX_SWEEPS)

   ```
3. 配置`nuscenes_dataset.yaml`：
   ```yaml
    DATASET: 'NuScenesDataset'
    DATA_PATH: './data/nuscenes'

    VERSION: 'v1.0-trainval'
    MAX_SWEEPS: 10
    PRED_VELOCITY: True
    SET_NAN_VELOCITY_TO_ZEROS: True
    FILTER_MIN_POINTS_IN_GT: 1

    DATA_SPLIT: {
        'train': train,
        'test': val
    }

    INFO_PATH: {
        'train': [nuscenes_infos_10sweeps_train.pkl],
        'test': [nuscenes_infos_10sweeps_val.pkl],
    }

    POINT_CLOUD_RANGE: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

    BALANCED_RESAMPLING: True 

    DATA_AUGMENTOR:
        DISABLE_AUG_LIST: ['placeholder']
        AUG_CONFIG_LIST:
            - NAME: gt_sampling
            DB_INFO_PATH:
                - nuscenes_dbinfos_10sweeps_withvelo.pkl
            PREPARE: {
                filter_by_min_points: [
                    'car:5','truck:5', 'construction_vehicle:5', 'bus:5', 'trailer:5',
                    'barrier:5', 'motorcycle:5', 'bicycle:5', 'pedestrian:5', 'traffic_cone:5'
                ],
            }

            SAMPLE_GROUPS: [
                'car:2','truck:3', 'construction_vehicle:7', 'bus:4', 'trailer:6',
                'barrier:2', 'motorcycle:6', 'bicycle:6', 'pedestrian:2', 'traffic_cone:2'
            ]

            NUM_POINT_FEATURES: 19
            DATABASE_WITH_FAKELIDAR: False
            REMOVE_EXTRA_WIDTH: [0.0, 0.0, 0.0]
            LIMIT_WHOLE_SCENE: True

            - NAME: random_world_flip
            ALONG_AXIS_LIST: ['x', 'y']

            - NAME: random_world_rotation
            WORLD_ROT_ANGLE: [-0.3925, 0.3925]

            - NAME: random_world_scaling
            WORLD_SCALE_RANGE: [0.95, 1.05]


    POINT_FEATURE_ENCODING: {
        encoding_type: absolute_coordinates_encoding,
        # used_feature_list: ['x', 'y', 'z', 'intensity', 'timestamp'],
        # src_feature_list: ['x', 'y', 'z', 'intensity', 'timestamp'],
        used_feature_list: ['x', 'y', 'z', 'rcs', 'vx', 'vy', 'vx_comp', 
                        'vy_comp', 'x_rms', 'y_rms', 'vx_rms', 'vy_rms'],
        src_feature_list: ['x', 'y', 'z', 'dyn_prop', 'id',
                        'rcs', 'vx', 'vy', 'vx_comp', 'vy_comp', 
                        'is_quality_valid', 'ambig_state', 'x_rms', 
                        'y_rms', 'invalid_state', 'pdh0', 'vx_rms', 
                        'vy_rms', 'timestamp'],
    }


    DATA_PROCESSOR:
        - NAME: mask_points_and_boxes_outside_range
        REMOVE_OUTSIDE_BOXES: True

        - NAME: shuffle_points
        SHUFFLE_ENABLED: {
            'train': True,
            'test': True
        }

        - NAME: transform_points_to_voxels
        VOXEL_SIZE: [0.1, 0.1, 0.2]
        MAX_POINTS_PER_VOXEL: 10
        MAX_NUMBER_OF_VOXELS: {
            'train': 60000,
            'test': 60000
        }

   ```
