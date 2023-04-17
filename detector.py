import torch
import numpy as np
import cv2

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer


class Detector:
    def __init__(self):
        # 指定模型的配置文件和 checkpoint 文件路径
        config_file = 'configs/rtmdet_tiny_8xb32-300e_coco.py'
        checkpoint_file = 'checkpoints/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # init detector and visualizer
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.visualizer = DetLocalVisualizer()
    
    def __del__(self):
        self.camera.release()

    def open_video(self, file_path):
        assert file_path is not None
        self.video = mmcv.VideoReader(file_path)
        self.detec_video = True
    
    def open_camera(self, camera_id = 0):
        self.camera = cv2.VideoCapture(camera_id)
        self.detec_video = False

    def read_video(self) -> np.ndarray:
        frame = self.video.read()
        if frame is None:
            raise Exception('Fail to read video!')
        return frame

    def read_camera(self) -> np.ndarray:
        _, frame = self.camera.read()
        if frame is None:
            raise Exception('Fail to read camera!')
        return frame

    def detect(self):
        """ This function will read a frame and detect objects.
            It will finally return a ndarray.
        Returns:
            _type_: numpy.ndarray
        """
        frame = self.read_video() if self.detec_video else self.read_camera()

        results = inference_detector(self.model, frame)
        self.visualizer.add_datasample(
            name='video_detector',
            image=frame,
            data_sample=results,
            draw_gt=False,
            show=False
        )
        res_img = self.visualizer.get_image()
        return res_img
        
