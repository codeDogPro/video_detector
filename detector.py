import torch
import numpy as np
import cv2

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.visualization import DetLocalVisualizer


class Detector:
    def __init__(self, config, checkpoint, device):
        # init detector and visualizer
        self.model = init_detector(config, checkpoint, device=device)
        self.visualizer = DetLocalVisualizer()
        # label's real name
        self.visualizer.dataset_meta = self.model.dataset_meta
    
    def __del__(self):
        if self.camera is not None:
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
        
