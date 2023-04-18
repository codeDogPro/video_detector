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
        return frame

    def read_camera(self) -> np.ndarray:
        _, frame = self.camera.read()
        return frame

    def set_score_thr(self, thr):
        # TODO:设置让visualizer只显示>=thr的bboxes
        pass

    def gen_obj_list(self, results):
        labels = results.get('pred_instances')['labels'].cpu().data.numpy()
        scores = results.get('pred_instances')['scores'].cpu().data.numpy()
        bboxes = results.get('pred_instances')['bboxes'].cpu().data.numpy()
        classes = self.model.dataset_meta['classes']  # need to change
        objects = []
        for i, label_id in enumerate(labels):
            label_txt = classes[label_id]
            objects.append((label_txt, scores[i], bboxes[i]))
        return objects

    def detect(self):
        """ This function will read a frame and detect objects.
        Returns:
            It will finally return a img(ndarray) and object list.
            _type_: numpy.ndarray, List
        """
        frame = self.read_video() if self.detec_video else self.read_camera()
        if frame is None:
            return None, None

        results = inference_detector(self.model, frame) 
        self.visualizer.add_datasample(
            name='video_detector',
            image=frame,
            data_sample=results,
            draw_gt=False,
            show=False
        )
        res_img = self.visualizer.get_image()
        objects = self.gen_obj_list(results)
        return res_img, objects
    
