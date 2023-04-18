import argparse
import sys
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from main_window import Ui_MainWindow

from detector import Detector


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # detector
        self.detector = Detector(*args)
        # qt gui
        self.timer = QtCore.QTimer(self) # 定时器，用于播放视频
        self.timer.timeout.connect(self.refreshFrame)
        self.import_btn.clicked.connect(self.openVideo)
        self.start_btn.clicked.connect(self.playVideo)
        self.stop_btn.clicked.connect(self.pauseVideo)

    def openVideo(self):
        file_path, _ = QFileDialog.getOpenFileName(self, '选择文件', './', 
                                                   '视频文件(*.mp4 *.mkv *.mov *.avi *.flv)')
        self.file_path.setText(file_path)  # self.file_path inherit Ui_MainWindow.fila_path
        self.detector.open_video(file_path)
    
    def openCamera(self):
        # TODO: add a button to open camera
        pass

    def playVideo(self):
        self.timer.start(50)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def pauseVideo(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def refreshFrame(self):
        res_img = self.detector.detect()
        q_image = QImage(res_img,
                         res_img.shape[1],
                         res_img.shape[0],
                         QImage.Format_RGB888
                        ).rgbSwapped().scaled(self.label.width(),
                                              self.label.height())
        self.label.setPixmap(QPixmap.fromImage(q_image))


def main():
    parser = argparse.ArgumentParser(description='video detector args')
    parser.add_argument('config', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('device', type=str)
    args = parser.parse_args()
    # print(args)

    app = QApplication(sys.argv)
    window = MainWindow(args.config, args.checkpoint, args.device)
    window.show()
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()
    