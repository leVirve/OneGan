import logging
import threading

import cv2
import scipy.io as io


log = logging.getLogger('onegan.io')


def save_mat(name, data):
    """ Save data into *.mat file
    """
    return io.savemat(name, data)


def load_mat(name):
    """ load data from *.mat file
    """
    return io.loadmat(name)


class InputStream:

    def __init__(self, stream):
        self.reader = cv2.VideoCapture(stream)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            ret, frame = self.reader.read()
            if ret:
                return frame
            raise StopIteration

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.reader.release()


class WebcamCaptureAsync:

    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

        self.num_capture = 0
        self.num_read = 0

    def start(self):
        if self.started:
            log.warn('Asynchronous video capturing has already been started.')
            return None
        log.info('Asynchronous video capturing started')
        self.started = True
        self.thread = threading.Thread(target=self.update)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.num_capture += 1

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
            self.num_read += 1
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()
        log.info('Asynchronous video capturing stopped')

    def __enter__(self):
        return self

    def __exit__(self, exec_type, exc_value, traceback):
        self.stop()
        self.cap.release()
