import cv2
import scipy.io as io


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
