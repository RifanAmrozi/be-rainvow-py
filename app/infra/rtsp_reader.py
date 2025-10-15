import cv2

class RtspReader:
    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url)

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def read_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
