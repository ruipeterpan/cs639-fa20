import cv2
import numpy as np

class Tracker:
    """Implementation for object tracker"""

    def __init__(self):
        """Initialize tracker members"""
        self.x, self.y, self.width, self.height = 0, 0, 0, 0
        self.track_window = None
        self.roi = None
        self.hsv_roi = None
        self.mask = None
        self.roi_hist = None
        self.term_crit = None
        self.isFirstIter = True

    def track(self, frame):
        frame_float32 = np.float32(frame)

        if self.isFirstIter:
            # initialization, set initial tracking region & calculate histogram
            self.isFirstIter = False
            # self.x, self.y, self.width, self.height = cv2.selectROI(frame, False)
            self.x, self.y, self.width, self.height = 100, 100, 100, 100
            self.track_window = (self.x, self.y, self.width, self.height)

            # set up the Region of
            # Interest for tracking
            self.roi = frame[
                self.y : self.y + self.height, self.x : self.x + self.width
            ]

            # convert ROI from BGR to
            # HSV format
            self.hsv_roi = cv2.cvtColor(frame_float32, cv2.COLOR_BGR2HSV)

            # perform masking operation
            self.mask = cv2.inRange(
                self.hsv_roi, np.array((0.0, 60.0, 40.0)), np.array((180.0, 255.0, 255))
            )

            self.roi_hist = cv2.calcHist(
                [self.hsv_roi], [0], self.mask, [180], [0, 180]
            )

            cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

            # Setup the termination criteria,
            # either 15 iteration or move by
            # atleast 2 pt
            self.term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 2)

        _, self.frame1 = cv2.threshold(frame_float32, 180, 155, cv2.THRESH_TOZERO_INV)

        # convert from BGR to HSV
        # format.
        self.hsv = cv2.cvtColor(self.frame1, cv2.COLOR_BGR2HSV)

        self.dst = cv2.calcBackProject([self.hsv], [0], self.roi_hist, [0, 180], 1)

        # apply Camshift to get the
        # new location            self.dst, self.track_window, self.term_crit

        self.ret2, self.track_window = cv2.CamShift(
            self.dst, self.track_window, self.term_crit
        )

        # Draw it on image
        pts = cv2.boxPoints(
            self.ret2
        )  # something like ((200.0, 472.0), (200.0, 228.0), (420.0, 228.0), (420.0, 472.0))

        # convert from floating
        # to integer
        pts = np.int0(pts)

        # Draw Tracking window on the
        # video frame.
        output = cv2.polylines(frame_float32, [pts], True, (0, 255, 255), 10)

        cv2.imwrite('./test.png', output)

        return pts, output
