import pyrealsense2 as rs
import numpy as np
import cv2


class RealSense:
    def __init__(self):
        # Configure depth and color streams
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        #self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # Start streaming
        self.pipeline.start(self.config)

    def get_depth(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        #color_frame = frames.get_color_frame()
        #if not depth_frame or not color_frame:
        #    continue
        # Convert images to numpy arrays

        depth_image = np.asanyarray(depth_frame.get_data())

        #color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Stack both images horizontally
        #images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', depth_colormap)


    def stop(self):
        self.pipeline.stop()
