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

        # decimation_filter
        self.dec_filter = rs.decimation_filter()
        self.dec_filter.set_option(rs.option.filter_magnitude, 7)

    def get_depth(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        #color_frame = frames.get_color_frame()
        #if not depth_frame or not color_frame:
        #    continue
        # Convert images to numpy arrays

        decimated_depth = self.dec_filter.process(depth_frame)
        #print('decimated_depth分辨率：{}'.format(np.asanyarray(decimated_depth.get_data()).shape))
        depth_image = np.asanyarray(decimated_depth.get_data())

        #color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Stack both images horizontally
        #images = np.hstack((color_image, depth_colormap))
        # Show images

        #cv2.namedWindow('before clip', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('before clip', depth_colormap)

        depth_image_clip = depth_image[5:63]
        depth_image_clip = depth_image_clip.T[2:89]
        self.depth_image_clip = depth_image_clip.T
        #print('裁减后分辨率：{}'.format(np.asanyarray(self.depth_image_clip).shape))
        #depth_colormap_clip = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image_clip, alpha=0.03), cv2.COLORMAP_JET)

        #cv2.namedWindow('after clip', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('after clip', depth_colormap_clip)


    def stop(self):
        self.pipeline.stop()


if __name__ == '__main__':
    realsense = RealSense()
    try:
        while True:
            realsense.get_depth()
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    except:
        realsense.stop()