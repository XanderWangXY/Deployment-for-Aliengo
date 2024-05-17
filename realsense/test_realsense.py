import pyrealsense2 as rs
import numpy as np
import cv2

if __name__ == "__main__":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    #config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Start streaming
    pipeline.start(config)

    dec_filter=rs.decimation_filter()
    dec_filter.set_option(rs.option.filter_magnitude, 7)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            #color_frame = frames.get_color_frame()
            #if not depth_frame or not color_frame:
            #    continue
            # Convert images to numpy arrays
            decimated_depth = dec_filter.process(depth_frame)
            print('decimated_depth分辨率：{}'.format(np.asanyarray(decimated_depth.get_data()).shape))
            #colorized_depth = np.asanyarray(colorizer.colorize(decimated_depth).get_data())

            depth_image = np.asanyarray(decimated_depth.get_data())
            #print(depth_image)
            #color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            #images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', depth_colormap)

            depth_image_clip = depth_image[5:63]
            depth_image_clip = depth_image_clip.T[2:89]
            depth_image_clip = depth_image_clip.T
            print('裁减后分辨率：{}'.format(np.asanyarray(depth_image_clip).shape))
            depth_colormap_clip = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_clip, alpha=0.03), cv2.COLORMAP_JET)

            #cv2.namedWindow('裁减后图像', cv2.WINDOW_AUTOSIZE)
            #cv2.imshow('裁减后图像', depth_colormap_clip)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        # Stop streaming
        pipeline.stop()

