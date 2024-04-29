import cv2
from realsense.realsense import RealSense

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
