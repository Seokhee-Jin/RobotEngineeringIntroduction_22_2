import cv2
import os
import numpy as np


class Converters:
    @staticmethod
    def before_after_test(image, converter):
        cv2.imshow('before', image)
        cv2.imshow(f"after '{converter.__name__}'", converter(image))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def convert_to_gray_scale(image):
        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return gray_img

    @staticmethod
    def convert_to_gaussian_blur(image):
        blur_img = cv2.GaussianBlur(image, (3, 3), 0)

        return blur_img

    @staticmethod
    def convert_to_canny(image):
        canny_img = cv2.Canny(image, 70, 210)

        return canny_img

    @staticmethod
    def convert_to_ROI(image):

        def region_of_interest(image, vertices, color3=(255, 255, 255), color1=255):
            mask = np.zeros_like(image)
            if len(image.shape) > 2:
                color = color3
            else:
                color = color1

            cv2.fillPoly(mask, vertices, color)
            ROI_image = cv2.bitwise_and(image, mask)

            return ROI_image

        height, width = image.shape[:2]
        vertices = np.array([[(50, height), (width/2 - 45, height/2 + 60),
                              (width/2 + 45, height/2 + 60), (width - 50, height)]], dtype=np.int32)

        ROI_img = region_of_interest(image, vertices)

        return ROI_img

    @staticmethod
    def convert_to_hough_transform(image):

        def draw_lines(image, lines, color=[0, 0, 255], thickness=2):
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(image, (x1, y1), (x2, y2), color, thickness)

        def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
            lines = cv2.HoughLinesP(image, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)

            return lines

        line_arr = hough_lines(image, 1, 1 * np.pi/180, 30, 10, 20)
        line_arr = np.squeeze(line_arr)

        slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

        line_arr = line_arr[np.abs(slope_degree) < 160]
        slope_degree = slope_degree[np.abs(slope_degree) < 160]

        line_arr = line_arr[np.abs(slope_degree) > 95]
        slope_degree = slope_degree[np.abs(slope_degree) > 95]

        L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
        hough_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        L_lines, R_lines = L_lines[:, None], R_lines[:, None]

        draw_lines(hough_img, L_lines)
        draw_lines(hough_img, R_lines)

        return hough_img

    @staticmethod
    def convert_to_weighted(top_image, bottom_image, a=1, b=1., c=0.):
        weighted_img = cv2.addWeighted(bottom_image, a, top_image, b, c)

        return weighted_img

    @staticmethod
    def convert_to_lane_detected(image):
        gray_image = Converters.convert_to_gray_scale(image)
        blur_image = Converters.convert_to_gaussian_blur(gray_image)
        canny_image = Converters.convert_to_canny(blur_image)
        ROI_image = Converters.convert_to_ROI(canny_image)
        hough_image = Converters.convert_to_hough_transform(ROI_image)
        lane_detected_img = Converters.convert_to_weighted(hough_image, image)

        return lane_detected_img


def write_lane_detected_video(video_filename: str, save_filename: str):

    cap = cv2.VideoCapture(video_filename)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #   fourcc: four character code. 코덱, 압축방식, 색상, 픽셀 포맷 등을 정의하는 코드.
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    out = cv2.VideoWriter(save_filename, fourcc, fps, frame_size)

    if not out.isOpened():
        exit("Filed: File open error.")
        cap.release()

    while True:
        read_ok, frame = cap.read()

        if not read_ok:
            break

        converted_frame = Converters.convert_to_lane_detected(frame)
        out.write(converted_frame)

    print(f"Success in creating \"{save_filename}\"")
    cap.release()
    out.release()


if __name__ == "__main__":
    """image converter test"""
    # image1_path = os.path.join('frame', 'line_following', '000000.jpg')
    # image1 = cv2.imread(image1_path)
    #
    # Converters.before_after_test(image1, Converters.convert_to_gray_scale)
    # Converters.before_after_test(image1, Converters.convert_to_lane_detected)

    """ final video writing test """
    write_lane_detected_video(video_filename="line_following.mp4", save_filename="line_detected.mp4")