import cv2
import os
import numpy as np

def video_to_frame(filename, directory="."):  # 사용자가 원하는 프레임으로 캡쳐. 원하는 장소에 저장.
    """
    :param filename: 동영상 경로
    :param directory: 캡쳐한 이미지들을 저장할 디렉토리
    """

    filename_without_extension = os.path.splitext(os.path.basename(filename))[0]
    directory_path = os.path.join(directory, 'frame', filename_without_extension)

    os.makedirs(directory_path, exist_ok=True)

    video = cv2.VideoCapture(filename)

    if not video.isOpened():
        print("Could not Open :", filename)
        exit(0)

    number = 0

    while (True):

        ret, image = video.read()
        if not ret: break

        cv2.imwrite(os.path.join(directory_path, "%06d.jpg" % number), image)
        print('Saved frame number :', filename_without_extension, str(int(video.get(1))))
        number += 1

    video.release()



class Filters:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.image_path = image_path
        print(f"'{os.path.basename(image_path)}' is loaded as cv2 object")

    def convert_to_gray_scale(self, image=None):
        img = self.image
        if image is not None:
            img = image

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cv2.imshow('gray_before', img)
        cv2.imshow('gray_after', gray_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return gray_img
#ㅠㅓㅗㅠㅓㅗ
    def convert_to_gaussian_blur(self, image=None):
        img = self.image
        if image is not None:
            img = image

        blur_img = cv2.GaussianBlur(img, (3, 3), 0)

        cv2.imshow('blur_before', img)
        cv2.imshow('blur_after', blur_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return blur_img

    def convert_to_canny(self, image=None):
        img = self.image
        if image is not None:
            img = image

        canny_img = cv2.Canny(img, 70, 210)

        cv2.imshow('Canny_before', img)
        cv2.imshow('Canny_after', canny_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return canny_img

    def convert_to_ROI(self, image=None):
        img = self.image
        if image is not None:
            img = image

        def region_of_interest(img, vertices, color3=(255, 255, 255), color1=255):
            mask = np.zeros_like(img)
            if len(img.shape) > 2:
                color = color3
            else:
                color = color1

            cv2.fillPoly(mask, vertices, color)
            ROI_image = cv2.bitwise_and(img, mask)

            return ROI_image


        height, width = img.shape[:2]
        vertices = np.array([[(50, height), (width/2 - 45, height/2 + 60),
                              (width/2 + 45, height/2 + 60), (width - 50, height)]], dtype=np.int32)

        ROI_img = region_of_interest(img, vertices)

        cv2.imshow('ROI_before', img)
        cv2.imshow('ROI_after', ROI_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return ROI_img

    def convert_to_hough_transform(self, image=None):
        img = self.image
        if image is not None:
            img = image

        def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(img, (x1, y1), (x2, y2), color, thickness)

        def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
            lines = cv2.HoughLinesP(img, rho, theta, threshold, minLineLength=min_line_len, maxLineGap=max_line_gap)
            #line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            #draw_lines(line_img, lines)

            return lines

        line_arr = hough_lines(img, 1, 1 * np.pi/180, 30, 10, 20)
        line_arr = np.squeeze(line_arr)

        slope_degree = (np.arctan2(line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

        line_arr = line_arr[np.abs(slope_degree) < 160]
        slope_degree = slope_degree[np.abs(slope_degree) < 160]

        line_arr = line_arr[np.abs(slope_degree) < 95]
        slope_degree = slope_degree[np.abs(slope_degree) > 95]
        print(line_arr.shape) #TODO (0,4)이다. exception 처리를 해줘야 하나보다.
        print(slope_degree.shape) #(23,)
        print(line_arr, '\n')
        print(slope_degree)
        L_lines, R_lines = line_arr[(slope_degree > 0), :], line_arr[(slope_degree < 0), :]
        hough_img = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        L_lines, R_lines = L_lines[:, None], R_lines[:, None]

        draw_lines(hough_img, L_lines)
        draw_lines(hough_img, R_lines)

        cv2.imshow('hough_before', img)
        cv2.imshow('hough_after', hough_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return hough_img

    def convert_to_weighted_img(self, top_image, bottom_image=None, a=1, b=1., c=0.):
        bottom_img = self.image
        if bottom_image is not None:
            bottom_img = bottom_image
        weighted_img = cv2.addWeighted(bottom_img, a, top_image, b, c)

        cv2.imshow('weighted_before', top_image)
        cv2.imshow('weighted_after', weighted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return weighted_img








def main():
    #video_to_frame('line_following.mp4')
    image1_path = os.path.join('frame', 'line_following', '000000.jpg')
    converter = Filters(image1_path)
    print(converter.image.shape)
    gray_image1 = converter.convert_to_gray_scale()
    (gray_image1.shape)
    blur_image1 = converter.convert_to_gaussian_blur(image=gray_image1)
    print(blur_image1.shape)
    canny_image1 = converter.convert_to_canny(image=blur_image1)
    print(canny_image1.shape)
    ROI_image1 = converter.convert_to_ROI(image=canny_image1)
    print(ROI_image1.shape)
    hough_image1 = converter.convert_to_hough_transform(image=ROI_image1)
    print(hough_image1.shape)
    weighted_image1 = converter.convert_to_weighted_img(hough_image1)
    print(weighted_image1.shape)





if __name__ == "__main__":
    main()