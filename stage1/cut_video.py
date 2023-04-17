# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import numpy as np
import os
import dlib


# 得到图片的掩膜利用的就是opencv的convexhull得到凸包然后就可以抠出人脸区域
def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int_)

    #hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    hull_mask = np.full(image_shape[0:2], 0, dtype=np.float32)
    # hull_mask = np.full((360, 640), 0, dtype=np.float32)

    # 全脸
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    return hull_mask


def get_image_part_mask(image_shape, image_landmarks, nose=False, mouth=False, eyes=False,  ie_polys=None):
    # get the mask of the image

    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    # int_lmrks = np.array(image_landmarks, dtype=np.int_)
    int_lmrks = image_landmarks

    # hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    hull_mask = np.full(image_shape[0:2], 0, dtype=np.float32)
    # hull_mask = np.full((360, 640), 0, dtype=np.float32)

    # nose
    if nose:
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((
                # 40
                int_lmrks[39:40],
                # 22, 23
                int_lmrks[21:23],
                # 43
                int_lmrks[42:43],
                # 50, 51, 52, 53, 54
                int_lmrks[49:54],
            ))), (1,))
    # mouth
    if mouth:
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(int_lmrks[48:61]), (1,))
    # eyes
    if eyes:
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(
            np.concatenate((
                # 27, 28, 29
                int_lmrks[26:29],
                # 17, 18
                int_lmrks[16:18],
                # 22, 23
                int_lmrks[21:23],
                # 1
                int_lmrks[0:1],
            ))), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    return hull_mask


# 掩模与原始图像进行与运算，返回图像是三通道。
def merge_add_mask(img_1, mask, isface=False):
    if mask is not None:
        height = mask.shape[0]
        width = mask.shape[1]
        # channel_num = mask.shape[2]
        for row in range(height):
            for col in range(width):
                if not isface:
                    if mask[row, col] == 0:
                        mask[row, col] = 0
                    else:
                        mask[row, col] = 255
                else:
                    if mask[row, col] == 0:
                        mask[row, col] = 255
                    else:
                        mask[row, col] = 0
        # print(mask.shape)
        b_channel, g_channel, r_channel = cv2.split(img_1)
        # print(b_channel.shape)
        r_channel = cv2.bitwise_and(r_channel, mask)
        g_channel = cv2.bitwise_and(g_channel, mask)
        b_channel = cv2.bitwise_and(b_channel, mask)
        res_img = cv2.merge((b_channel, g_channel, r_channel))
    else:
        res_img = img_1
    return res_img


def merge_add_alpha(img_1, mask):
    # merge rgb and mask into a rgba image
    r_channel, g_channel, b_channel = cv2.split(img_1)
    if mask is not None:
        alpha_channel = np.ones(mask.shape, dtype=img_1.dtype)
        alpha_channel *= mask*255
    else:
        alpha_channel = np.zeros(img_1.shape[:2], dtype=img_1.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA


# Press the green button in the gutter to run the script.
def loadLandmarks(mark_path):
    arr1 = np.loadtxt(mark_path, delimiter=',', dtype='float', encoding='utf-8', skiprows=1)
    result = []
    # 10 * 2 * 68   10帧图片的人脸标志点
    for j in range(0, 10):
        arr = [[0, 0] for i in range(68)]
        for i in range(0, 68):
            arr[i][0] = arr1[j][i + 5]
            arr[i][1] = arr1[j][i + 73]
        result.append(arr)
    # print(result)
    # np.save('./result2', result)
    return result


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def cutface(video_path, detector, predictor):
    print('裁剪视频：', video_path)
    # 加载视频
    cap = cv2.VideoCapture(video_path)
    frame_count = 1
    facelist = []
    cutfacelist = []
    while cap.isOpened():
        if frame_count > 5:
            break
        print('裁剪视频帧：', frame_count)
        suc, frame = cap.read()  # 读取一帧

        # 人脸数rects
        rects = detector(frame, 0)
        # faces存储full_object_detection对象
        faces = dlib.full_object_detections()
        full_object_detection = predictor(frame, rects[0])
        landmarks = np.matrix([[p.x, p.y] for p in full_object_detection.parts()])

        for i in range(len(rects)):
            faces.append(predictor(frame, rects[i]))

        face_images = dlib.get_face_chips(frame, faces, size=224)
        for image in face_images:
            cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 获得全脸的裁剪区
        allface_mask = get_image_hull_mask(frame.shape, landmarks)
        allface_img = merge_add_mask(frame.astype(np.uint8), allface_mask.astype(np.uint8))
        facelist.append(allface_img)
        cv2.imwrite(f'{frame_count}_cutface.jpg', allface_img)

        # 获得鼻子的裁剪区
        nose_mask = get_image_part_mask(frame.shape, landmarks, nose=True)
        nose_img = merge_add_mask(frame.astype(np.uint8), nose_mask.astype(np.uint8))
        cutfacelist.append(nose_img)
        cv2.imwrite(f'{frame_count}_nose.jpg', nose_img)

        # 获得眼睛的裁剪区
        eye_mask = get_image_part_mask(frame.shape, landmarks, eyes=True)
        eye_img = merge_add_mask(frame.astype(np.uint8), eye_mask.astype(np.uint8))
        cutfacelist.append(eye_img)
        cv2.imwrite(f'{frame_count}_eyes.jpg', eye_img)

        # 获得嘴部的裁剪区
        mouth_mask = get_image_part_mask(frame.shape, landmarks, mouth=True)
        mouth_img = merge_add_mask(frame.astype(np.uint8), mouth_mask.astype(np.uint8))
        cutfacelist.append(mouth_img)
        cv2.imwrite(f'{frame_count}_mouth.jpg', mouth_img)

        # 获得剩余部分
        face_mask = get_image_hull_mask(frame.shape, landmarks)
        cut_mask = get_image_part_mask(frame.shape, landmarks, nose=True, eyes=True, mouth=True)
        face_img = merge_add_mask(frame.astype(np.uint8), face_mask.astype(np.uint8))
        cut_face = merge_add_mask(face_img.astype(np.uint8), cut_mask.astype(np.uint8), isface=True)
        cutfacelist.append(cut_face)
        cv2.imwrite(f'{frame_count}_other.jpg', cut_face)

        frame_count += 1
        cap.release()
        cv2.destroyAllWindows()
    return facelist, cutfacelist

if __name__ == '__main__':
    # input a video

    video_path = '4.mp4'

    predictor_model = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()  # dlib人脸检测器
    predictor = dlib.shape_predictor(predictor_model)
    facelist, cutfacelist = cutface(video_path, detector, predictor)

