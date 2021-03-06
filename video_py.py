import cv2
import numpy as np

# 第一步：使用cv2.VideoCapture读取视频
camera = cv2.VideoCapture('51.MP4')
# 判断视频是否打开
if (camera.isOpened()):
    print('已获取到视频')
else:
    print('未获取到视频')

size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:' + repr(size))

# 第二步：cv2.getStructuringElement构造形态学使用的kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# 第三步：构造高斯混合模型
model = cv2.createBackgroundSubtractorMOG2()

while (True):
    # 第四步：读取视频中的图片，并使用高斯模型进行拟合
    ret, frame = camera.read()
    # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255
    fgmk = model.apply(frame)
    # 第五步：使用形态学的开运算做背景的去除
    fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)
    # 第六步：cv2.findContours计算fgmk的轮廓
    contours, hierarchy = cv2.findContours(fgmk.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
    for c in contours:
        if cv2.contourArea(c) < 1500:
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 第八步：进行图片的展示
    cv2.imshow('fgmk', fgmk)
    cv2.imshow('frame', frame)

    if cv2.waitKey(150) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()

