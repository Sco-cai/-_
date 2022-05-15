
import cv2
import numpy as np



# 第一步：使用cv2.VideoCapture读取视频
# camera = cv2.VideoCapture('/home/hc/file_jj/特征工程检测视频/24.mp4')
camera = cv2.VideoCapture(0)

size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print('size:' + repr(size))

# 第二步：cv2.getStructuringElement构造形态学使用的kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# 第三步：构造高斯混合模型
model = cv2.createBackgroundSubtractorMOG2()
# 创建保存视频的对象，设置编码格式，帧率，图像的宽高等
out = cv2.VideoWriter('28_1.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20,
                     (size[0], size[1]))  # 保存视频
while (True):
    # 第四步：读取视频中的图片，并使用高斯模型进行拟合
    ret, frame = camera.read()
    # 运用高斯模型进行拟合，在两个标准差内设置为0，在两个标准差外设置为255
    fgmk = model.apply(frame)
    # 第五步：使用形态学的开运算做背景的去除
    fgmk = cv2.morphologyEx(fgmk, cv2.MORPH_OPEN, kernel)
    # 第六步：cv2.findContours计算fgmk的轮廓
    # contours, hierarchy = cv2.findContours(fgmk.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
    #
    # img = cv2.drawContours(frame, contours, -1, (0, 0, 255), -1)


    # 第七步：使用斑点检测检测目标坐标
    # 设置SimpleBlobDetector_Params参数
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs = 10
    # 合并： 计算二进制图像中二进制斑点的重心，并合并更靠近minDistBetweenBlobs的斑点
    params.filterByInertia = False
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByConvexity = False
    # 颜色区分,提取亮点
    params.filterByColor = True
    params.blobColor = 255

    # 改变阈值
    params.minThreshold = 64
    params.maxThreshold = 255
    # 通过面积滤波
    # 这里的面积是基于像素单位的，主要是利于几何矩进行计算得到。
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 300



    # 通过惯性比滤波
    # 惯性率是跟偏心率，圆形的偏心率等于0， 椭圆的偏心率介于0和1之间，直线的偏心率接近于0， 基于几何矩计算惯性率比计算偏心率容易，所以OpenCV选择了惯性率这个特征值，根据惯性率可以计算出来偏心率
    params.filterByInertia = True
    params.minInertiaRatio = 0.001
    # 创建一个检测器并使用默认参数
    detector = cv2.SimpleBlobDetector_create(params)
    # 检测blobs.
    key_points = detector.detect(fgmk)
    # 绘制blob的红点
    draw_image = cv2.drawKeypoints(frame, key_points, np.array([]), (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    draw_image_1 = cv2.drawKeypoints(fgmk, key_points, np.array([]), (0, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    out.write(draw_image_1)  # 视频写入


    # 第八步：进行图片的展示
    cv2.imshow('fgmk', draw_image)
    cv2.imshow('frame', draw_image_1)
    if cv2.waitKey(50) & 0xff == 27:
        break


camera.release()
out.release()
cv2.destroyAllWindows()
