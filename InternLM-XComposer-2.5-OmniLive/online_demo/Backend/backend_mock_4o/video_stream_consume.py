import cv2
import threading
from queue import Queue


def scheduled_screenshot(video_stream_url: str, interval: int, frame_list: Queue, stop_event: threading.Event):
    # 打开视频文件
    cap = cv2.VideoCapture(video_stream_url)
    if not cap.isOpened():
        stop_event.set()
        raise Exception(f"video stream open failed, url: {video_stream_url}")

    # 获取视频的帧率
    #fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30
    # 计算每10秒对应的帧数
    frame_interval = int(interval * fps)
    print(f"scheduled_screenshot start, video stream url {video_stream_url}, fps: {cap.get(cv2.CAP_PROP_FPS)}, frame_interval: {frame_interval}")

    # 初始化当前帧的索引
    current_frame = 0

    screenshot_count = 0

    # 循环遍历视频的每一帧
    while not stop_event.is_set():  # 没有被主线程通知结束任务时继续循环

        # 更新当前帧的索引
        if (current_frame % frame_interval != 0):
            # 跳帧
            ret = cap.grab()
            # 如果读取失败，则退出循环
            if not ret:
                print(f"scheduled_screenshot read frame failed, video stream url {video_stream_url}, current_frame: {current_frame}, fps: {fps}, frame_interval: {frame_interval}")
                break
            current_frame += 1
            continue

        # 读取一帧
        ret, frame = cap.read()

        # 如果读取失败，则退出循环
        if not ret:
            print(
                f"scheduled_screenshot read frame failed, video stream url {video_stream_url}, current_frame: {current_frame}, fps: {fps}, frame_interval: {frame_interval}")
            break

        frame_list.append(frame)
        #print(len(frame_list))
        #print(frame_list[-1].mean())

        current_frame += 1

    # 释放视频文件和窗口资源
    cap.release()
    print(f"scheduled_screenshot 执行完毕")

