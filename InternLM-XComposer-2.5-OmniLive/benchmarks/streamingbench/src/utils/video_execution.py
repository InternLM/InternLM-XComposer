import os
from moviepy.editor import VideoFileClip
import pdb 

def split_video(video_file, ori_folder, start_time, end_time, type='proactive'):
    """
    Split video into prefix part based on timestamp.
    video_file: path to video file
    start_time: start time in seconds
    end_time: end time in seconds
    """
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = os.path.join(os.path.dirname(video_file), "tmp")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")

    if os.path.exists(output_file):
        print(f"Video file {output_file} already exists.")
        return output_file


    folder = f'{ori_folder}/{type}'
    filename = video_name.split('/')[-1]  # 获取 'sample_17_proactive.mp4'
    base_name = filename.split(f'_{type}')[0]  # 去掉 '_proactive'，结果为 'sample_17'
    
    video_path = os.path.join(folder, base_name, 'video.mp4')

    video = VideoFileClip(video_path)

    clip = video.subclip(start_time, end_time)
    
    clip.write_videofile(output_file)
    clip.close()
    video.close()
    print(f"Video: {output_file} splitting completed.")
    return output_file
    