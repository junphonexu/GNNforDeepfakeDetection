import json
import os
import shutil

if __name__ == '__main__':
    with open("F:/data/FMFCC-V-Competition/metadata.json") as json_file:
        datas = json.load(json_file)
    video_real = "F:/data/FMFCC-V-Competition/real"
    video_fake = "F:/data/Dataset/FMFCC-V-Competition/fake"
    video_path = "F:/data/FMFCC-V-Competition/videos"
    video_files = os.listdir(video_path)
    for i in video_files:
        if datas[i]['label'] == 'real':
            orin = os.path.join(video_path, i)
            dest = os.path.join(video_real, i)
            shutil.move(orin, dest)
        elif datas[i]['label'] == 'fake':
            orin = os.path.join(video_path, i)
            dest = os.path.join(video_fake, i)
            shutil.move(orin, dest)


