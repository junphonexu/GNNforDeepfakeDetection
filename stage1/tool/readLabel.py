import json

if __name__ == '__main__':
    with open("F:/data/FMFCC-V-Competition/metadata.json") as json_file:
        datas = json.load(json_file)
    print(datas["10000001.mp4"]['label'])
