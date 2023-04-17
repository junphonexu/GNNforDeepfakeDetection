import os, random, shutil, json

if __name__ == '__main__':
    with open("../data/metadata.json") as json_file:
        datas = json.load(json_file)
    fileDir = "../data/train-cutface"

    facename = open('../data/txt/train_face.txt', mode='a')
    # cutfacename = open('../data/txt/train_cutface.txt', mode='a')
    label = open('../data/txt/train_label.txt', mode='a')
    # label1 = open('../data/txt/train_label1.txt', mode='a')
    pathDir = os.listdir(fileDir)
    for item in pathDir:
        print(item.split("_")[0])
        imgpic = os.listdir(os.path.join(fileDir, item))
        facename.write('train-cutface' + '/' + item + '/' + imgpic[0] + '\n')
        facename.write('train-cutface' + '/' + item + '/' + imgpic[1] + '\n')
        facename.write('train-cutface' + '/' + item + '/' + imgpic[2] + '\n')
        facename.write('train-cutface' + '/' + item + '/' + imgpic[3] + '\n')
        facename.write('train-cutface' + '/' + item + '/' + imgpic[4] + '\n')
        name = item.split('_')[0]
        if datas[name + '.mp4']['label'] == 'real':
            label.write('1\n')
            label1.write('1\n')
            label1.write('1\n')
            label1.write('1\n')
            label1.write('1\n')
            label1.write('1\n')
        else:
            label.write('0\n')
            label1.write('0\n')
            label1.write('0\n')
            label1.write('0\n')
            label1.write('0\n')
            label1.write('0\n')
        # for i in range(1, 6):
        #     cutfacename.write('output_cutpartface/' + name + '/' + str(i) + '_eyes.jpg' + '\n')
        #     cutfacename.write('output_cutpartface/' + name + '/' + str(i) + '_mouth.jpg' + '\n')
        #     cutfacename.write('output_cutpartface/' + name + '/' + str(i) + '_nose.jpg' + '\n')
        #     cutfacename.write('output_cutpartface/' + name + '/' + str(i) + '_other.jpg' + '\n')
    facename.close()
    label.close()
    cutfacename.close()
