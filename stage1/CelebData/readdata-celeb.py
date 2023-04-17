import os, random, shutil, json

if __name__ == '__main__':
    fileDir = "F:/cdf/real"

    train_facename = open('real_train.txt', mode='a')
    train_label = open('real_train_label.txt', mode='a')

    test_facename = open('real_test.txt', mode='a')
    test_label = open('real_test_label.txt', mode='a')

    pathDir = os.listdir(fileDir)

    flag = 0

    for item in pathDir:
        flag += 1
        print(item.split("_")[0])
        imgpic = os.listdir(os.path.join(fileDir, item))

        if flag % 10 == 9 or flag % 10 == 0:
            test_facename.write(fileDir + '/' + item + '/' + imgpic[0] + '\n')
            test_facename.write(fileDir + '/' + item + '/' + imgpic[1] + '\n')
            test_facename.write(fileDir + '/' + item + '/' + imgpic[2] + '\n')
            test_facename.write(fileDir + '/' + item + '/' + imgpic[3] + '\n')
            test_facename.write(fileDir + '/' + item + '/' + imgpic[4] + '\n')

            # test_label.write('1\n')
            # test_label.write('1\n')
            # test_label.write('1\n')
            # test_label.write('1\n')
            # test_label.write('1\n')

            test_label.write('0\n')
            test_label.write('0\n')
            test_label.write('0\n')
            test_label.write('0\n')
            test_label.write('0\n')
        else:
            train_facename.write(fileDir + '/' + item + '/' + imgpic[0] + '\n')
            train_facename.write(fileDir + '/' + item + '/' + imgpic[1] + '\n')
            train_facename.write(fileDir + '/' + item + '/' + imgpic[2] + '\n')
            train_facename.write(fileDir + '/' + item + '/' + imgpic[3] + '\n')
            train_facename.write(fileDir + '/' + item + '/' + imgpic[4] + '\n')

            # train_label.write('1\n')
            # train_label.write('1\n')
            # train_label.write('1\n')
            # train_label.write('1\n')
            # train_label.write('1\n')

            train_label.write('0\n')
            train_label.write('0\n')
            train_label.write('0\n')
            train_label.write('0\n')
            train_label.write('0\n')

    train_facename.close()
    train_label.close()

    test_facename.close()
    test_label.close()

