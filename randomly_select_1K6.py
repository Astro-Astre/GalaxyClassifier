# ,*, coding: utf,8,*,
import os
import shutil
import random
CATALOG = "D:/Code/MachineLearning/Data/2021.02.14_GalaxyClassifier/catalog/"
DATA = "D:/Code/MachineLearning/Data/2021.02.14_GalaxyClassifier/raw_data/fits/"
TRAIN_DATA = "D:/Code/MachineLearning/Data/2021.02.14_GalaxyClassifier/train_data/"
TEST_DATA = "D:/Code/MachineLearning/Data/2021.02.14_GalaxyClassifier/test_data/"
VERIFY_DATA = "D:/Code/MachineLearning/Data/2021.02.14_GalaxyClassifier/verify_data/"
random.seed(1)


def createPackage(model_package: str):
    if not os.path.isdir(model_package):
        os.mkdir(model_package)


def select(label, path):
    path_data = os.listdir(path + "/" + label + '/g')
    random_num = 300
    old_path = path
    label_train_path = TRAIN_DATA + label + '/'
    label_test_path = TEST_DATA + label + '/'
    label_verify_path = VERIFY_DATA + label + '/'
    # print(old_path)
    # print(label_train_path)
    # if len(path_data) < 1600:
    #     random_num = len(path_data)
    index = random.sample(range(0, len(path_data)), random_num)
    # print(index)
    # print(len(index))
    for i in range(len(index)):
        shutil.move(old_path + "/" + label + "/g/" + path_data[index[i]], label_verify_path + "g/" + path_data[index[i]])
        shutil.move(old_path + "/" + label + "/r/" + path_data[index[i]], label_verify_path + "r/" + path_data[index[i]])
        shutil.move(old_path + "/" + label + "/z/" + path_data[index[i]], label_verify_path + "z/" + path_data[index[i]])
    # print(old_path + "/" + label + "/g/" + path_data[index[0]])
    # print(label_train_path + "g/" + path_data[index[0]])


if __name__ == '__main__':
    path = TRAIN_DATA
    select('Ec', path)
    select('Ei', path)
    select('Er', path)
    select('Sb', path)
    select('SBb', path)
    select('SBc', path)
    select('Sc', path)
    select('Sc_t', path)
    select('Sd', path)
    select('Sen', path)
    select('Ser', path)
    select('Merger', path)
