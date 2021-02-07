# -*- coding: utf-8-*-
import os

CATALOG = "D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/catalog/"
DATA = "D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/raw_data/"


def remove_missing(source, object):
    """
    grz三个band可能不是数据齐全，所以我们在删除小数据之后使三个文件夹的文件匹配。
    :param source: 文件夹1
    :param object: 文件夹2
    :return: 0
    """
    object_list = os.listdir(object)
    source_list = os.listdir(source)
    for i in range(len(object_list)):
        object_list[i] = object_list[i][:-5]
    for i in range(len(source_list)):
        source_list[i] = source_list[i][:-5]
    ret = list(set(source_list).difference(set(object_list)))
    for i in range(len(ret)):
        os.remove(source+'/'+ret[i]+'.fits')


def remove_incomplete(path, size):
    """
    有些通道的数据可能比较小或者缺失，需要删除小于正常文件大小的文件。
    :param path: 文件夹路径名
    :param size: 正常文件大小，建议设置的比齐稍小一些
    :return: 0
    """
    object_list = os.listdir(path)
    for i in range(len(object_list)):
        if os.path.getsize(path + '/' + object_list[i]) < size:
            os.remove(path + '/' + object_list[i])


if __name__ == "__main__":
    data_path = DATA+'fits/all_redshift_under_0.1_galaxy_with_class/'
    fits_size = 260000
    remove_incomplete(data_path + 'g', fits_size)
    remove_incomplete(data_path + 'r', fits_size)
    remove_incomplete(data_path + 'z', fits_size)
    # 反复匹配两个文件夹，时间耗得多些但是保证没问题。
    remove_missing(data_path + '/g', data_path + '/r')
    remove_missing(data_path + '/g', data_path + '/z')
    remove_missing(data_path + '/r', data_path + '/g')
    remove_missing(data_path + '/r', data_path + '/z')
    remove_missing(data_path + '/z', data_path + '/g')
    remove_missing(data_path + '/z', data_path + '/r')
