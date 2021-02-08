# -*- coding: utf-8-*-
import os

import numpy as np

CATALOG = "D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/catalog/"
DATA = "D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/raw_data/"


def get_info(path):
    """
    :param path:
    :return: dr7objid, ra, dec, gz2_class, redshift
    """
    object_dir: list = os.listdir(path)
    ra = []
    dec = []
    gz2_class = []
    dr7objid = []
    redshift = []
    print(object_dir[0].split('-'))
    k = 0
    j = 0
    print(len(object_dir))
    for i in range(len(object_dir)):
    # for i in range(10):
        if len(object_dir[i].split('-')) == 6:
            ra.append(float(object_dir[i].split('-')[0].split('=')[1]))
            dec.append(float(object_dir[i].split('-')[2]))
            gz2_class.append(object_dir[i].split('-')[3].split('=')[1])
            dr7objid.append(object_dir[i].split('-')[4].split('=')[1])
            redshift.append(float(object_dir[i].split('-')[5].split('=')[1][:-5]))
        elif len(object_dir[i].split('-')) == 5:
            ra.append(float(object_dir[i].split('-')[0].split('=')[1]))
            dec.append(float(object_dir[i].split('-')[1].split('=')[1]))
            gz2_class.append(object_dir[i].split('-')[2].split('=')[1])
            dr7objid.append(object_dir[i].split('-')[3].split('=')[1])
            redshift.append(float(object_dir[i].split('-')[4].split('=')[1][:-5]))
    print(ra)
    print(dec)
    print(gz2_class)
    print(dr7objid)
    print(redshift)
    ret = set(sorted(gz2_class))
    print("Sc+t" in ret)
    print("Sb+t" in ret)
    print(len(ret))
    print(ret)


if __name__ == '__main__':
    data_path = DATA + 'fits/all_redshift_under_0.1_galaxy_with_class/'
    get_info(data_path + 'g')
