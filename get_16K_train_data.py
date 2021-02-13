# ,*, coding: utf,8,*,
import os
import shutil
from astropy.io import fits

CATALOG = "D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/catalog/"
DATA = "D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/raw_data/"


def move(path, dr7obj, label):
    old_path = path
    label_path = path + label + '/'
    # print(old_path + 'g/' + dr7obj)
    # D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/raw_data/fits/all_redshift_under_0.1_
    # galaxy_with_class/g/ra=0.006464,dec=-0.092593,class=Sb,dr7objid=587731186203885750,redshift=0.078132.fits

    # print(label_path + 'g/' + dr7obj)
    # D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/raw_data/fits/all_redshift_under_0.1_
    # galaxy_with_class/Sb/g/ra=0.006464,dec=-0.092593,class=Sb,dr7objid=587731186203885750,redshift=0.078132.fits

    shutil.move(old_path + '/g/' + dr7obj, label_path + 'g/' + dr7obj)
    shutil.move(old_path + '/r/' + dr7obj, label_path + 'r/' + dr7obj)
    shutil.move(old_path + '/z/' + dr7obj, label_path + 'z/' + dr7obj)


def get_info(path):
    """
    :param path:
    :return: dr7objid, ra, dec, gz2_fuzzy_class, redshift
    """
    object_dir: list = os.listdir(path)
    ra = []
    dec = []
    gz2_fuzzy_class = []
    dr7objid = []
    redshift = []
    old_dir = DATA + 'fits/all_redshift_under_0.1_galaxy_with_class/'
    for i in range(len(object_dir)):
    # for i in range(10):
        ra.append(float(object_dir[i].split(',')[0].split('=')[1]))
        dec.append(float(object_dir[i].split(',')[1].split('=')[1]))
        gz2_fuzzy_class.append(object_dir[i].split(',')[2].split('=')[1])
        dr7objid.append(object_dir[i].split(',')[3].split('=')[1])
        redshift.append(float(object_dir[i].split(',')[4].split('=')[1].split('.fits')[0]))
    # print(object_dir[0])
    # ra = 0.006464, dec = -0.092593,class =Sb, dr7objid=587731186203885750, redshift=0.078132.fits
    for i in range(len(dr7objid)):
    # for i in range(1):
        percent: float = 1.0 * i / len(dr7objid)  # 用于显示进度
        if not gz2_fuzzy_class[i] == 'N':
            move(old_dir, object_dir[i], gz2_fuzzy_class[i])
        print("进度：%.4f" % (percent * 100), "--------%d" % i)


if __name__ == '__main__':
    data_path = DATA + 'fits/all_redshift_under_0.1_galaxy_with_class/'
    get_info(data_path + 'g')
