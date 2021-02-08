# -*- coding: utf-8-*-
import csv
import sys
import re
import time

import numpy as np

from astropy.io import fits

CATALOG = r"D:\\Code\\MachineLearning\\Data\\2020.12.15_MergerClassifier\\catalog\\"


def get_redshift(path: str):
    redshift = []
    ra = []
    dec = []
    redshifterr = []
    with open(path) as f:
        header = csv.DictReader(f)
        for row in header:
            redshift.append(float(row['redshift']))
            ra.append(float(row['ra']))
            dec.append(float(row['dec']))
            redshifterr.append(float(row['redshifterr']))
        return redshift, ra, dec, redshifterr


def get_class(path: str):
    label = []
    ra = []
    dec = []
    with open(path) as file:
        header = csv.DictReader(file)
        for row in header:
            label.append(str(row['gz2_class']))
            ra.append(float(row['ra']))
            dec.append(float(row['dec']))
        return label, ra, dec


def match_class(data, class_list, label_list):
    output = 'N'
    for p in range(len(class_list)):
        if class_list[p].search(data):
            output = label_list[p]
        elif data[:4] == 'Sc?t':
            output = 'Sc?t'
    return output


class GetFits:
    def __init__(self, filename):
        self.filename = filename

    # 打开filename的fits文件，后续使用直接调用hdul
    def openFits(self):
        hdul = fits.open(self.filename)
        return hdul


if __name__ == '__main__':
    start = time.time()
    # redshift_path内存有325704个星系中部分的红移值
    # class_path内存有239695个GZ2的分类标签
    # 坐标用redshift_path的，精度高一些
    redshift_path: str = CATALOG + "gz2sample.fits"
    class_path: str = CATALOG + "gz2_hart16.fits"
    label_list = [r"Er", r"SBc", r"Sb",
                  r"Ei", r"Sc2m", r"Sc", r"Ser",
                  r"Ec", r"Sd", r"SBb", r"Sen"]
    class_list = [re.compile("Er"), re.compile("SBc"), re.compile("Sb"),
                  re.compile("Ei"), re.compile("Sc2m"), re.compile('Sc'), re.compile("Ser"),
                  re.compile("Ec"), re.compile("Sd"), re.compile("SBb"), re.compile("Sen")]
    redshift_hdul = GetFits(redshift_path).openFits()
    # print(redshift_hdul.info())
    class_hdul = GetFits(class_path).openFits()
    # print(class_hdul.info())

    redshift_dr7objid = redshift_hdul[1].data['objid']
    redshift_ra = redshift_hdul[1].data['ra']
    redshift_dec = redshift_hdul[1].data['dec']
    redshift = redshift_hdul[1].data['redshift']
    redshifterr = redshift_hdul[1].data['redshifterr']
    # gz2sample.fits默认按objid已经排好序了，后四列数据都可以用float，故这里直接拼合后四列
    # 这里的索引顺序不会出错
    redshift_path_data = np.array([redshift_ra, redshift_dec, redshift, redshifterr]).T
    # 索引0:ra, 1:dec, 2:redshift, 3:redshifterr
    index = []
    # 去掉nan，记住ndarray不能用for删除，否则会list of index，因为删除当前位置的，下一个会往前走（类似于链表）
    for i in range(redshift_path_data.shape[0]):
        if np.isnan(redshift_path_data[i][3]):
            index.append(i)
    redshift_path_data = np.delete(redshift_path_data, index, axis=0)

    # 以前没注意到的地方，ndarray里面数据类型一定会强制转为相同的
    # 这里要取得objid（int64）的排序索引，然后给这两列数据按此索引排序
    # 因为希望用int64的两列objid做匹配，如果和gz2_class合并，会导致强制转换为str
    data = redshift_path_data[np.argsort(redshift_path_data[:, 0])]

    class_dr7objid = class_hdul[1].data['dr7objid'].astype('int64')
    sort_index = np.argsort(class_dr7objid)
    class_dr7objid = class_dr7objid[sort_index].T
    class_gz2class = class_hdul[1].data['gz2_class']
    class_gz2class = class_gz2class[sort_index].T

    # csv_file_name = CATALOG + "galaxy_classifier_catalog.fits"
    fits_file_name = CATALOG + "galaxy_fuzzy_classifier_catalog.fits"

    fits_dr7objid = []
    fits_ra = []
    fits_dec = []
    fits_redshift = []
    fits_redshifterr = []
    fits_gz2_class = []
    k = 0
    for i in range(redshift_dr7objid.shape[0]):
    # for i in range(100):
        percent: float = 1.0 * i / redshift_dr7objid.shape[0]  # 用于显示进度
        for j in range(k, class_dr7objid.shape[0]):
            if redshift_dr7objid[i] == class_dr7objid[j]:
                k = j
                fits_dr7objid.append(class_dr7objid[j])
                fits_ra.append(redshift_ra[i])
                fits_dec.append(redshift_dec[i])
                fits_redshift.append(redshift[i])
                fits_redshifterr.append(redshifterr[i])
                fits_gz2_class.append(class_gz2class[j])
                break
        print("进度：%.4f" % (percent * 100), "--------%d" % i)
        sys.stdout.write("进度：%.4f" % (percent * 100))
        sys.stdout.write("%\r")
        sys.stdout.flush()
    # fits_dr7objid: numpy.int64
    # fits_ra: numpy.float64
    # fits_dec: numpy.float64
    # fits_redshift: numpy.float32
    # fits_redshifterr: numpy.float32
    # fits_gz2_class: str

    # 深度拷贝！！
    fuzzy_gz2_class = fits_gz2_class.copy()
    for i in range(len(fuzzy_gz2_class)):
        fuzzy_gz2_class[i] = match_class(fuzzy_gz2_class[i], class_list, label_list)

    hdu = fits.BinTableHDU.from_columns([fits.Column(name='dr7objid', format='K', array=fits_dr7objid),
                                         fits.Column(name='ra', format='D', array=fits_ra),
                                         fits.Column(name='dec', format='D', array=fits_dec),
                                         fits.Column(name='redshift', format='D', array=fits_redshift),
                                         fits.Column(name='redshifterr', format='D', array=fits_redshifterr),
                                         fits.Column(name='gz2_class', format='44A', array=fits_gz2_class),
                                         fits.Column(name='gz2_fuzzy_class', format='44A', array=fuzzy_gz2_class),
                                         ])
    hdu.writeto(fits_file_name)
    print("time cost:", time.time() - start)
























    # with open(csv_file_name, "w", newline='') as csv_file:
    #     writer = csv.DictWriter(csv_file, ["dr7objid", "ra", "dec",
    #                                        "redshift", "redshifterr", "gz2_class"])
    #     writer.writeheader()
    #
    # for i in range(redshift_dr7objid.shape[0]):
    #     with open(csv_file_name, mode='a', newline='', encoding='utf8') as csv_file:
    #         writer = csv.DictWriter(csv_file, ["dr7objid", "ra", "dec",
    #                                            "redshift", "redshifterr", "gz2_class"])
    #         percent: float = 1.0 * i / redshift_dr7objid.shape[0]  # 用于显示进度
    #         for j in range(class_dr7objid.shape[0]):
    #             # print(redshift_dr7objid[i], class_dr7objid[j])
    #             if redshift_dr7objid[i] == class_dr7objid[j]:
    #                 print(type(class_dr7objid[j]))
    #                 print(class_dr7objid[j])
    #                 writer.writerow({"dr7objid": class_dr7objid[j],
    #                                  "ra": redshift_ra[i],
    #                                  "dec": redshift_dec[i],
    #                                  "redshift": redshift[i],
    #                                  "redshifterr": redshifterr[i],
    #                                  "gz2_class": class_gz2class[j]})
    #                 break
    #         print("进度：%.4f" % (percent * 100), "--------%d" % i)
    #         sys.stdout.write("进度：%.4f" % (percent * 100))
    #         sys.stdout.write("%\r")
    #         sys.stdout.flush()
