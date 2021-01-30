# -*- coding: utf-8-*-
import csv
import sys

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


class GetFits:
    def __init__(self, filename):
        self.filename = filename

    # 打开filename的fits文件，后续使用直接调用hdul
    def openFits(self):
        hdul = fits.open(self.filename)
        return hdul


if __name__ == '__main__':
    #     redshift_path: str = CATALOG + "galaxy_redshift_radec.csv"
    #     class_path: str = CATALOG + "gz2_hart16.csv"
    #     fits_path: str = CATALOG + "gz2_hart16.fits"
    #
    #     redshift, redshift_ra, redshift_dec, redshifterr = get_redshift(redshift_path)
    #     label, class_ra, class_dec = get_class(class_path)
    #
    #     fits_hdul = GetFits(fits_path).openFits()
    #     dr7objid = fits_hdul[1].data['dr7objid']
    #     fits_ra = fits_hdul[1].data['ra']
    #     fits_dec = fits_hdul[1].data['dec']
    #     location = np.array([fits_ra, fits_dec])
    #     print(type(location))
    #     with open(CATALOG + "galaxy_classifier_catalog1.csv", "w", newline='') as csv_file:
    #         writer = csv.DictWriter(csv_file, ["dr7objid", "ra", "dec",
    #                                            "redshift", "redshifterr", "class"])
    #         writer.writeheader()
    #         for i in range(len(redshift_dec)):
    #             for j, item in enumerate(class_ra):
    #                 for k in range(location.shape[1]):
    #                     if (redshift_ra[i] == class_ra[j] == location[k, 0]) and\
    #                             (redshift_dec[i] == class_dec[j] == location[k, 1]):
    #                         writer.writerow({"dr7objid": dr7objid[k],
    #                                          "ra": redshift_ra[i],
    #                                          "dec": redshift_dec[i],
    #                                          "redshift": redshift[i],
    #                                          "redshifterr": redshifterr[i],
    #                                          "class": label[j]})
    #                         exit()
    #
    # redshift_path内存有325704个星系中部分的红移值
    # class_path内存有239695个GZ2的分类标签
    # 坐标用redshift_path的，精度高一些
    redshift_path: str = CATALOG + "gz2sample.fits"
    class_path: str = CATALOG + "gz2_hart16.fits"

    redshift_hdul = GetFits(redshift_path).openFits()
    class_hdul = GetFits(class_path).openFits()

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

    csv_file_name = CATALOG + "galaxy_classifier_catalog1.csv"
    with open(csv_file_name, "w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, ["dr7objid", "ra", "dec",
                                           "redshift", "redshifterr", "gz2_class"])
        writer.writeheader()

    for i in range(redshift_dr7objid.shape[0]):
        with open(csv_file_name, mode='a', newline='', encoding='utf8') as csv_file:
            writer = csv.DictWriter(csv_file, ["dr7objid", "ra", "dec",
                                               "redshift", "redshifterr", "gz2_class"])
            percent: float = 1.0 * i / redshift_dr7objid.shape[0]  # 用于显示进度
            for j in range(class_dr7objid.shape[0]):
                # print(redshift_dr7objid[i], class_dr7objid[j])
                if redshift_dr7objid[i] == class_dr7objid[j]:
                    print(type(class_dr7objid[j]))
                    print(class_dr7objid[j])
                    writer.writerow({"dr7objid": class_dr7objid[j],
                                     "ra": redshift_ra[i],
                                     "dec": redshift_dec[i],
                                     "redshift": redshift[i],
                                     "redshifterr": redshifterr[i],
                                     "gz2_class": class_gz2class[j]})
                    break
            print("进度：%.4f" % (percent * 100), "--------%d" % i)
            sys.stdout.write("进度：%.4f" % (percent * 100))
            sys.stdout.write("%\r")
            sys.stdout.flush()
