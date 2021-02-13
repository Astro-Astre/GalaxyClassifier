# -*- coding: utf-8-*-
import csv
import time
import os

from astropy.io import fits

CATALOG = r"D:\\Code\\MachineLearning\\Data\\2020.12.15_MergerClassifier\\catalog\\"
DATA = "D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/raw_data/"


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


def rename(dir, ra, dec, redshift, label, dr7objid):
    old_name_g = dir + "g/" + "ra=%f,dec=%f,class=Sc,dr7objid=%s,redshift=%f.fits" \
                 % (ra, dec, dr7objid, redshift)
    new_name_g = dir + "g/" + "ra=%f,dec=%f,class=%s,dr7objid=%s,redshift=%f.fits" \
                 % (ra, dec, label, dr7objid, redshift)
    old_name_r = dir + "r/" + "ra=%f,dec=%f,class=Sc,dr7objid=%s,redshift=%f.fits" \
                 % (ra, dec, dr7objid, redshift)
    new_name_r = dir + "r/" + "ra=%f,dec=%f,class=%s,dr7objid=%s,redshift=%f.fits" \
                 % (ra, dec, label, dr7objid, redshift)
    old_name_z = dir + "z/" + "ra=%f,dec=%f,class=Sc,dr7objid=%s,redshift=%f.fits" \
                 % (ra, dec, dr7objid, redshift)
    new_name_z = dir + "z/" + "ra=%f,dec=%f,class=%s,dr7objid=%s,redshift=%f.fits" \
                 % (ra, dec, label, dr7objid, redshift)
    os.rename(old_name_g, new_name_g)
    os.rename(old_name_r, new_name_r)
    os.rename(old_name_z, new_name_z)


if __name__ == '__main__':
    start = time.time()
    true_path: str = CATALOG + "galaxy_fuzzy_classifier_catalog.fits"
    wrong_name = DATA + 'fits/all_redshift_under_0.1_galaxy_with_class/'
    wrong_dir = os.listdir(wrong_name + 'g')

    true_hdul = GetFits(true_path).openFits()

    true_dr7objid = true_hdul[1].data['dr7objid']
    true_gz2_class = true_hdul[1].data['gz2_class']

    wrong_ra = []
    wrong_dec = []
    wrong_gz2_fuzzy_class = []
    wrong_dr7objid = []
    wrong_redshift = []
    for i in range(len(wrong_dir)):
        wrong_ra.append(float(wrong_dir[i].split(',')[0].split('=')[1]))
        wrong_dec.append(float(wrong_dir[i].split(',')[1].split('=')[1]))
        wrong_gz2_fuzzy_class.append(wrong_dir[i].split(',')[2].split('=')[1])
        wrong_dr7objid.append(wrong_dir[i].split(',')[3].split('=')[1])
        wrong_redshift.append(float(wrong_dir[i].split(',')[4].split('=')[1].split('.fits')[0]))

    for i in range(len(wrong_dr7objid)):
        percent: float = 1.0 * i / len(wrong_dr7objid)  # 用于显示进度
        if str(wrong_gz2_fuzzy_class[i]) == 'Sc':
            for j in range(true_dr7objid.shape[0]):
                if true_dr7objid[i] == true_dr7objid[j]:
                    if true_gz2_class[j] == 'Sc2m':
                        rename(wrong_name, wrong_ra[i], wrong_dec[i], wrong_redshift[i], "Sc2m",
                               wrong_dr7objid[i])
        print("进度：%.4f" % (percent * 100), "--------%d" % i)
    print("time cost:", time.time() - start)
