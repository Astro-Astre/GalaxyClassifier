# -*- coding: utf-8-*-
import csv
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
    redshift_path: str = CATALOG + "galaxy_redshift_radec.csv"
    class_path: str = CATALOG + "gz2_hart16.csv"
    fits_path: str = CATALOG + "gz2_hart16.fits"

    redshift, redshift_ra, redshift_dec, redshifterr = get_redshift(redshift_path)
    label, class_ra, class_dec = get_class(class_path)
    fits_hdul = GetFits(fits_path).openFits()
    dr7objid = fits_hdul[1].data['dr7objid']
    fits_ra = fits_hdul[1].data['ra']
    fits_dec = fits_hdul[1].data['dec']
    with open(CATALOG + "galaxy_classifier_catalog.csv", "w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, ["dr7objid", "ra", "dec",
                                           "redshift", "redshifterr", "class"])
        # writer.writeheader()
        for i in range(len(redshift_ra)):
            for j, item in enumerate(class_ra):
                if (redshift_ra[i] == class_ra[j]) and (redshift_dec[i] == class_dec[j]):
                    writer.writerow({"dr7objid": dr7objid[i],
                                     "ra": redshift_ra[i],
                                     "dec": redshift_dec[i],
                                     "redshift": redshift[i],
                                     "redshifterr": redshifterr[i],
                                     "class": label[j]})
