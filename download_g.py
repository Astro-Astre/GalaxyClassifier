# -*- coding: utf-8-*-
import csv
import os
from galaxy_classifier import createPackage

CATALOG = "D:\\Code\\MachineLearning\\Data\\2020.12.15_MergerClassifier\\catalog\\"
DATA = "D:\\Code\\MachineLearning\\Data\\2020.12.15_MergerClassifier\\raw_data\\"


def readCSV(path):
    ra = []
    dec = []
    label = []
    dr7objid = []
    class_dr7objid = []
    redshift = []

    with open(path) as f:
        header = csv.DictReader(f)
        for row in header:
            ra.append(row['ra'])
            dec.append(row['dec'])
            label.append(row['class'])
            dr7objid.append(row['dr7objid'])
            class_dr7objid.append(row['class_dr7objid'])
            redshift.append(row['redshift'])
        return ra, dec, label, dr7objid, class_dr7objid, redshift


def download_from_desi(fits: bool, jpg: bool, g: bool, r: bool, z: bool, data_name: str,
                       pix_scale: float, ra: float, dec: float,
                       label: str, dr7objid: str, redshift: float):
    save_dir = DATA
    control = "https://www.legacysurvey.org/viewer/"
    if fits == jpg:
        raise Exception("fits(boolean) cannot equal to jpg(boolean)")
    if fits:
        control += "fits"
        save_dir += "fits\\"
    if jpg:
        control += "jpeg"
        save_dir += "jpg\\"
    control += "-cutout?ra=%f&dec=%f&layer=dr8&pixscale=%f&bands=" % (ra, dec, pix_scale)
    if g:
        control += "g"
    if r:
        control += "r"
    if z:
        control += "z"
    save_dir += data_name
    createPackage(save_dir)
    url = "wget" + control + data_name + "\\" + "ra=%f&dec=%f&class=%s&dr7objid=%d&redshift=%f" \
          % (ra, dec, label, dr7objid, redshift)
    if fits:
        url += '.fits'
    if jpg:
        url += '.jpg'
    os.system(url)


if __name__ == "__main__":
    pix_scale: float = 0.262
    ra, dec, label, dr7objid, class_dr7objid, redshift = readCSV(CATALOG + "galaxy_classifier_catalog.csv")
    if len(ra) == len(dec) == len(label) == len(dr7objid) == len(class_dr7objid) == len(redshift):
        for i in range(len(ra)):
            download_from_desi(fits=True, jpg=False, g=True, r=False, z=False,
                               data_name="all_redshift<0.1_galaxy_with_class",
                               pix_scale=0.262, ra=float(ra[i]), dec=float(dec[i]),
                               label=label[i], dr7objid=dr7objid[i], redshift=float(redshift[i]),)
