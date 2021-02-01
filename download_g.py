# -*- coding: utf-8-*-
import csv
import os

from astropy.io import fits


# CATALOG = "D:\\Code\\MachineLearning\\Data\\2020.12.15_MergerClassifier\\catalog\\"
# DATA = "D:\\Code\\MachineLearning\\Data\\2020.12.15_MergerClassifier\\raw_data\\"

CATALOG = "/mnt/d/Code/MachineLearning/Data/2020.12.15_MergerClassifier/catalog/"
DATA = "/mnt/d/Code/MachineLearning/Data/2020.12.15_MergerClassifier/raw_data/"


def createPackage(model_package: str):
    if not os.path.isdir(model_package):
        os.mkdir(model_package)


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


class GetFits():
    def __init__(self, path):
        self.path = path
        self.hdul = fits.open(self.path)


def download_from_desi(fits: bool, jpg: bool, g: bool, r: bool, z: bool, data_name: str,
                       pix_scale: float, ra: float, dec: float,
                       label: str, dr7objid: str, redshift: float):
    save_dir = DATA
    control = "https://www.legacysurvey.org/viewer/"
    if fits == jpg:
        raise Exception("fits(boolean) cannot equal to jpg(boolean)")
    if fits:
        control += "fits"
        save_dir += "fits/"
    if jpg:
        control += "jpeg"
        save_dir += "jpg/"
    control += "-cutout?ra=%f&dec=%f&layer=dr8&pixscale=%f&bands=" % (ra, dec, pix_scale)
    save_dir += data_name
    createPackage(save_dir)
    if g:
        control += "g' "
        save_dir += "/g"
    if r:
        control += "r' "
        save_dir += "/r"
    if z:
        control += "z' "
        save_dir += "/z"
    createPackage(save_dir)
    url = "wget '" + control + "-O" + save_dir + "/" + "ra=%f-dec=%f-class=%s-dr7objid=%s-redshift=%f" \
          % (ra, dec, label, dr7objid, redshift)
    if fits:
        url += '.fits'
    if jpg:
        url += '.jpg'
    os.system(url)


if __name__ == "__main__":
    pix_scale: float = 0.262
    # ra, dec, label, dr7objid, class_dr7objid, redshift = readCSV(CATALOG + "galaxy_classifier_catalog.csv")
    fits_hdul = GetFits(CATALOG + "galaxy_classifier_catalog.fits")
    ra = fits_hdul.hdul[1].data[:]['ra']
    dec = fits_hdul.hdul[1].data[:]['dec']
    gz2_class = fits_hdul.hdul[1].data[:]['gz2_class']
    dr7objid = fits_hdul.hdul[1].data[:]['dr7objid']
    redshift = fits_hdul.hdul[1].data[:]['redshift']
    if len(ra) == len(dec) == len(gz2_class) == len(dr7objid) == len(redshift):
        for i in range(len(ra)):
            download_from_desi(fits=True, jpg=False, g=True, r=False, z=False,
                               data_name="all_redshift_under_0.1_galaxy_with_class",
                               pix_scale=0.262, ra=float(ra[i]), dec=float(dec[i]),
                               label=gz2_class[i], dr7objid=dr7objid[i], redshift=float(redshift[i]),)
