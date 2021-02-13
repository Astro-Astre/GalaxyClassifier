# ,*, coding: utf,8,*,
import os
from astropy.io import fits

CATALOG = "D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/catalog/"
DATA = "D:/Code/MachineLearning/Data/2020.12.15_MergerClassifier/raw_data/"


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
    for i in range(len(object_dir)):
    # for i in range(10):
        ra.append(float(object_dir[i].split(',')[0].split('=')[1]))
        dec.append(float(object_dir[i].split(',')[1].split('=')[1]))
        gz2_fuzzy_class.append(object_dir[i].split(',')[2].split('=')[1])
        dr7objid.append(object_dir[i].split(',')[3].split('=')[1])
        redshift.append(float(object_dir[i].split(',')[4].split('=')[1].split('.fits')[0]))
    fits_file_name = CATALOG + "some_galaxy_fuzzy_classifier_catalog.fits"
    hdu = fits.BinTableHDU.from_columns([fits.Column(name='dr7objid', format='K', array=dr7objid),
                                         fits.Column(name='ra', format='D', array=ra),
                                         fits.Column(name='dec', format='D', array=dec),
                                         fits.Column(name='redshift', format='D', array=redshift),
                                         fits.Column(name='gz2_fuzzy_class', format='44A', array=gz2_fuzzy_class),
                                         ])
    hdu.writeto(fits_file_name)


if __name__ == '__main__':
    data_path = DATA + 'fits/all_redshift_under_0.1_galaxy_with_class/'
    get_info(data_path + 'g')
