# -*- coding: utf-8-*-
import csv

CATALOG = r"D:\\Code\\MachineLearning\\Data\\2020.12.15_MergerClassifier\\catalog\\"


def get_redshift(path: str):
    dr7objid = []
    redshift = []
    ra = []
    dec = []
    redshifterr = []
    with open(path) as f:
        header = csv.DictReader(f)
        for row in header:
            dr7objid.append(int(row['objid']))
            redshift.append(float(row['redshift']))
            ra.append(float(row['ra']))
            dec.append(float(row['dec']))
            redshifterr.append(float(row['redshifterr']))
        return dr7objid, redshift, ra, dec, redshifterr


def get_class(path: str):
    class_dr7objid = []
    label = []
    ra = []
    dec = []
    with open(path) as file:
        header = csv.DictReader(file)
        for row in header:
            label.append(str(row['gz2_class']))
            class_dr7objid.append(int(row['dr7objid']))
            ra.append(float(row['ra']))
            dec.append(float(row['dec']))
        return class_dr7objid, label, ra, dec


if __name__ == '__main__':
    redshift_path: str = CATALOG + "galaxy_redshift_radec.csv"
    class_path: str = CATALOG + "gz2_hart16.csv"
    dr7objid, redshift, redshift_ra, redshift_dec, redshifterr = get_redshift(redshift_path)
    class_dr7objid, label, class_ra, class_dec = get_class(class_path)
    with open(CATALOG + "galaxy_classifier_catalog.csv", "w", newline='') as csv_file:
        writer = csv.DictWriter(csv_file, ["dr7objid", "class_dr7objid", "ra", "dec",
                                           "redshift", "redshifterr", "class"])
        # writer.writeheader()
        for i in range(len(dr7objid)):
            for j, item in enumerate(class_ra):
                if (redshift_ra[i] == class_ra[j]) and (redshift_dec[i] == class_dec[j]):
                    writer.writerow({"dr7objid": dr7objid[i],
                                     "class_dr7objid": class_dr7objid[j],
                                     "ra": redshift_ra[i],
                                     "dec": redshift_dec[i],
                                     "redshift": redshift[i],
                                     "redshifterr": redshifterr[i],
                                     "class": label[j]})
