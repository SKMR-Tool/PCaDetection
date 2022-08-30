"""
All rights reserved.
--Yang Song (songyangmri@gmail.com)
--2021/1/4
"""
import numpy as np
import pandas as pd

from MeDIT.Log import CustomerCheck
from MeDIT.Others import IterateCase
from MeDIT.SaveAndLoad import LoadImage

def EstimateVolume():

    from CNNModel.SuccessfulModel.ProstateSegment import ProstateSegmentationTrumpetNet

    log = CustomerCheck(r'C:\Users\yangs\Desktop\cad_volume.csv',
                        data=['PCa Num', 'PCa Volume', 'Prostate Volume'],
                        rewrite=True, follow_write=True)

    seg = ProstateSegmentationTrumpetNet()
    seg.LoadConfigAndModel(r'D:\SuccessfulModel\ProstateSegmentTrumpetNet')

    for case in IterateCase(r'w:\PrcoessedData\JSPH_PCa\Total',
                            start_from=r'2017-2018-CA_formal_ZZ^zhang zhi'):
        patient = case.name
        index = patient.index(r'formal_')
        name = patient[index + len('formal_'):]

        prostate_path = case / 'ProstateROI_TrumpetNet.nii.gz'
        if not prostate_path.exists():
            t2_path = case / 't2.nii'
            if not t2_path.exists():
                continue

            seg.Run(t2_path, store_folder=str(case))

        image, prostate, _ = LoadImage(case / 'ProstateROI_TrumpetNet.nii.gz', dtype=int)
        res = image.GetSpacing()
        prostate_volume = prostate.sum() * res[0] * res[1] * res[2]

        pca_candidate = list(case.glob('roi?.nii'))
        for one_pca_path in pca_candidate:
            num = one_pca_path.name[3]
            pca_image, one_pca, _ = LoadImage(one_pca_path, dtype=int)
            res = pca_image.GetSpacing()
            pca_volume = one_pca.sum() * res[0] * res[1] * res[2]

            log.AddOne(name, [num, pca_volume, prostate_volume])

    log.Save()

def SplitCaAndNc():
    all = pd.read_csv(r'w:\PrcoessedData\JSPH_PCa\VolumeStatistics.csv', index_col=0)
    train_df = pd.read_csv(r'w:\PrcoessedData\JSPH_PCa\CADtrain_clinic_utf8.csv', index_col=2, encoding='utf-8')
    test_df = pd.read_csv(r'w:\PrcoessedData\JSPH_PCa\CADtest_estimate.csv', index_col=4)
    test_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    train_nc = train_df[train_df['GS'].isnull()]
    train_ca = train_df[train_df['GS'].notnull()]
    print(train_df.shape, train_nc.shape, train_ca.shape)

    all_train_ca = all.loc[list(set(train_ca.index) & set(all.index))]
    all_train_nc = all.loc[list(set(train_nc.index) & set(all.index))]
    print(all.shape, all_train_ca.shape, all_train_nc.shape)

    test_nc1 = test_df[test_df['GSgroup'].isnull()]
    test_left = test_df[test_df['GSgroup'].notnull()]

    test_left['GSgroup'] = test_left['GSgroup'].astype(int)

    test_ca_index = set(test_df.index) & set((test_left['GSgroup'] > 0).index)
    test_nc2_index = set(test_df.index) & set((test_left['GSgroup'] == 0).index)

    test_nc_index = list(set(test_nc1.index) | test_nc2_index)
    print(len(test_ca_index), len(test_nc_index))

    all_test_ca = all.loc[list(set(test_ca_index) & set(all.index))]
    all_test_nc = all.loc[list(set(test_nc_index) & set(all.index))]
    print(all.shape, all_test_ca.shape, all_test_nc.shape)

    ca = pd.concat((all_train_ca, all_test_ca), axis=0)
    nc = pd.concat((all_train_nc, all_test_nc), axis=0)
    ca.to_csv(r'C:\Users\yangs\Desktop\CA_volume.csv')
    nc.to_csv(r'C:\Users\yangs\Desktop\NC_volume.csv')

SplitCaAndNc()
