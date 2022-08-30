# 生成概率图和后处理过的ROI
from pathlib import Path

import os
import shutil
from tqdm import tqdm

from pathlib import Path
import pandas as pd
import numpy as np

from MeDIT.Others import IterateCase, CopyFolder, CopyFile, MakeFolder
from MeDIT.Log import CustomerCheck
from MeDIT.SaveAndLoad import LoadImage
from MeDIT.ArrayProcess import GetRoiRange, ExtractBlock, ExtractPatch
from MeDIT.Normalize import NormalizeZByRoi
from T4T.Utility import FILE_SEPARATOR

from SY.CAD.path_config import nii_folder

def DetectCA():
    from CNNModel.SuccessfulModel.CSProstateCancerDetection import CSDetectTrumpetNetWithROI
    model = CSDetectTrumpetNetWithROI()
    model.LoadConfigAndModel(r'z:\SuccessfulModel\PCaDetectTrumpetNetBlurryROI1500QA')

    for case in IterateCase(nii_folder):
        t2 = case / 't2_Resize.nii'
        adc = case / 'adc_Reg_Resize.nii'
        dwi = list(case.glob('dwi*Reg_Resize.nii'))[0]
        prostate_roi = case / 'ProstateROI_TrumpetNet_Resize.nii.gz'

        assert(t2.exists() and adc.exists() and dwi.exists() and prostate_roi.exists())
        model.Run(str(t2), str(adc), str(dwi), prostate_roi_image=str(prostate_roi),
                  store_folder=str(case / 'PCaBlurryROI_TrumpetNet_DWI1500_Resize.nii.gz'))

# DetectCA()

############################################################################
# 用来生成后续诊断的区域

def GeneratePositivePiradsLabel(source_root, dest_root, prepare_shape):
    pca_roi_folder = MakeFolder(dest_root / 'One_PcaRoi_Slice')
    pirads_value_path = str(dest_root / 'one_pcaroi_pirads.csv')
    if Path(pirads_value_path).exists():
        pirads_value = pd.read_csv(pirads_value_path, index_col=0)
    else:
        pirads_value = pd.DataFrame(columns=['PI-RADS'])

    for case in IterateCase(source_root, start_from='2017-2018-CA_formal_CHEN ZHANG LIANG'):
        if 'add-NC' in case.name:
            continue

        prostate_roi = LoadImage(case / 'ProstateROI_TrumpetNet_Resize.nii.gz', dtype=int)[1]

        pca_roi_candidate = sorted(list(case.glob('roi?_Resize.nii')))
        for roi_index, pca_path in enumerate(pca_roi_candidate):
            info_path = case / 'roi{}_info.csv'.format(roi_index)
            if not info_path.exists():
                continue

            df = pd.read_csv(str(case / 'roi{}_info.csv'.format(roi_index)), index_col=0)
            value = df.loc['PI-RADS'][0]

            one_pca_roi = LoadImage(str(pca_path))[1]
            _, _, slice_index_list = GetRoiRange(one_pca_roi)

            if len(slice_index_list) == 0:
                continue

            for slice_index in slice_index_list:
                if slice_index == 0 or slice_index == one_pca_roi.shape[-1] - 1:
                    continue

                x_list, y_list, _ = GetRoiRange(prostate_roi)
                center_x = (max(x_list) + min(x_list)) // 2
                center_y = (max(y_list) + min(y_list)) // 2

                pca_roi_block = ExtractBlock(one_pca_roi, prepare_shape, center_point=[center_x, center_y,
                                                                                     slice_index])[0]
                pca_roi_block = pca_roi_block.transpose((2, 0, 1))

                name = '{}{}{}{}{}'.format(case.name, FILE_SEPARATOR,
                                           'slice{}'.format(slice_index), FILE_SEPARATOR,
                                           'roi{}'.format(roi_index)
                                           )

                np.save(str(pca_roi_folder / '{}.npy'.format(name)), pca_roi_block)

                pirads_value.loc[name] = value
        pirads_value.to_csv(pirads_value_path)


# GeneratePositivePiradsLabel(Path(r'w:\PrcoessedData\JSPH_PCa\Total'),
#                             Path(r'w:\CNNFormatData\CsPCaDetectionTrain'),
#                             prepare_shape=[240, 240, 3])

def BinaryCAD():
    from SY.CAD.Probability2Mask import Prob2Mask
    from MeDIT.SaveAndLoad import SaveNiiImage

    prob_mask = Prob2Mask(ignore_volume=np.pi * 4. * (5. ** 3) / 3.)

    for case in IterateCase(nii_folder):
        pred_image, pred, ref = LoadImage(str(case / 'PCaBlurryROI_TrumpetNet_DWI1500_Resize.nii.gz'))
        prostate_image, prostate, _, = LoadImage(case / 'ProstateROI_TrumpetNet_Resize.nii.gz', dtype=int)
        segs = prob_mask.Run(pred_image, prostate_roi=prostate)

        pred_values = segs.keys()
        pred_values = sorted(pred_values, reverse=True)
        print(pred_values)
        count = 0
        for pred in pred_values:
            if count == 3:
                break

            mask = segs[pred]
            mask_image = ref.BackToImage(mask)

            store_path = case / 'PCaBlurryROI_TrumpetNet_DWI1500_Resize_{}_{:.3f}.nii.gz'.format(count, pred)
            SaveNiiImage(store_path, mask_image)
            count += 1

# BinaryCAD()

def GenerateDetectPiradsLabel(source_root, dest_root, prepare_shape):
    pca_roi_folder = MakeFolder(dest_root / 'One_DetectRoi_Slice')
    pirads_value_path = str(dest_root / 'one_detectroi_pirads.csv')
    if Path(pirads_value_path).exists():
        pirads_value = pd.read_csv(pirads_value_path, index_col=0)
    else:
        pirads_value = pd.DataFrame(columns=['PI-RADS'])

    for case in IterateCase(source_root):
        if 'add-NC' in case.name:
            continue

        prostate_roi = LoadImage(case / 'ProstateROI_TrumpetNet_Resize.nii.gz', dtype=int)[1]
        x_list, y_list, _ = GetRoiRange(prostate_roi)
        center_x = (max(x_list) + min(x_list)) // 2
        center_y = (max(y_list) + min(y_list)) // 2

        detect_roi_candidate = sorted(list(case.glob('PCaBlurryROI_TrumpetNet_DWI1500_Resize_?_*.nii.gz')))
        pca_roi_candidate = sorted(list(case.glob('roi?_Resize.nii')))

        for roi_index, detect_roi_path in enumerate(detect_roi_candidate):
            one_detect_roi = LoadImage(str(detect_roi_path))[1]
            is_skip = False

            for _, pca_path in enumerate(pca_roi_candidate):
                one_pca_roi = LoadImage(str(pca_path))[1]
                if (one_detect_roi * one_pca_roi).sum() / one_pca_roi.sum() < 0.2 and \
                    (one_detect_roi * one_pca_roi).sum() / one_detect_roi.sum() < 0.2:
                    # 该ROI应该在有记录当中
                    is_skip = True
                    break

            if is_skip:
                _, _, slice_index_list = GetRoiRange(one_detect_roi)

                if len(slice_index_list) == 0:
                    continue

                areas = {}
                for slice_index in slice_index_list:
                    if slice_index == 0 or slice_index == one_detect_roi.shape[-1] - 1:
                        continue

                    pca_roi_block = ExtractBlock(one_detect_roi, prepare_shape,
                                                 center_point=(center_x, center_y, slice_index))[0]
                    pca_roi_block = pca_roi_block.transpose((2, 0, 1))
                    areas[slice_index] = pca_roi_block.sum()

                areas = {k: v for k, v in sorted(areas.items(), key=lambda item: item[1], reverse=True)}

                for detect_roi_index, (slice_index, _) in enumerate(areas.items()):
                    if detect_roi_index == 3:
                        break
                    pca_roi_block = ExtractBlock(one_detect_roi, prepare_shape,
                                                 center_point=(center_x, center_y, slice_index))[0]
                    pca_roi_block = pca_roi_block.transpose((2, 0, 1))
                    name = '{}{}{}{}{}'.format(case.name, FILE_SEPARATOR,
                                               'slice{}'.format(slice_index), FILE_SEPARATOR,
                                               'detectroi{}'.format(roi_index)
                                               )

                    np.save(str(pca_roi_folder / '{}.npy'.format(name)), pca_roi_block)

                    pirads_value.loc[name] = 2

        pirads_value.to_csv(pirads_value_path)

# GenerateDetectPiradsLabel(Path(r'w:\PrcoessedData\JSPH_PCa\Total'),
#                             Path(r'w:\CNNFormatData\CsPCaDetectionTrain'),
#                             prepare_shape=[240, 240, 3])

# 用来生成jpg文件，用以发给医生进行check
def _GenerateCheckFigure():
    from MeDIT.SaveAndLoad import SaveArrayAsImage
    from MeDIT.Visualization import MergeImageWithRoi
    from MeDIT.Normalize import Normalize01
    def _MergeDataAndRoi(data, roi):
        data = Normalize01(data)
        merge = []
        for slice_index in range(data.shape[0]):
            merge.append(MergeImageWithRoi(data[slice_index, ...], roi[slice_index, ...]))
        return np.concatenate((merge), axis=1)

    roi_folder = Path(r'w:\CNNFormatData\CsPCaDetectionTrain\One_DetectRoi_Slice')
    t2_folder = Path('w:\CNNFormatData\CsPCaDetectionTrain\T2Slice3')
    adc_folder = Path('w:\CNNFormatData\CsPCaDetectionTrain\AdcSlice3')
    dwi_folder = Path('w:\CNNFormatData\CsPCaDetectionTrain\DwiSlice3')

    store_folder = Path(r'w:\Temp\DetectRoiCheck')

    for case in tqdm(roi_folder.iterdir()):
        case_slice = case.name.split(FILE_SEPARATOR)[:-1]

        t2 = list(t2_folder.glob('{}*.npy'.format(FILE_SEPARATOR.join(case_slice))))
        if len(t2) != 1:
            continue

        adc = list(adc_folder.glob('{}*.npy'.format(FILE_SEPARATOR.join(case_slice))))
        if len(adc) != 1:
            continue

        dwi = list(dwi_folder.glob('{}*.npy'.format(FILE_SEPARATOR.join(case_slice))))
        if len(dwi) != 1:
            continue

        roi = np.load(str(case)).astype(int)
        t2, adc, dwi = np.load(str(t2[0])), np.load(str(adc[0])), np.load(str(dwi[0]))
        t2 = _MergeDataAndRoi(t2, roi)
        adc = _MergeDataAndRoi(adc, roi)
        dwi = _MergeDataAndRoi(dwi, roi)
        show = np.concatenate((t2, adc, dwi), axis=0)

        name = store_folder / '{}.jpg'.format(case.name[:-4])
        SaveArrayAsImage(show, str(name), dpi=(96, 96), format='jpeg', size=10)


_GenerateCheckFigure()