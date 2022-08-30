'''
对于一个特定的癌灶，缩小一圈，计算平均概率
然后将此癌灶放大一圈，概率图减去，通过阈值从高到底进行设计，看假阳性增加情况。
'''

import math
from copy import deepcopy

import numpy as np
from itertools import product
from scipy import ndimage
from scipy.cluster.hierarchy import fcluster, linkage
from collections import OrderedDict
from skimage.morphology import binary_opening, binary_dilation
from scipy.ndimage import label

from MeDIT.ImageProcess import GetDataFromSimpleITK

physical_distance_square = lambda point_1, point_2, resolution: \
            np.sum(np.square((np.asarray(point_1) - np.asarray(point_2)) * np.asarray(resolution)))
distance = lambda x, y: np.sum(np.square(np.asarray(x) - np.asarray(y)))



class SegmentPrediction(object):
    def __init__(self, threshold='local', ratio=0.5):
        self.threhsold = threshold
        self.ratio = ratio
        pass

    def GetNeighbourhood(self, pt, checked, dims):
        nbhd = []

        if (pt[0] > 0) and not checked[pt[0]-1, pt[1], pt[2]]:
            nbhd.append((pt[0]-1, pt[1], pt[2]))
        if (pt[1] > 0) and not checked[pt[0], pt[1]-1, pt[2]]:
            nbhd.append((pt[0], pt[1]-1, pt[2]))
        if (pt[2] > 0) and not checked[pt[0], pt[1], pt[2]-1]:
            nbhd.append((pt[0], pt[1], pt[2]-1))

        if (pt[0] < dims[0]-1) and not checked[pt[0]+1, pt[1], pt[2]]:
            nbhd.append((pt[0]+1, pt[1], pt[2]))
        if (pt[1] < dims[1]-1) and not checked[pt[0], pt[1]+1, pt[2]]:
            nbhd.append((pt[0], pt[1]+1, pt[2]))
        if (pt[2] < dims[2]-1) and not checked[pt[0], pt[1], pt[2]+1]:
            nbhd.append((pt[0], pt[1], pt[2]+1))

        return nbhd

    def Grow(self, img, seed, region_shape=(3, 3, 3)):
        """
        img: ndarray, ndim=3
            An image volume.

        seed: tuple, len=3
            Region growing starts from this point.

        t: int
            The image neighborhood radius for the inclusion criteria.
        """
        seg = np.zeros(img.shape, dtype=np.bool)
        checked = np.zeros_like(seg)
        target_value = img[seed]
        max_value = deepcopy(target_value)

        seg[seed] = True
        checked[seed] = True
        needs_check = self.GetNeighbourhood(seed, checked, img.shape)

        value = 0
        if self.threhsold == 'seed':
            value = target_value * self.ratio

        while len(needs_check) > 0:

            pt = needs_check.pop()

            # Its possible that the point was already checked and was
            # put in the needs_check stack multiple times.
            if checked[pt]: continue

            checked[pt] = True

            if self.threhsold == 'local':
                # Handle borders.
                imin = max(pt[0] - region_shape[0], 0)
                imax = min(pt[0] + region_shape[0], img.shape[0] - 1)
                jmin = max(pt[1] - region_shape[1], 0)
                jmax = min(pt[1] + region_shape[1], img.shape[1] - 1)
                kmin = max(pt[2] - region_shape[2], 0)
                kmax = min(pt[2] + region_shape[2], img.shape[2] - 1)
                value = img[imin:imax + 1, jmin:jmax + 1, kmin:kmax + 1].mean()

            if img[pt] >= value and img[pt] > 1e-3 and img[pt] <= max_value:
                seg[pt] = True
                needs_check += self.GetNeighbourhood(pt, checked, img.shape)

        return seg

    def BinaryPrediction(self, image, seed_list, region_shape=(3, 3, 3)):
        final_mask = np.zeros(image.shape)

        count = 1

        sem1 = np.asarray([[0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0]])
        sem2 = np.asarray([[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0],
                           [0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]])
        sem = np.stack((sem2, sem2), axis=2)
        while seed_list:
            seed = seed_list.pop(0)
            # print('Current seed: {}, current probability: {:.3f}'.format(seed, image[seed]))

            ignore_region_mask = np.asarray(final_mask, dtype=bool)
            ignore_region_mask = binary_dilation(ignore_region_mask, selem=np.ones((7, 7, 3)))
            ignore_region_mask = np.asarray(ignore_region_mask, dtype=int)

            seed_list = [seed for seed in seed_list if ignore_region_mask[seed] != 1]

            segment = self.Grow(image * (1 - ignore_region_mask), seed, region_shape)
            if segment.astype(int).sum() == 1:
                continue
            # print(segment.astype(int).sum())

            open_segment = binary_opening(deepcopy(segment), sem)
            if open_segment.astype(int).sum() == 0:
                open_segment = segment
            label_mask, label_value = label(open_segment)

            if label_mask[seed] != 0:
                final_mask[label_mask == label_mask[seed]] = count
            elif (label_mask == label_mask[seed]).astype(int).sum() < 10000:
                final_mask[segment] = count
            else:
                continue
            # print((label_mask == label_mask[seed]).astype(int).sum())

            count += 1

        return final_mask


class RegionEstimation():
    def __init__(self):
        self.diameter = 10
        self.prediction = np.array([])
        self.resolution = ()

    def GenerateSphereFileter(self):
        rows, columns, slices = math.ceil(self.diameter / 2 / self.resolution[0]) * 2 + 1, \
                                math.ceil(self.diameter / 2 / self.resolution[1]) * 2 + 1, \
                                math.ceil(self.diameter / 2 / self.resolution[2]) * 2 + 1
        center_point = (math.ceil(self.diameter / 2 / self.resolution[0]),
                        math.ceil(self.diameter / 2 / self.resolution[1]),
                        math.ceil(self.diameter / 2 / self.resolution[2]))
        sphere_filter = np.zeros((rows, columns, slices))
        for x, y, z in product(range(rows), range(columns), range(slices)):
            if physical_distance_square((x, y, z), center_point, self.resolution) < (self.diameter / 2) ** 2:
                sphere_filter[x, y, z] = 1
        return sphere_filter

    def MergePosition(self, probablility_location_pair):
        remain_probability_localtion_pair = deepcopy(probablility_location_pair)
        removed_location = []
        for target_location, target_threshold in sorted(remain_probability_localtion_pair.items(), key=lambda k: k[1]):
            for location, threshold in probablility_location_pair.items():
                if (target_threshold < threshold) and (
                        physical_distance_square(target_location, location, self.resolution) < self.diameter ** 2):
                    remain_probability_localtion_pair[target_location] = threshold
                    removed_location.append(target_location)

        for location in removed_location:
            remain_probability_localtion_pair.pop(location, None)

        remain_probability_localtion_pair = OrderedDict(
            sorted(remain_probability_localtion_pair.items(), key=lambda k: k[1]))
        return remain_probability_localtion_pair

    def SplitRegions(self, location_list):
        # 通过分析所有的坐标点，区分区域
        isolate_location_list = []
        for location in location_list:
            isolate_location_list.append([location[0] * self.resolution[0],
                                          location[1] * self.resolution[1],
                                          location[2] * self.resolution[2]])
        z = linkage(isolate_location_list, method='single')
        corresponding_region = fcluster(z, t=self.diameter * 1.5, criterion='distance')
        return corresponding_region

    def GetCanidateRegion(self, filter, prostate_roi):
        result = ndimage.filters.maximum_filter(self.prediction, footprint=filter)
        diff_pred = np.asarray(((self.prediction - result) * prostate_roi > -1e-8), dtype=int)

        local_candidate_x, local_candidate_y, local_candidate_z = \
            np.where((prostate_roi * diff_pred) == 1)
        probability_location_pair = {}
        for x, y, z in zip(local_candidate_x, local_candidate_y, local_candidate_z):
            probability_location_pair[(x, y, z)] = self.prediction[x, y, z]
        probability_location_pair = OrderedDict(sorted(probability_location_pair.items(), key=lambda k: k[1]))

        region_count = len(set(probability_location_pair.values()))
        while True:
            probability_location_pair = self.MergePosition(probability_location_pair)
            if region_count == len(set(probability_location_pair.values())):
                break
            region_count = deepcopy(len(set(probability_location_pair.values())))

        return probability_location_pair

    def GetSeedList(self, probability_location_pair):
        localtions = probability_location_pair.keys()
        region_part = self.SplitRegions(localtions)

        seed_list = []
        for one_part in set(region_part):
            region_location = [list(localtions)[index] for index in np.where(region_part == one_part)[0].tolist()]

            seed = ()
            max_probability, max_local = 0, ()
            for region in region_location:
                if probability_location_pair[region] > max_probability:
                    seed = region
                    max_probability = probability_location_pair[region]
            if max_probability < 0.1:
                continue
            seed_list.append(seed)

        # 对seed list 根据概率值排序
        seed_list = sorted(seed_list, key=lambda x: probability_location_pair[x], reverse=True)
        return seed_list


class Prob2Mask(object):
    def __init__(self, ignore_volume=-1.):
        self.region_estimation = RegionEstimation()
        self.prediction_segmentation = SegmentPrediction(threshold='seed')
        self.ignore_volume = ignore_volume

    def Run(self, prediction_image, prostate_roi):
        prediction, ref = GetDataFromSimpleITK(prediction_image)

        self.region_estimation.prediction = prediction
        self.region_estimation.resolution = prediction_image.GetSpacing()

        probability_location_pair = self.region_estimation.GetCanidateRegion(
            self.region_estimation.GenerateSphereFileter(), prostate_roi)
        seed_list = self.region_estimation.GetSeedList(probability_location_pair)

        all_seg = self.prediction_segmentation.BinaryPrediction(prediction, seed_list)

        ignore_voxels = np.prod(prediction_image.GetSpacing()) * self.ignore_volume
        segs = {}
        label_im, nb_labels = ndimage.label(all_seg)
        for i in range(1, nb_labels + 1):
            one_mask = (label_im == i).astype(int)

            if one_mask.sum() < ignore_voxels:
                continue

            pred = prediction[one_mask == 1].max()
            segs[pred] = one_mask

        return OrderedDict(sorted(segs.items()))

if __name__ == '__main__':
    from MeDIT.SaveAndLoad import LoadImage
    from MeDIT.Visualization import Imshow3DArray, Merge3DImageWithRoi, FlattenImages

    prob_image, prob, _ = LoadImage(r'w:\PrcoessedData\JSPH_PCa\Total\2012-2016-CA_formal_CFC^chen fu '
                           r'can\PCaBlurryROI_TrumpetNet_DWI1500_Resize.nii.gz')
    _, prostate, _ = LoadImage(r'w:\PrcoessedData\JSPH_PCa\Total\2012-2016-CA_formal_CFC^chen fu '
                        r'can\ProstateROI_TrumpetNet_Resize.nii.gz')
    pca_label = LoadImage(r'w:\PrcoessedData\JSPH_PCa\Total\2012-2016-CA_formal_CFC^chen fu can\roi0_Resize.nii',
                      dtype=int)[1]

    prob_mask = Prob2Mask()
    segs = prob_mask.Run(prob_image, prostate_roi=prostate)

    show_images = []
    for key, mask in segs.items():
        show_images.append(Merge3DImageWithRoi(prob, [mask, prostate, pca_label]))
    result = FlattenImages(show_images, is_show=True)
    print(result.shape)
    Imshow3DArray(result)
