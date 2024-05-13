import os
import nibabel
import numpy as np
import sys
import pydicom

W, H = 512, 512

ct_masks = 'data/Pancreas_Segmentation/CT_Masks'
npy_masks = 'data/Pancreas_Segmentation/NPY_Masks'

ct_images = 'data/Pancreas_Segmentation/CT_Images'
npy_images = 'data/Pancreas_Segmentation/NPY_Images'


def convert_masks_and_images():
    if not os.path.exists(npy_masks):
        os.makedirs(npy_masks)

    if not os.path.exists(npy_images):
        os.makedirs(npy_images)

    N = sum(1 for f in os.listdir(ct_masks) if f.endswith('.nii.gz'))
    M = sum(os.path.isdir(os.path.join(ct_images, d)) for d in os.listdir(ct_images))

    if N != M:
        print("The number of directories in CT_Masks and CT_Images does not match.")
        sys.exit("Exiting the script due to mismatch.")

    for n in range(N):
        volumeID = '{:0>4}'.format(n + 1)
        filename1 = 'label' + volumeID + '.nii.gz'
        file1 = os.path.join(ct_masks, filename1)

        if not os.path.exists(file1):
            print('File ' + volumeID + ' does not exist, skipping...')
            continue

        print('Processing File ' + volumeID)
        data = nibabel.load(file1).get_fdata().transpose(1, 0, 2)
        print('  Data shape is ' + str(data.shape) + ' .')

        filename2 = volumeID + '.npy'
        file2 = os.path.join(npy_masks, filename2)
        np.save(file2, data)
        print('File ' + volumeID + ' is saved in ' + file2 + ' .')

    for n in range(N):
        volumeID = '{:0>4}'.format(n + 1)
        print('Processing File ' + volumeID)
        filename1 = 'PANCREAS_' + volumeID
        directory1 = os.path.join(ct_images, filename1)

        if not os.path.exists(directory1):
            print('   Director ' + volumeID + ' does not exist, skipping...')
            continue

        filename2 = volumeID + '.npy'
        for path_, _, files in os.walk(directory1):
            L = len(files)
            if L > 0:
                print('  ' + str(L) + ' slices along the axial view.')
                data = np.zeros((W, H, L), dtype=np.int16)
                for f in sorted(files):
                    file1 = os.path.abspath(os.path.join(path_, f))
                    image = pydicom.dcmread(file1)
                    sliceID = image.InstanceNumber - 1
                    if image.pixel_array.shape[0] != 512 or image.pixel_array.shape[1] != 512:
                        exit('  Error: DICOM image does not fit ' + str(W) + 'x' + str(H) + ' size!')
                    data[:, :, sliceID] = image.pixel_array
                file2 = os.path.join(npy_images, filename2)
                np.save(file2, data)
        print('File ' + volumeID + ' is saved in ' + file2 + '.')


def main():
    convert_masks_and_images()


if __name__ == '__main__':
    main()
