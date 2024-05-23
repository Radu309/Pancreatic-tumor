import time
from utils import *


class DataPreprocessor:
    def __init__(self, folds, low_range, high_range):
        self.folds = folds
        self.low_range = low_range
        self.high_range = high_range
        self.image_npy_path = 'data/Pancreas_Segmentation/NPY_Images'
        self.image_path = 'data/Pancreas_Segmentation/train/images'
        self.mask_npy_path = 'data/Pancreas_Segmentation/NPY_Masks'
        self.mask_path = 'data/Pancreas_Segmentation/train/masks'
        self.list_training = 'data/Pancreas_Segmentation/train/training_Z.txt'

    def main(self):
        image_list = []
        image_filename = []
        for directory, _, file_ in os.walk(self.image_npy_path):
            for filename in sorted(file_):
                image_list.append(os.path.join(directory, filename))
                image_filename.append(os.path.splitext(filename)[0])

        mask_list = []
        mask_filename = []
        for directory, _, file_ in os.walk(self.mask_npy_path):
            for filename in sorted(file_):
                mask_list.append(os.path.join(directory, filename))
                mask_filename.append(os.path.splitext(filename)[0])

        if len(image_list) != len(mask_list):
            exit('Error: the number of masks and the number of images are not equal!')

        total_samples = len(image_list)

        directory = os.path.dirname(self.list_training)
        if not os.path.exists(directory):
            os.makedirs(directory)
        output = open(self.list_training, 'w')
        output.close()

        print('Initialization starts.')

        for i in range(total_samples):
            start_time = time.time()
            print('Processing ' + str(i + 1) + ' out of ' + str(len(image_list)) + ' files.')

            image = np.load(image_list[i])
            mask = np.load(mask_list[i])

            # slice_number is the number of slices of corresponding dimension (X/Y/Z)
            # Z for now
            slice_number = mask.shape[2]

            image_directory_ = os.path.join(self.image_path, image_filename[i])
            if not os.path.exists(image_directory_):
                os.makedirs(image_directory_)

            mask_directory_ = os.path.join(self.mask_path, mask_filename[i])
            if not os.path.exists(mask_directory_):
                os.makedirs(mask_directory_)

            # for storing the total number of pixels of ground truth mask
            sum_ = np.zeros(slice_number, dtype=int)
            # for storing bounding boxes of ground truth masks (A_min, A_max, B_min, B_max)
            minA = np.zeros(slice_number, dtype=int)
            maxA = np.zeros(slice_number, dtype=int)
            minB = np.zeros(slice_number, dtype=int)
            maxB = np.zeros(slice_number, dtype=int)
            # for storing mean pixel value of each slice
            average = np.zeros(slice_number, dtype=float)

            # iterate through all slices of current case i and current plane
            for j in range(slice_number):
                # image_filename_ sample dir: image_X /  0001  / 0001.npy
                #                              plane/ case num / slice num
                image_filename_ = os.path.join(
                    image_directory_, '{:0>4}'.format(j) + '.npy')
                mask_filename_ = os.path.join(
                    mask_directory_, '{:0>4}'.format(j) + '.npy')

                image_ = image[:, :, j]
                mask_ = mask[:, :, j]

                # threshold image to specified range
                image_[image_ < self.low_range] = self.low_range
                image_[image_ > self.high_range] = self.high_range

                # save sliced image and mask
                if not os.path.isfile(image_filename_) or not os.path.isfile(mask_filename_):
                    np.save(image_filename_, image_)
                    np.save(mask_filename_, mask_)

                # compute the mean value of the slice
                average[j] = float(image_.sum()) / (image_.shape[0] * image_.shape[1])

                # this is the sum of pixel numbers of a ground truth mask
                sum_[j] = np.count_nonzero(mask_ == 1)
                # record the coordinates of ground truth mask pixels
                arr = np.nonzero(mask_ == 1)

                # save the bounding box of ground truth mask (A_min, A_max, B_min, B_max)
                # delimits for images: A is for rows and B for columns
                minA[j] = np.min(arr[0]) if arr[0].size > 0 else 0
                maxA[j] = np.max(arr[0]) if arr[0].size > 0 else 0
                minB[j] = np.min(arr[1]) if arr[1].size > 0 else 0
                maxB[j] = np.max(arr[1]) if arr[1].size > 0 else 0

            # iterate each slice of current case i
            for j in range(slice_number):
                image_filename_ = os.path.join(
                    image_directory_, '{:0>4}'.format(j) + '.npy')
                mask_filename_ = os.path.join(
                    mask_directory_, '{:0>4}'.format(j) + '.npy')

                # append the following output to training_X/Y/Z.txt
                with open(self.list_training, 'a+') as output:
                    # we need to train images with pixels (100 is the minimum I selected)
                    if sum_[j] >= 100:
                        # case number, slice number
                        output.write(f"{i} {j}")
                        # image file name, mask file name
                        output.write(f" {image_filename_} {mask_filename_}")
                        # average pixel value of slice j, case i, and current plane
                        output.write(f" {average[j]}")
                        # sum of ground truth pixels, and bounding box of gt mask (A_min, A_max, B_min, B_max)
                        output.write(f" {sum_[j]} {minA[j]} {maxA[j]} {minB[j]} {maxB[j]}")
                        output.write('\n')

            print(f"Processed {i + 1} out of {len(image_list)} files: {time.time() - start_time} second(s) elapsed.")

        # create the 4 training image lists
        print('Writing training image list.')
        for f in range(self.folds):
            list_training_ = training_set_filename(f)
            output = open(list_training_, 'w')
            for i in range(total_samples):
                if in_training_set(total_samples, i, self.folds, f):
                    output.write(str(i) + ' ' + image_list[i] + ' ' + mask_list[i] + '\n')
            output.close()

        # create the 4 test image lists
        print('Writing testing image list.')
        for f in range(self.folds):
            list_testing_ = testing_set_filename(f)
            output = open(list_testing_, 'w')
            for i in range(total_samples):
                if not in_training_set(total_samples, i, self.folds, f):
                    output.write(str(i) + ' ' + image_list[i] + ' ' + mask_list[i] + '\n')
            output.close()

        print('Initialization is done.')


if __name__ == '__main__':
    # for more details, check the pipeline file
    # first argv for utils file
    data_path = sys.argv[1]
    folds = int(sys.argv[2])
    low_range = int(sys.argv[3])
    high_range = int(sys.argv[4])

    data_preprocessor = DataPreprocessor(folds, low_range, high_range)
    data_preprocessor.main()
