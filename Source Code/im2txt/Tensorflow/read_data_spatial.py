import h5py
import numpy as np

class DataIterator:
    def __init__(self, encoded_image_path, enocded_block_path, caption_vector_path):

        # structure of h5 file ['test_set', 'train_set', 'validation_set']
        file = h5py.File(encoded_image_path, 'r')

        #file_block = h5py.File("data/image_vgg19_block5_pool_feature.h5", 'r')
        file_block = h5py.File(enocded_block_path, 'r')

        encoded_images = file["train_set"]
        images_block = file_block["train_set"]

        images_and_captions = []
        index = -1
        for line in open(caption_vector_path):
            strs = line.split()
            if len(strs) == 1:
                index = index + 1
                if index == 8000:
                    break
            else:
                nums = []
                for e in strs:
                    nums.append(int(e))

                images_and_captions.append([encoded_images[index], self.get_image_parts(images_block[index]), nums])

        self.images_and_captions = images_and_captions

        self.iter_order = np.random.permutation(len(images_and_captions))
        self.cur_iter_index = 0
        print("Finish loading data")

        print('Training set size is {}'.format(len(images_and_captions)))

    def get_image_parts(self, block_output):
        image00 = np.concatenate((block_output[0][0], block_output[0][1], block_output[1][0], block_output[1][1]), 0)
        image01 = np.concatenate((block_output[0][2], block_output[0][3], block_output[0][4],
                                  block_output[1][2], block_output[1][3], block_output[1][4]), 0)
        image02 = np.concatenate((block_output[0][5], block_output[0][6], block_output[1][5], block_output[1][6]), 0)

        image10 = np.concatenate((block_output[2][0], block_output[2][1],
                                  block_output[3][0], block_output[3][1],
                                  block_output[4][0], block_output[4][1]), 0)
        image11 = np.concatenate((block_output[2][2], block_output[2][3], block_output[2][4],
                                  block_output[3][2], block_output[3][3], block_output[3][4],
                                  block_output[4][2], block_output[4][3], block_output[4][4]), 0)
        image12 = np.concatenate((block_output[2][5], block_output[2][6],
                                  block_output[3][5], block_output[3][6],
                                  block_output[4][5], block_output[4][6]), 0)

        image20 = np.concatenate((block_output[5][0], block_output[5][1],
                                  block_output[6][0], block_output[6][1]), 0)
        image21 = np.concatenate((block_output[5][2], block_output[5][3], block_output[5][4],
                                  block_output[6][2], block_output[6][3], block_output[6][4]), 0)
        image22 = np.concatenate((block_output[5][5], block_output[5][6],
                                  block_output[6][5], block_output[6][6]), 0)

        return [image00, image01, image02, image10, image11, image12, image20, image21, image22]

    def get_image_by_index(self, index):
        _image, _parts_image, _ = self.images_and_captions[index]
        return _image, _parts_image

    def next_batch(self, batch_size):
        if self.cur_iter_index + batch_size >= len(self.images_and_captions):
            self.iter_order = np.random.permutation(len(self.images_and_captions))
            self.cur_iter_index = 0

        images = []
        image_parts = []
        captions = []

        for i in range(batch_size):
            image, image_part, caption = self.images_and_captions[self.cur_iter_index+i]
            images.append(image)
            image_parts.append(image_part)
            captions.append(caption)

        input_seqs, target, masks = self.build_caption_batch(captions)

        self.cur_iter_index = self.cur_iter_index + batch_size
        return images, image_parts, input_seqs, target, masks



    def build_caption_batch(self, captions):
        input_seqs = []
        target = []
        masks = []
        max_len = 0
        for caption in captions:
            if len(caption) > max_len:
                max_len = len(caption)

        max_len = max_len - 1

        for caption in captions:
            want_len = len(caption) - 1
            input_seqs.append(caption[0:want_len] + [0] * (max_len - want_len))
            target.append(caption[1:(want_len + 1)] + [0] * (max_len - want_len))
            masks.append([1] * want_len + [0] * (max_len - want_len))

        return input_seqs, target, masks