import h5py
import numpy as np

class DataIterator:
    def __init__(self, encoded_image_path, caption_vector_path):

        # structure of h5 file ['test_set', 'train_set', 'validation_set']
        file = h5py.File(encoded_image_path, 'r')
        encoded_images = file['train_set']

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
                images_and_captions.append([encoded_images[index], nums])

        self.images_and_captions = images_and_captions

        self.iter_order = np.random.permutation(len(images_and_captions))
        self.cur_iter_index = 0
        print("Finish loading data")

        print('Training set size is {}'.format(len(images_and_captions)))

    def next_batch(self, batch_size):
        if self.cur_iter_index + batch_size >= len(self.images_and_captions):
            self.iter_order = np.random.permutation(len(self.images_and_captions))
            self.cur_iter_index = 0

        images = []
        captions = []

        for i in range(batch_size):
            image, caption = self.images_and_captions[self.cur_iter_index+i]
            images.append(image)
            captions.append(caption)

        input_seqs, target, masks = self.build_caption_batch(captions)

        self.cur_iter_index = self.cur_iter_index + batch_size
        return images, input_seqs, target, masks


    # transform raw captions to three parts
    # an input caption meets following requirements:
    #   1) start with 1 (the start flag)
    #   2) end with 2 (the end flag)
    # assume the batch size is 4, and the captions are:
    # [1, 3, 4, 5, 6, 2]
    # [1, 5, 6, 7, 2]
    # [1, 5, 3, 2]
    # [1, 7, 9, 10, 2]
    # then the outputs are
    # input_seqs:
    # [1, 3, 4, 5, 6]
    # [1, 5, 6, 7, 0]
    # [1, 5, 3, 0, 0]
    # [1, 7, 9, 10, 0]
    # target_seqs:
    # [3, 4, 5, 6, 2]
    # [5, 6, 7, 2, 0]
    # [5, 3, 2, 0, 0]
    # [7, 9, 10, 2, 0]
    # input_masks:
    # [1, 1, 1, 1, 1]
    # [1, 1, 1, 1, 0]
    # [1, 1, 1, 0, 0]
    # [1. 1, 1, 1, 0]
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