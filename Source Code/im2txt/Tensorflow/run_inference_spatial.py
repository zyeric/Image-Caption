import vocabulary
import inference_wrapper_spatial
import configuration
import h5py
import tensorflow as tf
import caption_generator_spatial
import math
import numpy as np


def get_image_parts(block_output):
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

vocab = vocabulary.Vocabulary("data/no_segment/dictionary.txt")

check_point_step = 200000

model = inference_wrapper_spatial.InferenceWrapper()
restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               "train_log_spatial_emb_512/{}.ckpt".format(check_point_step))


file = h5py.File("data/image_vgg19_fc1_feature.h5", 'r')
file_block = h5py.File("data/image_vgg19_block5_pool_feature.h5", 'r')

encoded_images = file['validation_set']
images_parts = file_block['validation_set']

sess = tf.InteractiveSession()
restore_fn(sess)

generator = caption_generator_spatial.CaptionGenerator(model, vocab)

# 下面的代码输出了一个beam=3的所有内容
# with open('train_caption_output.txt', 'w') as f:
#     for iter in range(400):
#         index = 2 + iter * 20
#         captions = generator.beam_search(sess, encoded_images[index])
#         f.write("Captions for image {}\r\n".format(index + 1))
#         print("Captions for image {}".format(index + 1))
#         for i, caption in enumerate(captions):
#             # Ignore begin and end words.
#             sentence = [vocab.id_to_word(w - 1) for w in caption.sentence[1:-1]]
#             sentence = " ".join(sentence)
#             f.write("  %d) %s (p=%f)\r\n" % (i, sentence, math.exp(caption.logprob)))
#             print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

# 下面的代码按网络评分格式输出到文件
with open('D:\Image-Caption\Results\\validation_output\\valid_caption_output_spatial_no_segment_{}_emb_512.txt'.format(check_point_step), 'w') as f:
    for index in range(1000):
        captions = generator.beam_search(sess, encoded_images[index], get_image_parts(images_parts[index]))
        caption = captions[0]
        sentence = [vocab.id_to_word(w - 1) for w in caption.sentence[1:-1]]
        full_str = "".join(sentence)
        f.write(repr(8000+index))
        for word in full_str:
            f.write(' {}'.format(word))
        f.write('\n')
        if (index+1) % 100 == 0:
            print(repr(index+1))