from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import configuration
import show_and_tell_model
import ops.inputs as input_op
import read_data

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()



iter = read_data.DataIterator(encoded_image_path="data/image_vgg19_fc1_feature.h5",
                              dic_path="data/dictionary.txt",
                              caption_vector_path="data/train_vector.txt")

sess = tf.InteractiveSession()

model = show_and_tell_model.ShowAndTellModel(model_config, mode="train", train_inception=False)
model.build()

sess.run(tf.global_variables_initializer())

# images_and_captions = input_op.init_image_embeddings_and_captions(read_size=1000)
#
# batch1_images = []
# batch1_captions = []
# for image, caption in images_and_captions:
#     batch1_images.append(image)
#     batch1_captions.append(caption)
#     if len(batch1_images) == 32:
#         break
# batch1_in_seqs, batch1_tar_seqs, batch1_masks = input_op.build_batch(batch1_captions)

for i in range(10000):
    images, in_seqs, tar_seqs, masks = iter.next_batch(32)
    loss = model.run_batch(sess, images, in_seqs, tar_seqs, masks)
    if i % 100 == 0:
        print(loss)