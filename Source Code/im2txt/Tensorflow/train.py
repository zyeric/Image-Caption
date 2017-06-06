from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import configuration
import show_and_tell_model
import read_data

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()



iter = read_data.DataIterator(encoded_image_path="data/image_vgg19_fc1_feature.h5",
                              caption_vector_path="data/train_vector.txt")

sess = tf.InteractiveSession()

model = show_and_tell_model.ShowAndTellModel(model_config, mode="train", train_inception=False)
model.build()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

loss_stored = []

# step = epoch * train_data_set / batch_size =
# current 10 epoch

for i in range(500):
    images, in_seqs, tar_seqs, masks = iter.next_batch(model_config.batch_size)
    loss = model.run_batch(sess, images, in_seqs, tar_seqs, masks)
    #every 100 steps print loss value
    if (i+1) % 100 == 0:
        print('step: {}, loss: {}'.format(i+1, loss))
        loss_stored.append(loss)

    #every 1000 steps save check-point file
    if (i+1) % 500 == 0:
        save_path = saver.save(sess, 'train_log/{}.ckpt'.format(i+1))

with open('train_log/loss.txt', 'w') as f:
    for e in loss_stored:
        f.write(repr(e))
        f.write('\n')