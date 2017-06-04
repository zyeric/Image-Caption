from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import configuration
import show_and_tell_model

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

g = tf.Graph()
with g.as_default():
    # Build the model.
    model = show_and_tell_model.ShowAndTellModel(
        model_config, mode="train", train_inception=False)
    model.build()


    learning_rate = tf.constant(training_config.initial_learning_rate)
    if training_config.learning_rate_decay_factor > 0:
        num_batches_per_epoch = (training_config.num_examples_per_epoch /
                                 model_config.batch_size)
        decay_steps = int(num_batches_per_epoch *
                          training_config.num_epochs_per_decay)

        def _learning_rate_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
              learning_rate,
              global_step,
              decay_steps=decay_steps,
              decay_rate=training_config.learning_rate_decay_factor,
              staircase=True)

        learning_rate_decay_fn = _learning_rate_decay_fn

    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=training_config.optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)

    # Set up the Saver for saving and restoring model checkpoints.
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)

tf.contrib.slim.learning.train(
      train_op,
      "train_log",
      log_every_n_steps=1,
      graph=g,
      global_step=model.global_step,
      number_of_steps=100000,
      init_fn=model.init_fn,
      saver=saver)

tf.app.run()