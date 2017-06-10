"""Image-to-text implementation based on http://arxiv.org/abs/1411.4555.
"Show and Tell: A Neural Image Caption Generator"
Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


import tensorflow as tf


class ShowAndTellModel(object):

  def __init__(self, config, mode, train_inception=False):
    """Basic setup.
    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "inference".
      train_inception: Whether the inception submodel variables are trainable.
    """
    assert mode in ["train", "eval", "inference"]
    self.config = config
    self.mode = mode
    self.train_inception = train_inception

    # Reader for the input data.
    self.reader = tf.TFRecordReader()

    # To match the "Show and Tell" paper we initialize all variables with a
    # random uniform initializer.
    self.initializer = tf.random_uniform_initializer(
        minval=-self.config.initializer_scale,
        maxval=self.config.initializer_scale)

    # A float32 Tensor with shape [batch_size, height, width, channels].
    self.images = None

    self.images00 = None
    self.images01 = None
    self.images02 = None
    self.images10 = None
    self.images11 = None
    self.images12 = None
    self.images20 = None
    self.images21 = None
    self.images22 = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 Tensor with shape [batch_size, padded_length].
    self.target_seqs = None

    # An int32 0/1 Tensor with shape [batch_size, padded_length].
    self.input_mask = None

    # A float32 Tensor with shape [batch_size, embedding_size].
    self.image_embeddings = None

    # A float32 Tensor with shape [batch_size, padded_length, embedding_size].
    self.seq_embeddings = None

    # A float32 scalar Tensor; the total loss for the trainer to optimize.
    self.total_loss = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_losses = None

    # A float32 Tensor with shape [batch_size * padded_length].
    self.target_cross_entropy_loss_weights = None

    # Function to restore the inception submodel from checkpoint.
    self.init_fn = None

    # Global step Tensor.
    self.global_step = None

    self.train_op = None

  def is_training(self):
    """Returns true if the model is built for training mode."""
    return self.mode == "train"

  def build_inputs(self):
    """Input prefetching, preprocessing and batching.
    Outputs:
      self.images
      self.input_seqs
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    """
    if self.mode == "inference":
      # In inference mode, images and inputs are fed via placeholders.
      image_feed = tf.placeholder(dtype=tf.float32, shape=[4096], name="image_feed")

      image00_feed = tf.placeholder(dtype=tf.float32, shape=[2048], name="image00_feed")
      image01_feed = tf.placeholder(dtype=tf.float32, shape=[3072], name="image01_feed")
      image02_feed = tf.placeholder(dtype=tf.float32, shape=[2048], name="image02_feed")
      image10_feed = tf.placeholder(dtype=tf.float32, shape=[3072], name="image10_feed")
      image11_feed = tf.placeholder(dtype=tf.float32, shape=[4608], name="image11_feed")
      image12_feed = tf.placeholder(dtype=tf.float32, shape=[3072], name="image12_feed")
      image20_feed = tf.placeholder(dtype=tf.float32, shape=[2048], name="image20_feed")
      image21_feed = tf.placeholder(dtype=tf.float32, shape=[3072], name="image21_feed")
      image22_feed = tf.placeholder(dtype=tf.float32, shape=[2048], name="image22_feed")

      input_feed = tf.placeholder(dtype=tf.int64,
                                  shape=[None],  # batch_size
                                  name="input_feed")

      # insert batch dimensions.
      images = tf.expand_dims(image_feed, 0)
      images00 = tf.expand_dims(image00_feed, 0)
      images01 = tf.expand_dims(image01_feed, 0)
      images02 = tf.expand_dims(image02_feed, 0)
      images10 = tf.expand_dims(image10_feed, 0)
      images11 = tf.expand_dims(image11_feed, 0)
      images12 = tf.expand_dims(image12_feed, 0)
      images20 = tf.expand_dims(image20_feed, 0)
      images21 = tf.expand_dims(image21_feed, 0)
      images22 = tf.expand_dims(image22_feed, 0)

      input_seqs = tf.expand_dims(input_feed, 1)

      # No target sequences or input mask in inference mode.
      target_seqs = None
      input_mask = None

      self.images = images
      self.images00 = images00
      self.images01 = images01
      self.images02 = images02
      self.images10 = images10
      self.images11 = images11
      self.images12 = images12
      self.images20 = images20
      self.images21 = images21
      self.images22 = images22

      self.input_seqs = input_seqs
      self.input_mask = input_mask
      self.target_seqs = target_seqs
    else:
      self.images = tf.placeholder(tf.float32, shape=[self.config.batch_size, 4096])

      # 9 parts of image
      self.images00 = tf.placeholder(tf.float32, shape=[self.config.batch_size, 2048])
      self.images01 = tf.placeholder(tf.float32, shape=[self.config.batch_size, 3072])
      self.images02 = tf.placeholder(tf.float32, shape=[self.config.batch_size, 2048])
      self.images10 = tf.placeholder(tf.float32, shape=[self.config.batch_size, 3072])
      self.images11 = tf.placeholder(tf.float32, shape=[self.config.batch_size, 4608])
      self.images12 = tf.placeholder(tf.float32, shape=[self.config.batch_size, 3072])
      self.images20 = tf.placeholder(tf.float32, shape=[self.config.batch_size, 2048])
      self.images21 = tf.placeholder(tf.float32, shape=[self.config.batch_size, 3072])
      self.images22 = tf.placeholder(tf.float32, shape=[self.config.batch_size, 2048])


      self.input_seqs = tf.placeholder(tf.int32, shape=[self.config.batch_size, None])
      self.target_seqs = tf.placeholder(tf.int32, shape=[self.config.batch_size, None])
      self.input_mask = tf.placeholder(tf.int32, shape=[self.config.batch_size, None])


  def build_image_embeddings(self):

      # Map inception output into embedding space.
      with tf.variable_scope("whole_image_embedding") as scope:
          whole_image_embeddings = tf.contrib.layers.fully_connected(
              inputs=self.images,
              num_outputs=256,
              activation_fn=None,
              weights_initializer=self.initializer,
              biases_initializer=None,
              scope=scope)

      with tf.variable_scope("image11_embedding") as scope:
          image11_embeddings = tf.contrib.layers.fully_connected(
              inputs=self.images11,
              num_outputs=128,
              activation_fn=None,
              weights_initializer=self.initializer,
              biases_initializer=None,
              scope=scope)

      with tf.variable_scope("image00_embedding") as scope:
          image00_embeddings = tf.contrib.layers.fully_connected(
              inputs=self.images00,
              num_outputs=32,
              activation_fn=None,
              weights_initializer=self.initializer,
              biases_initializer=None,
              scope=scope)

      with tf.variable_scope("image02_embedding") as scope:
          image02_embeddings = tf.contrib.layers.fully_connected(
            inputs=self.images02,
            num_outputs=32,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)

      with tf.variable_scope("image20_embedding") as scope:
          image20_embeddings = tf.contrib.layers.fully_connected(
            inputs=self.images20,
            num_outputs=32,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)

      with tf.variable_scope("image22_embedding") as scope:
          image22_embeddings = tf.contrib.layers.fully_connected(
            inputs=self.images22,
            num_outputs=32,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)

      with tf.variable_scope("image01_embedding") as scope:
          image01_embeddings = tf.contrib.layers.fully_connected(
            inputs=self.images01,
            num_outputs=64,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)

      with tf.variable_scope("image10_embedding") as scope:
          image10_embeddings = tf.contrib.layers.fully_connected(
            inputs=self.images10,
            num_outputs=64,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)

      with tf.variable_scope("image12_embedding") as scope:
          image12_embeddings = tf.contrib.layers.fully_connected(
            inputs=self.images12,
            num_outputs=64,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)

      with tf.variable_scope("image21_embedding") as scope:
          image21_embeddings = tf.contrib.layers.fully_connected(
            inputs=self.images21,
            num_outputs=64,
            activation_fn=None,
            weights_initializer=self.initializer,
            biases_initializer=None,
            scope=scope)

      image_embeddings = tf.concat([whole_image_embeddings,
                                        image00_embeddings, image01_embeddings, image02_embeddings,
                                        image10_embeddings, image11_embeddings, image12_embeddings,
                                        image20_embeddings, image21_embeddings, image22_embeddings], 1)

      # Save the embedding size in the graph.
      tf.constant(self.config.embedding_size, name="embedding_size")

      self.image_embeddings = image_embeddings

  def build_seq_embeddings(self):
    """Builds the input sequence embeddings.
    Inputs:
      self.input_seqs
    Outputs:
      self.seq_embeddings
    """
    with tf.variable_scope("seq_embedding"), tf.device("/cpu:0"):
      embedding_map = tf.get_variable(
          name="map",
          shape=[self.config.vocab_size, self.config.embedding_size],
          initializer=self.initializer)
      seq_embeddings = tf.nn.embedding_lookup(embedding_map, self.input_seqs)

    self.seq_embeddings = seq_embeddings

  def build_model(self):
    """Builds the model.
    Inputs:
      self.image_embeddings
      self.seq_embeddings
      self.target_seqs (training and eval only)
      self.input_mask (training and eval only)
    Outputs:
      self.total_loss (training and eval only)
      self.target_cross_entropy_losses (training and eval only)
      self.target_cross_entropy_loss_weights (training and eval only)
    """
    # This LSTM cell has biases and outputs tanh(new_c) * sigmoid(o), but the
    # modified LSTM in the "Show and Tell" paper has no biases and outputs
    # new_c * sigmoid(o).
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=self.config.num_lstm_units, state_is_tuple=True)
    if self.mode == "train":
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
          lstm_cell,
          input_keep_prob=self.config.lstm_dropout_keep_prob,
          output_keep_prob=self.config.lstm_dropout_keep_prob)

    with tf.variable_scope("lstm", initializer=self.initializer) as lstm_scope:
      # Feed the image embeddings to set the initial LSTM state.
      zero_state = lstm_cell.zero_state(
          batch_size=self.image_embeddings.get_shape()[0], dtype=tf.float32)
      _, initial_state = lstm_cell(self.image_embeddings, zero_state)

      # Allow the LSTM variables to be reused.
      lstm_scope.reuse_variables()

      if self.mode == "inference":
        # In inference mode, use concatenated states for convenient feeding and
        # fetching.
        tf.concat(axis=1, values=initial_state, name="initial_state")

        # Placeholder for feeding a batch of concatenated states.
        state_feed = tf.placeholder(dtype=tf.float32,
                                    shape=[None, sum(lstm_cell.state_size)],
                                    name="state_feed")
        state_tuple = tf.split(value=state_feed, num_or_size_splits=2, axis=1)

        # Run a single LSTM step.
        lstm_outputs, state_tuple = lstm_cell(
            inputs=tf.squeeze(self.seq_embeddings, axis=[1]),
            state=state_tuple)

        # Concatentate the resulting state.
        tf.concat(axis=1, values=state_tuple, name="state")
      else:
        # Run the batch of sequence embeddings through the LSTM.
        sequence_length = tf.reduce_sum(self.input_mask, 1)
        lstm_outputs, _ = tf.nn.dynamic_rnn(cell=lstm_cell,
                                            inputs=self.seq_embeddings,
                                            sequence_length=sequence_length,
                                            initial_state=initial_state,
                                            dtype=tf.float32,
                                            scope=lstm_scope)

    # Stack batches vertically.
    lstm_outputs = tf.reshape(lstm_outputs, [-1, lstm_cell.output_size])

    with tf.variable_scope("logits") as logits_scope:
      logits = tf.contrib.layers.fully_connected(
          inputs=lstm_outputs,
          num_outputs=self.config.vocab_size,
          activation_fn=None,
          weights_initializer=self.initializer,
          scope=logits_scope)

    if self.mode == "inference":
      tf.nn.softmax(logits, name="softmax")
    else:
      targets = tf.reshape(self.target_seqs, [-1])
      weights = tf.to_float(tf.reshape(self.input_mask, [-1]))

      # Compute losses.
      losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets,
                                                              logits=logits)
      batch_loss = tf.div(tf.reduce_sum(tf.multiply(losses, weights)),
                          tf.reduce_sum(weights),
                          name="batch_loss")

      tf.losses.add_loss(batch_loss)
      total_loss = tf.losses.get_total_loss()

      # Add summaries.
      tf.summary.scalar("losses/batch_loss", batch_loss)
      tf.summary.scalar("losses/total_loss", total_loss)
      for var in tf.trainable_variables():
        tf.summary.histogram("parameters/" + var.op.name, var)

      self.total_loss = total_loss
      self.target_cross_entropy_losses = losses  # Used in evaluation.
      self.target_cross_entropy_loss_weights = weights  # Used in evaluation.


  def setup_global_step(self):
    """Sets up the global step Tensor."""
    global_step = tf.Variable(
        initial_value=0,
        name="global_step",
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """Creates all ops for training and evaluation."""
    self.build_inputs()
    self.build_image_embeddings()
    self.build_seq_embeddings()
    self.build_model()
    self.setup_global_step()
    print("Finish building model")
    if self.mode == "train":
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.total_loss)

  def run_batch(self, session, _images, _images_parts, _input_seqs, _target_seqs, _input_mask):
    #session.run(tf.initialize_all_variables())

    _images00 = []
    _images01 = []
    _images02 = []
    _images10 = []
    _images11 = []
    _images12 = []
    _images20 = []
    _images21 = []
    _images22 = []

    for i in range(self.config.batch_size):
        _images00.append(_images_parts[i][0])
        _images01.append(_images_parts[i][1])
        _images02.append(_images_parts[i][2])
        _images10.append(_images_parts[i][3])
        _images11.append(_images_parts[i][4])
        _images12.append(_images_parts[i][5])
        _images20.append(_images_parts[i][6])
        _images21.append(_images_parts[i][7])
        _images22.append(_images_parts[i][8])


    feed = {self.input_seqs: _input_seqs, self.images: _images, self.input_mask: _input_mask, self.target_seqs: _target_seqs,
            self.images00: _images00, self.images01: _images01, self.images02: _images02,
            self.images10: _images10, self.images11: _images11, self.images12: _images12,
            self.images20: _images20, self.images21: _images21, self.images22: _images22}
    loss, _ = session.run([self.total_loss, self.train_op], feed_dict=feed)
    return loss