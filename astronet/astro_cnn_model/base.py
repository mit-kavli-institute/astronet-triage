"""Base layers for AstroCNN models."""

import tensorflow as tf


class Block(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.layers = []

  def call(self, x, training):
    y = x
    for layer in self.layers:
      y = layer(y, training=training)
    return y


class ConvBlock(Block):

  def __init__(self, name, block_params, **kwargs):
    super().__init__(**kwargs)
    self.key = name  # Can't use name as it's reserved by Keras
    self.block_params = block_params

    for i in range(block_params.cnn_num_blocks):
      block_name = f'{name}_block_{i+1}'
      num_filters = int(
          float(block_params.cnn_initial_num_filters) *
          block_params.cnn_block_filter_factor**i)
      for j in range(block_params.cnn_block_size):
        Conv1D = (
            tf.keras.layers.SeparableConv1D
            if block_params.get('separable') else tf.keras.layers.Conv1D)
        self.layers.append(
            Conv1D(
                filters=num_filters,
                kernel_size=block_params.cnn_kernel_size,
                padding=block_params.convolution_padding,
                activation='relu',
                name=f'{block_name}_conv_{j+1}'))
      if block_params.pool_size:
        self.layers.append(
            tf.keras.layers.MaxPool1D(
                pool_size=block_params.pool_size,
                strides=block_params.pool_strides,
                name=f'{block_name}_pool'))
    self.layers.append(tf.keras.layers.Flatten())


class DenseBlock(Block):

  def __init__(self,
               num_layers,
               layer_size,
               use_batch_norm=False,
               dropout_rate=None,
               activation='relu',
               **kwargs):
    super().__init__(**kwargs)
    self.num_layers = num_layers
    self.layer_size = layer_size
    self.activation = activation
    for _ in range(num_layers):
      self.layers.append(
          tf.keras.layers.Dense(units=layer_size, activation=activation))
      if use_batch_norm:
        self.layers.append(tf.keras.layers.BatchNormalization())
      self.layers.append(tf.keras.layers.Dropout(dropout_rate))


class OutputLayer(Block):

  def __init__(self, n_labels, exclusive_labels=None, **kwargs):
    super().__init__(**kwargs)
    self.n_labels = n_labels
    self.exclusive_labels = exclusive_labels
    activation = 'softmax' if (n_labels > 1 and exclusive_labels) else 'sigmoid'
    self.layers.append(tf.keras.layers.Dense(n_labels, activation=activation))
