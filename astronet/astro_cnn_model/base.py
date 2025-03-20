"""Base functions for AstroCNN models."""

import tensorflow as tf


def create_conv_block(name, block_params):
  """Creates a convolutional block."""
  layers = []
  for i in range(block_params.cnn_num_blocks):
    block_name = f'{name}_block_{i+1}'
    num_filters = int(
        float(block_params.cnn_initial_num_filters) *
        block_params.cnn_block_filter_factor**i)
    for j in range(block_params.cnn_block_size):
      Conv1D = (
          tf.keras.layers.SeparableConv1D
          if block_params.get('separable') else tf.keras.layers.Conv1D)
      layers.append(
          Conv1D(
              filters=num_filters,
              kernel_size=block_params.cnn_kernel_size,
              padding=block_params.convolution_padding,
              activation='relu',
              name=f'{block_name}_conv_{j+1}'))
    if block_params.pool_size:
      layers.append(
          tf.keras.layers.MaxPool1D(
              pool_size=block_params.pool_size,
              strides=block_params.pool_strides,
              name=f'{block_name}_pool'))
  layers.append(tf.keras.layers.Flatten())
  return layers


def create_ts_blocks(hparams):
  blocks = {}
  for name, block_params in hparams.time_series_hidden.items():
    blocks[name] = create_conv_block(name, block_params)
  return blocks


def apply_block(block, input, training):
  y = input
  for layer in block:
    y = layer(y, training=training)
  return y
