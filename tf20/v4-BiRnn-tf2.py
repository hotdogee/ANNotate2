r"""Entry point for trianing a RNN-based classifier for the pfam data.

This is an updated version for TensorFlow 2.19 that maintains compatibility
with the parameters used in the original TensorFlow 1.12 implementation.

python v4-BiRnn-tf2.py \
  --training_data train_data \
  --eval_data eval_data \
  --model_dir ./checkpoints/ \
  --rnn_cell_type gru
"""

import coloredlogs
from keras import layers, models, optimizers, callbacks, metrics, regularizers
import tensorflow as tf
import os
import sys
import ast
import glob
import gzip
import json
import msgpack
import logging
import argparse
import datetime
import functools
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configure colored output
import colorama
from colorama import Fore, Back, Style
colorama.init()

# Import TensorFlow

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")
if not tf.__version__.startswith('2.'):
  raise ImportError("This script requires TensorFlow 2.x")

# Amino acid vocabulary - matching the original implementation
aa_list = list(' FLIMVPAWGSTYQNCO*UHKRDEBZX-')

# Configure logger
coloredlogs.install(level='DEBUG')
logger = tf.get_logger()

# For reproducibility
tf.random.set_seed(42)

# Custom TqdmCallback for progress display during training


class TqdmCallback(callbacks.Callback):
  def __init__(self, epochs, verbose=1):
    super(TqdmCallback, self).__init__()
    self.epochs = epochs
    self.verbose = verbose
    self.tqdm = None

  def on_train_begin(self, logs=None):
    self.tqdm = tqdm(total=self.epochs, desc="Training")

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    log_str = " - ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
    if self.verbose:
      self.tqdm.set_postfix_str(log_str)
    self.tqdm.update(1)

  def on_train_end(self, logs=None):
    self.tqdm.close()

# Parse sequence examples from TFRecord files (matching the original implementation)


def parse_sequence_example(serialized, mode):
  """Parse a tf.SequenceExample from the TFRecord file."""
  # Match the original implementation exactly
  context_features = {
      # Empty in the original
  }
  if mode != 'predict':
    sequence_features = {
        'protein': tf.io.FixedLenSequenceFeature([], dtype=tf.string),
        'domains': tf.io.FixedLenSequenceFeature([], dtype=tf.string, allow_missing=True)
    }
  else:
    sequence_features = {
        'protein': tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    }

  context, sequence = tf.io.parse_single_sequence_example(
      serialized=serialized,
      context_features=context_features,
      sequence_features=sequence_features
  )

  # Decode raw bytes to integers, matching the original implementation
  protein = tf.io.decode_raw(
      bytes=sequence['protein'],
      out_type=tf.uint8
  )
  protein = tf.cast(protein, tf.int32)
  protein = tf.squeeze(protein, axis=[])

  # Calculate length from protein shape, as in the original
  lengths = tf.shape(protein, out_type=tf.int32)[0]

  # For training and evaluation, decode domains
  if mode != 'predict':
    domains = tf.io.decode_raw(sequence['domains'], out_type=tf.uint16)
    domains = tf.cast(domains, tf.int32)
    domains = tf.squeeze(domains, axis=[])

    features = {
        'protein': protein,
        'lengths': lengths
    }

    return features, domains
  else:
    features = {
        'protein': protein,
        'lengths': lengths
    }

    return features, None

# Create a BiRNN model with Keras


def create_model(params):
  """Create a bidirectional RNN model using Keras."""
  # Input layers
  protein_input = layers.Input(shape=(None,), dtype=tf.int32, name='protein')
  lengths_input = layers.Input(shape=(), dtype=tf.int32, name='lengths')

  # Mask for padding - Fix: use Lambda layer to wrap tf operations
  mask = layers.Lambda(lambda x: tf.cast(
    tf.sign(x), dtype=tf.float32))(protein_input)

  # Embedding layer
  embedding = layers.Embedding(
      input_dim=params.vocab_size,
      output_dim=params.embed_dim,
      embeddings_initializer=tf.keras.initializers.RandomUniform(
          minval=-0.5,
          maxval=0.5
      ),
      mask_zero=True,
      name='embedding'
  )(protein_input)

  # Apply dropout to embedding
  if params.embedded_dropout > 0:
    embedding = layers.Dropout(
      rate=params.embedded_dropout)(embedding, training=True)

  # Convolutional layers
  conv_output = embedding

  # Add Conv1D bank (multiple kernel sizes) if specified
  if params.use_conv_1_bank:
    conv_bank_outputs = []
    bank_size = params.conv_1_bank_size

    for k in range(1, bank_size + 1):
      # Skip even kernel sizes if using only odd ones
      if params.use_conv_1_bank_odd and k % 2 == 0:
        continue

      conv = layers.Conv1D(
          filters=params.conv_1_filters,
          kernel_size=k,
          strides=params.conv_1_strides,
          padding='same',
          activation=None,
          use_bias=False,
          kernel_initializer='glorot_uniform',
          name=f'conv1d_bank_{k}'
      )(conv_output)

      # Batch normalization if specified
      if params.use_conv_batch_norm:
        conv = layers.BatchNormalization(
            axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=False
        )(conv)

      # Activation after batch norm
      conv = layers.Activation('relu')(conv)

      conv_bank_outputs.append(conv)

    # Concatenate all bank outputs
    conv_output = layers.Concatenate(axis=-1)(conv_bank_outputs)
  else:
    # Use single Conv1D layer
    conv_output = layers.Conv1D(
        filters=params.conv_1_filters,
        kernel_size=params.conv_1_kernel_size if hasattr(
          params, 'conv_1_kernel_size') else 7,
        strides=params.conv_1_strides,
        padding='same',
        activation=None,
        use_bias=False,
        kernel_initializer='glorot_uniform',
        name='conv1d'
    )(conv_output)

    # Batch normalization if specified
    if params.use_conv_batch_norm:
      conv_output = layers.BatchNormalization(
          axis=-1,
          momentum=0.99,
          epsilon=0.001,
          center=True,
          scale=False
      )(conv_output)

    # Activation after batch norm
    conv_output = layers.Activation('relu')(conv_output)

  # Add residual connection if specified
  if params.use_conv_1_residual:
    # Make sure embedding and conv_output have same dimensions
    residual_input = layers.Conv1D(
        filters=conv_output.shape[-1],
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='residual_projection'
    )(embedding)

    conv_output = layers.Add()([residual_input, conv_output])

  # Add highway network if specified
  if params.use_conv_1_highway:
    for i in range(params.conv_1_highway_depth):
      highway_layer = layers.Dense(
          units=params.conv_1_highway_units,
          activation='sigmoid',
          bias_initializer=tf.keras.initializers.Constant(-1.0),
          name=f'highway_gate_{i}'
      )

      transform_gate = highway_layer(conv_output)

      highway_output = layers.Dense(
          units=params.conv_1_highway_units,
          activation='relu',
          name=f'highway_output_{i}'
      )(conv_output)

      # Project conv_output to match highway_output dimensions if needed
      if conv_output.shape[-1] != params.conv_1_highway_units:
        conv_output = layers.Dense(
            units=params.conv_1_highway_units,
            activation=None,
            name=f'highway_projection_{i}'
        )(conv_output)

      conv_output = layers.Lambda(
          lambda x: x[0] * x[1] + (1.0 - x[0]) * x[2]
      )([transform_gate, highway_output, conv_output])

  # Apply dropout to conv output if specified
  if params.conv_1_dropout > 0:
    conv_output = layers.Dropout(rate=params.conv_1_dropout)(conv_output)

  # RNN layers
  rnn_output = conv_output

  # Create stacked bidirectional RNN layers
  for i, units in enumerate(params.rnn_num_units):
    # Choose appropriate RNN cell type
    if params.rnn_cell_type.lower() == 'cudnngru' or params.rnn_cell_type.lower() == 'gru':
      rnn_layer = layers.Bidirectional(
          layers.GRU(
              units=units,
              return_sequences=True,
              dropout=params.rnn_dropout if i > 0 else 0.0,
              recurrent_dropout=0.0,
              implementation=2,  # More efficient implementation
              name=f'gru_{i}'
          ),
          name=f'bidirectional_gru_{i}'
      )(rnn_output)
    elif params.rnn_cell_type.lower() == 'cudnnlstm' or params.rnn_cell_type.lower() == 'lstm':
      rnn_layer = layers.Bidirectional(
          layers.LSTM(
              units=units,
              return_sequences=True,
              dropout=params.rnn_dropout if i > 0 else 0.0,
              recurrent_dropout=0.0,
              implementation=2,  # More efficient implementation
              name=f'lstm_{i}'
          ),
          name=f'bidirectional_lstm_{i}'
      )(rnn_output)
    else:
      # Default to GRU if unrecognized cell type
      rnn_layer = layers.Bidirectional(
          layers.GRU(
              units=units,
              return_sequences=True,
              dropout=params.rnn_dropout if i > 0 else 0.0,
              recurrent_dropout=0.0,
              implementation=2,
              name=f'gru_default_{i}'
          ),
          name=f'bidirectional_gru_default_{i}'
      )(rnn_output)

    # Apply batch normalization if specified
    if params.use_rnn_batch_norm:
      rnn_layer = layers.BatchNormalization(
          axis=-1,
          momentum=0.99,
          epsilon=0.001,
          center=True,
          scale=True
      )(rnn_layer)

    rnn_output = rnn_layer

  # Output layer
  logits = layers.Dense(
      units=params.num_classes,
      activation=None,
      name='logits'
  )(rnn_output)

  # Apply masking for padding - Fix: using Lambda layer
  logits = layers.Lambda(
    lambda x: x[0] * tf.expand_dims(x[1], -1))([logits, mask])

  # Create the model with the appropriate inputs
  model = models.Model(
      inputs={'protein': protein_input, 'lengths': lengths_input},
      outputs=logits,
      name='pfam_birnn'
  )

  return model

# Learning rate schedule function for warmup


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, learning_rate, warmup_steps, warmup_repeat_steps,
               warmup_start_lr, decay_steps=None, decay_rate=None, schedule='exp'):
    super(WarmupSchedule, self).__init__()
    self.learning_rate = learning_rate
    self.warmup_steps = warmup_steps
    self.warmup_repeat_steps = warmup_repeat_steps
    self.warmup_start_lr = warmup_start_lr
    self.decay_steps = decay_steps
    self.decay_rate = decay_rate
    self.schedule = schedule

  def __call__(self, step):
    step = tf.cast(step, tf.float32)

    if self.schedule == 'exp':
      # Exponential warmup
      if self.warmup_repeat_steps > 0:
        # Calculate the effective step for warmup
        effective_step = tf.math.floormod(step, self.warmup_repeat_steps)
        # Check if we need to do warmup
        do_warmup = tf.less(effective_step, self.warmup_steps)

        def warmup():
          # Exponential warmup
          warmup_factor = effective_step / self.warmup_steps
          warmup_lr = self.warmup_start_lr * tf.math.pow(
              self.learning_rate / self.warmup_start_lr,
              warmup_factor
          )
          return warmup_lr

        def no_warmup():
          # Apply decay if specified
          if self.decay_steps and self.decay_rate:
            return self.learning_rate * tf.math.pow(
                self.decay_rate,
                tf.math.floor(step / self.decay_steps)
            )
          return self.learning_rate

        lr = tf.cond(do_warmup, warmup, no_warmup)
        return lr
      elif self.warmup_steps > 1:
        # If warmup steps but no repeat
        do_warmup = tf.less(step, self.warmup_steps)

        def warmup():
          # Exponential warmup
          warmup_factor = step / self.warmup_steps
          warmup_lr = self.warmup_start_lr * tf.math.pow(
              self.learning_rate / self.warmup_start_lr,
              warmup_factor
          )
          return warmup_lr

        def no_warmup():
          # Apply decay if specified
          if self.decay_steps and self.decay_rate:
            return self.learning_rate * tf.math.pow(
                self.decay_rate,
                tf.math.floor(step / self.decay_steps)
            )
          return self.learning_rate

        lr = tf.cond(do_warmup, warmup, no_warmup)
        return lr
      else:
        # No warmup, just potential decay
        if self.decay_steps and self.decay_rate:
          return self.learning_rate * tf.math.pow(
              self.decay_rate,
              tf.math.floor(step / self.decay_steps)
          )
        return self.learning_rate
    elif self.schedule == 'linear':
      # Linear warmup
      if self.warmup_repeat_steps > 0:
        # Calculate the effective step for warmup
        effective_step = tf.math.floormod(step, self.warmup_repeat_steps)
        # Check if we need to do warmup
        do_warmup = tf.less(effective_step, self.warmup_steps)

        def warmup():
          # Linear warmup
          warmup_factor = effective_step / self.warmup_steps
          warmup_lr = self.warmup_start_lr + \
              (self.learning_rate - self.warmup_start_lr) * warmup_factor
          return warmup_lr

        def no_warmup():
          # Apply decay if specified
          if self.decay_steps and self.decay_rate:
            return self.learning_rate * tf.math.pow(
                self.decay_rate,
                tf.math.floor(step / self.decay_steps)
            )
          return self.learning_rate

        lr = tf.cond(do_warmup, warmup, no_warmup)
        return lr
      elif self.warmup_steps > 1:
        # If warmup steps but no repeat
        do_warmup = tf.less(step, self.warmup_steps)

        def warmup():
          # Linear warmup
          warmup_factor = step / self.warmup_steps
          warmup_lr = self.warmup_start_lr + \
              (self.learning_rate - self.warmup_start_lr) * warmup_factor
          return warmup_lr

        def no_warmup():
          # Apply decay if specified
          if self.decay_steps and self.decay_rate:
            return self.learning_rate * tf.math.pow(
                self.decay_rate,
                tf.math.floor(step / self.decay_steps)
            )
          return self.learning_rate

        lr = tf.cond(do_warmup, warmup, no_warmup)
        return lr
      else:
        # No warmup, just potential decay
        if self.decay_steps and self.decay_rate:
          return self.learning_rate * tf.math.pow(
              self.decay_rate,
              tf.math.floor(step / self.decay_steps)
          )
        return self.learning_rate
    else:
      # No warmup, just potential decay
      if self.decay_steps and self.decay_rate:
        return self.learning_rate * tf.math.pow(
            self.decay_rate,
            tf.math.floor(step / self.decay_steps)
        )
      return self.learning_rate

# Set up custom loss function with masking for padding


def masked_sparse_categorical_crossentropy(y_true, y_pred):
  # Create a mask from the labels to ignore padded positions
  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

  # Apply sparse categorical crossentropy
  loss = tf.keras.losses.sparse_categorical_crossentropy(
      y_true=y_true,
      y_pred=y_pred,
      from_logits=True
  )

  # Apply the mask
  masked_loss = loss * mask

  # Compute mean loss over non-padding positions
  total_loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

  return total_loss

# Custom accuracy metric that ignores padding


class MaskedAccuracy(tf.keras.metrics.Metric):
  def __init__(self, name='masked_accuracy', **kwargs):
    super(MaskedAccuracy, self).__init__(name=name, **kwargs)
    self.correct_count = self.add_weight(
      name='correct_count', initializer='zeros')
    self.total_count = self.add_weight(name='total_count', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    # Create mask for non-padding positions
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)

    # Get predictions
    predictions = tf.argmax(y_pred, axis=-1)

    # Check for correct predictions
    correct = tf.cast(tf.equal(predictions, tf.cast(
      y_true, tf.int64)), tf.float32) * mask

    # Update counts
    self.correct_count.assign_add(tf.reduce_sum(correct))
    self.total_count.assign_add(tf.reduce_sum(mask))

  def result(self):
    return self.correct_count / self.total_count

  def reset_state(self):
    self.correct_count.assign(0)
    self.total_count.assign(0)

# Bucket by sequence length function similar to the original


def bucket_by_sequence_length(dataset, batch_size, bucket_boundaries, bucket_batch_sizes=None, drop_remainder=False):
  """Groups elements in dataset by sequence length into buckets.

  Args:
      dataset: A tf.data.Dataset containing elements with 'lengths' field
      batch_size: Default batch size if bucket_batch_sizes is None
      bucket_boundaries: List of sequence lengths to use as boundaries
      bucket_batch_sizes: Optional list of batch sizes for each bucket
      drop_remainder: Whether to drop remainder elements in each bucket

  Returns:
      A dataset grouped into batches by sequence length
  """
  if bucket_batch_sizes is None:
    # If not specified, use the same batch size for all buckets
    bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

  def element_to_bucket_id(features, labels):
    """Return the bucket id for this element based on sequence length."""
    seq_length = features['lengths']

    # Find the appropriate bucket for this length
    bucket_id = tf.size(bucket_boundaries) - tf.reduce_sum(
        tf.cast(tf.greater_equal(bucket_boundaries, seq_length), tf.int32))

    return bucket_id

  def window_size_fn(bucket_id):
    """Return the batch size for this bucket id."""
    return bucket_batch_sizes[bucket_id]

  # Group elements into buckets and then into batches
  return dataset.apply(
      tf.data.experimental.group_by_window(
          key_func=element_to_bucket_id,
          reduce_func=lambda bucket_id, ds: ds.padded_batch(
              batch_size=window_size_fn(bucket_id),
              padded_shapes=({
                  'protein': tf.TensorShape([None]),
                  'lengths': tf.TensorShape([])
              }, tf.TensorShape([None])),
              padding_values=({
                  'protein': tf.constant(0, dtype=tf.int32),
                  'lengths': tf.constant(0, dtype=tf.int32)
              }, tf.constant(0, dtype=tf.int32)),
              drop_remainder=drop_remainder
          ),
          window_size_func=window_size_fn
      )
  )


def pad_to_multiples(features, labels, pad_to_mutiples_of=8, padding_values=0):
  """Nvidia Volta Tensor Cores are enabled when data shape is multiples of 8
  """
  max_len = tf.shape(labels)[1]
  target_len = tf.cast(tf.multiply(tf.ceil(tf.truediv(
    max_len, pad_to_mutiples_of)), pad_to_mutiples_of), tf.int32)
  paddings = [[0, 0], [0, target_len - max_len]]
  features['protein'] = tf.pad(
    tensor=features['protein'], paddings=paddings, constant_values=padding_values)
  return features, tf.pad(tensor=labels, paddings=paddings, constant_values=padding_values)

# Input function to read TFRecord data


def input_fn(mode, params):
  """Input function for training and evaluation."""
  is_training = (mode == 'train')

  # Set up file pattern based on mode
  if is_training:
    file_pattern = params.training_data
  else:
    file_pattern = params.eval_data

  # List files first, matching original
  dataset = tf.data.Dataset.list_files(
      file_pattern=file_pattern,
      shuffle=is_training
  )

  # Interleave files
  dataset = dataset.interleave(
      lambda filename: tf.data.TFRecordDataset(
          filenames=filename,
          compression_type=None,
          buffer_size=params.dataset_buffer * 1024 * 1024
      ),
      cycle_length=params.dataset_parallel_reads,
      num_parallel_calls=tf.data.AUTOTUNE
  )

  # Shuffle if training
  if is_training:
    dataset = dataset.shuffle(buffer_size=params.shuffle_buffer)
    dataset = dataset.repeat(params.repeat_count)

  # Parse examples
  dataset = dataset.map(
      lambda x: parse_sequence_example(x, 'train' if is_training else 'eval'),
      num_parallel_calls=tf.data.AUTOTUNE
  )

  # Our inputs are variable length, so bucket, dynamic batch and pad them.
  if mode != 'predict':
    padded_shapes = ({'protein': [None], 'lengths': []}, [None])
  else:
    padded_shapes = {'protein': [None], 'lengths': []}

  # Apply bucketing
  dataset = dataset.bucket_by_sequence_length(
      element_length_func=lambda seq, dom: seq['lengths'],
      bucket_boundaries=[2 ** x for x in range(5, 15)],  # 32 ~ 16384
      bucket_batch_sizes=[params.batch_size * 2 **
                          x for x in range(10, -1, -1)],  # 1024 ~ 1
      padded_shapes=padded_shapes,
      padding_values=None,  # Defaults to padding with 0.
      pad_to_bucket_boundary=False
  ).map(
      functools.partial(
        pad_to_multiples, pad_to_mutiples_of=8, padding_values=0),
      num_parallel_calls=int(params.num_cpu_threads / 2)
  )

  # Prefetch for better performance
  dataset = dataset.prefetch(buffer_size=params.prefetch_buffer)

  return dataset


def main():
  # Parse command line arguments
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.register('type', 'list', lambda v: ast.literal_eval(v))

  # Basic parameters
  parser.add_argument(
      '--model_dir',
      type=str,
      default='./checkpoints/Exp_tensorflow',
      help='Path for saving model checkpoints during training')
  parser.add_argument(
      '--experiment_name',
      type=str,
      default='Exp',
      help='Experiment name for logging purposes')
  parser.add_argument(
      '--host_script_name',
      type=str,
      default='tensorflow',
      help='Host script name for logging purposes')
  parser.add_argument(
      '--training_data',
      type=str,
      default='dataset/pfam-regions-d0-s0-train.tfrecords',
      help='Path to training data (tf.Example in TFRecord format)')
  parser.add_argument(
      '--eval_data',
      type=str,
      default='dataset/pfam-regions-d0-s0-train.tfrecords',
      help='Path to evaluation data (tf.Example in TFRecord format)')
  parser.add_argument(
      '--metadata_path',
      type=str,
      default='dataset/pfam-regions-d0-s0-meta.json',
      help='Path to metadata.json generated by prep_dataset')
  parser.add_argument(
      '--num_classes',
      type=int,
      default=16714,  # From 8086K2-2.sh
      help='Number of domain classes.')

  # Dataset parameters
  parser.add_argument(
      '--dataset_buffer',
      type=int,
      default=256,
      help='Number of MB in the read buffer.')
  parser.add_argument(
      '--dataset_parallel_reads',
      type=int,
      default=1,
      help='Number of files to read in parallel.')
  parser.add_argument(
      '--shuffle_buffer',
      type=int,
      default=16 * 1024,
      help='Maximum number elements that will be buffered when shuffling input.')
  parser.add_argument(
      '--repeat_count',
      type=int,
      default=1,
      help='Number of times the dataset should be repeated.')
  parser.add_argument(
      '--batch_size',
      type=int,
      default=1,  # From 8086K2-2.sh
      help='Batch size to use for training/evaluation.')
  parser.add_argument(
      '--prefetch_buffer',
      type=int,
      default=64,
      help='Maximum number of batches that will be buffered when prefetching.')

  # Model architecture parameters
  parser.add_argument(
      '--vocab_size',
      type=int,
      default=len(aa_list),
      help='Vocabulary size.')
  parser.add_argument(
      '--embed_dim',
      type=int,
      default=32,
      help='Embedding dimensions.')
  parser.add_argument(
      '--embedded_dropout',
      type=float,
      default=0.2,  # From 8086K2-2.sh
      help='Dropout rate used for embedding layer outputs.')

  # Convolution parameters
  parser.add_argument(
      '--use_conv_1_bank',
      type='bool',
      default='True',  # From 8086K2-2.sh
      help='Use convolution bank.')
  parser.add_argument(
      '--use_conv_1_bank_odd',
      type='bool',
      default='True',  # From 8086K2-2.sh
      help='Use convolution bank with only odd kernel sizes.')
  parser.add_argument(
      '--conv_1_bank_size',
      type=int,
      default=15,  # From 8086K2-2.sh
      help='Convolution bank kernal sizes 1 to bank_size.')
  parser.add_argument(
      '--conv_1_filters',
      type=int,
      default=32,  # From 8086K2-2.sh
      help='Number of convolution filters.')
  parser.add_argument(
      '--conv_1_kernel_size',
      type=int,
      default=7,
      help='Length of the convolution filters.')
  parser.add_argument(
      '--conv_1_strides',
      type=int,
      default=1,  # From 8086K2-2.sh
      help='The number of entries by which the filter is moved right at each step.')
  parser.add_argument(
      '--conv_1_dropout',
      type=float,
      default=0.2,  # From 8086K2-2.sh
      help='Dropout rate used for convolution layer outputs.')
  parser.add_argument(
      '--use_conv_batch_norm',
      type='bool',
      default='True',  # From 8086K2-2.sh
      help='Apply batch normalization after convolution layers.')
  parser.add_argument(
      '--use_conv_1_residual',
      type='bool',
      default='True',  # From 8086K2-2.sh
      help='Add residual connection after convolution layer 1.')
  parser.add_argument(
      '--use_conv_1_highway',
      type='bool',
      default='True',  # From 8086K2-2.sh
      help='Add a highway network after convolution layer 1.')
  parser.add_argument(
      '--conv_1_highway_depth',
      type=int,
      default=3,  # From 8086K2-2.sh
      help='Number of layers of highway network.')
  parser.add_argument(
      '--conv_1_highway_units',
      type=int,
      default=512,  # From 8086K2-2.sh
      help='Number of units per layer of highway network.')
  parser.add_argument(
      '--use_conv_1_prenet',
      type='bool',
      default='False',  # From 8086K2-2.sh
      help='Add prenet before convolution layer 1.')

  # RNN parameters
  parser.add_argument(
      '--rnn_cell_type',
      type=str,
      choices=['lstm', 'gru', 'cudnnlstm', 'cudnngru'],
      default='gru',  # Match CudnnGRU from 8086K2-2.sh
      help='RNN Cell Type')
  parser.add_argument(
      '--rnn_num_units',
      type='list',
      default='[512,512,512,512]',  # From 8086K2-2.sh
      help='Number of node per recurrent network layer.')
  parser.add_argument(
      '--rnn_dropout',
      type=float,
      default=0.0,  # From 8086K2-2.sh
      help='Dropout rate used between rnn layers.')
  parser.add_argument(
      '--use_rnn_batch_norm',
      type='bool',
      default='True',  # From 8086K2-2.sh
      help='Apply batch normalization after recurrent layers.')

  # Optimizer parameters
  parser.add_argument(
      '--optimizer',
      type=str,
      choices=['adam', 'sgd', 'rmsprop', 'momentum'],
      default='momentum',  # From 8086K2-2.sh
      help='Optimizer to use for training.')
  parser.add_argument(
      '--learning_rate_decay_fn',
      type=str,
      default='exponential_decay',  # From 8086K2-2.sh
      help='Learning rate decay function. One of "none", "exponential_decay"')
  parser.add_argument(
      '--learning_rate_decay_steps',
      type=int,
      default=350000,  # From 8086K2-2.sh
      help='Decay learning_rate by decay_rate every decay_steps.')
  parser.add_argument(
      '--learning_rate_decay_rate',
      type=float,
      default=0.7,  # From 8086K2-2.sh
      help='Learning rate decay rate.')
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.065,  # From 8086K2-2.sh
      help='Learning rate used for training.')
  parser.add_argument(
      '--warmup_steps',
      type=int,
      default=35000,  # From 8086K2-2.sh
      help='Learning rate warmup steps needed to reach specified learning_rate.')
  parser.add_argument(
      '--warmup_repeat_steps',
      type=int,
      default=0,  # From 8086K2-2.sh
      help='Restart warmup every this many steps.')
  parser.add_argument(
      '--warmup_start_lr',
      type=float,
      default=0.001,  # From 8086K2-2.sh
      help='Learning rate warmup starting multiplier value.')
  parser.add_argument(
      '--warmup_schedule',
      type=str,
      default='exp',  # From 8086K2-2.sh
      help='Learning rate warmup schedule. One of "exp", "linear", "none"')

  # Training parameters
  parser.add_argument(
      '--epochs',
      type=int,
      default=100,
      help='Number of epochs to train')
  parser.add_argument(
      '--save_model_freq',
      type=int,
      default=1,
      help='Save model every this many epochs')
  parser.add_argument(
      '--num_cpu_threads',
      type=int,
      default=0,
      help='Number of CPU threads to use, defaults to half the number of hardware threads.')
  parser.add_argument(
      '--job',
      type=str,
      choices=['train', 'eval', 'predict', 'dataprep', 'export', 'plot'],
      default='train',
      help='Set job type to run')

  # Parse flags
  FLAGS, extra_args = parser.parse_known_args()
  FLAGS.num_cpu_threads = FLAGS.num_cpu_threads or os.cpu_count()

  # Set up loggers
  coloredlogs.DEFAULT_FIELD_STYLES = dict(
      asctime=dict(color='green'),
      hostname=dict(color='magenta', bold=True),
      levelname=dict(color='black', bold=True),
      programname=dict(color='cyan', bold=True),
      name=dict(color='blue'))
  coloredlogs.DEFAULT_LEVEL_STYLES = dict(
      spam=dict(color='green', faint=True),
      debug=dict(color='green'),
      verbose=dict(color='blue'),
      info=dict(),
      notice=dict(color='magenta'),
      warning=dict(color='yellow'),
      success=dict(color='green', bold=True),
      error=dict(color='red'),
      critical=dict(color='red', bold=True))
  coloredlogs.DEFAULT_LOG_FORMAT = '%(asctime)s \x1b[1;35m{}\x1b[0m \x1b[34m{}[%(process)d]\x1b[0m %(levelname)s %(message)s'.format(
      FLAGS.experiment_name, FLAGS.host_script_name
  )

  # Load metadata if specified
  metadata = None
  if FLAGS.metadata_path:
    try:
      with open(FLAGS.metadata_path, 'r') as f:
        metadata = json.load(f)
      print(f"Loaded metadata from {FLAGS.metadata_path}")
    except Exception as e:
      print(f"Error loading metadata: {e}")

  # Create model directory if it doesn't exist
  os.makedirs(FLAGS.model_dir, exist_ok=True)

  # Set up learning rate schedule
  decay_steps = FLAGS.learning_rate_decay_steps if FLAGS.learning_rate_decay_fn == 'exponential_decay' else None
  decay_rate = FLAGS.learning_rate_decay_rate if FLAGS.learning_rate_decay_fn == 'exponential_decay' else None

  lr_schedule = WarmupSchedule(
      learning_rate=FLAGS.learning_rate,
      warmup_steps=FLAGS.warmup_steps,
      warmup_repeat_steps=FLAGS.warmup_repeat_steps,
      warmup_start_lr=FLAGS.warmup_start_lr,
      decay_steps=decay_steps,
      decay_rate=decay_rate,
      schedule=FLAGS.warmup_schedule
  )

  # Create the optimizer
  if FLAGS.optimizer.lower() == 'adam':
    optimizer = tf.keras.optimizers.Adam(
      learning_rate=lr_schedule, epsilon=0.05)
  elif FLAGS.optimizer.lower() == 'sgd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
  elif FLAGS.optimizer.lower() == 'rmsprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
  elif FLAGS.optimizer.lower() == 'momentum':
    optimizer = tf.keras.optimizers.SGD(
      learning_rate=lr_schedule, momentum=0.9)
  else:
    # Default to Adam
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

  # Create the model
  model = create_model(FLAGS)

  # Compile the model
  model.compile(
      optimizer=optimizer,
      loss=masked_sparse_categorical_crossentropy,
      metrics=[MaskedAccuracy()]
  )

  # Print model summary
  model.summary()

  # If job is 'plot', create a visualization of the model and exit
  if FLAGS.job == 'plot':
    try:
      # Make sure the directory exists
      os.makedirs(FLAGS.model_dir, exist_ok=True)

      # Import visualization utilities
      import keras

      # Define the output file path
      plot_file = os.path.join(
        FLAGS.model_dir, f"{FLAGS.experiment_name}_model.png")

      # Create the plot with detailed layer information
      keras.utils.plot_model(
          model,
          to_file=plot_file,
          show_shapes=True,
          show_dtype=False,
          show_layer_names=True,
          rankdir='TB',  # Top to bottom layout
          expand_nested=True,
          dpi=96
      )

      print(f"Model architecture plotted to {plot_file}")
      # Exit without training
      return
    except Exception as e:
      print(f"Error plotting model: {e}")
      print("Please make sure pydot and graphviz are installed.")
      print("  pip install pydot")
      print("  apt-get install graphviz")
      return

  # Set up model callbacks
  callbacks_list = [
      callbacks.ModelCheckpoint(
          filepath=os.path.join(FLAGS.model_dir, 'model-{epoch:04d}.h5'),
          save_best_only=False,
          save_weights_only=False,
          save_freq='epoch',
          period=FLAGS.save_model_freq
      ),
      callbacks.TensorBoard(
          log_dir=os.path.join(FLAGS.model_dir, 'logs'),
          write_graph=True,
          update_freq='epoch'
      ),
      TqdmCallback(epochs=FLAGS.epochs)
  ]

  # Create input dataset for training
  train_dataset = input_fn('train', FLAGS)

  # Train the model
  # Dataset is already properly formatted with batching and bucketing
  history = model.fit(
      train_dataset,
      epochs=FLAGS.epochs,
      callbacks=callbacks_list
  )

  # Save the final model
  model.save(os.path.join(FLAGS.model_dir, 'final_model'))

  print(f"Training completed. Model saved to {FLAGS.model_dir}")


if __name__ == '__main__':
  main()
