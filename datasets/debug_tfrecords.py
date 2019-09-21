import tensorflow as tf
tf.enable_eager_execution()
tf_record = "./pfam-regions-d0-s0-train-1.tfrecords"
e = tf.python_io.tf_record_iterator(tf_record).__next__()
features = {
    'protein': tf.FixedLenFeature(shape=(), dtype=tf.string),
    'domains': tf.FixedLenFeature(shape=(), dtype=tf.string)
}
parsed = tf.parse_single_example(
    serialized=e,
    # A scalar (0-D Tensor) of type string, a single binary
    # serialized `Example` proto.
    features=features,
    # A `dict` mapping feature keys to `FixedLenFeature` or
    # `VarLenFeature` values.
    example_name=None,
    #  A scalar string Tensor, the associated name (optional).
    name=None
    # A name for this operation (optional).
)
parsed['protein'] = tf.decode_raw(
    parsed['protein'], out_type=tf.uint8, little_endian=True, name=None
)
# tf.Tensor: shape=(sequence_length,), dtype=uint8
parsed['protein'] = tf.cast(x=parsed['protein'], dtype=tf.int32, name=None)
# embedding_lookup expects int32 or int64
# tf.Tensor: shape=(sequence_length,), dtype=int32
parsed['lengths'] = tf.shape(
    input=parsed['protein'], name=None, out_type=tf.int32
)[0]
domains = tf.decode_raw(parsed['domains'], out_type=tf.uint16)
# tf.Tensor: shape=(sequence_length,), dtype=uint16
domains = tf.cast(domains, tf.int32)
# sparse_softmax_cross_entropy_with_logits expects int32 or int64
# tf.Tensor: shape=(sequence_length,), dtype=int32
del parsed['domains']
parsed, domains
