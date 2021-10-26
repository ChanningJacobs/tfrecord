import tensorflow as tf
import numpy as np
import pprint


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = [value.numpy()] # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def convert_to_feature(value):
    '''Converts non-None type value into feature. Returns the feature and the tf.dtype dtype.'''
    tensor = tf.convert_to_tensor(value) # string values have been converted to bytes
    dtype = tensor.dtype
    shape = tensor.shape
    feature = None
    if len(tensor.shape) > 1:
        serialized = tf.io.serialize_tensor(tensor)
        feature = _bytes_feature(serialized)
    else:
        if len(tensor.shape) == 0:
            tensor = tf.expand_dims(tensor, 0)
        tensor = tensor.numpy().tolist()
        if isinstance(tensor[0], bytes):
            # Note: Large videos/tensors can be converted into a list of encoded byte strings and saved out here
            feature = _bytes_feature(tensor)
        elif isinstance(tensor[0], float):
            feature = _float_feature(tensor)
        elif isinstance(tensor[0], (int, bool)):
            feature = _int64_feature(tensor)
        else:
            raise TypeError(f'Unexpected type {type(tensor[0])} from {tensor[0]}.')
    return feature, dtype, shape


# complete_dict -> create_schema -> feature_description
# incomplete_dict -> converter(decoupled from schema) -> tfrecord
# tfrecord + schema -> parse -> dataset


def serialize(unserialized_dict):
    feature = {}
    for key, val in unserialized_dict.items():
        if val is not None:
            val, _, _ = convert_to_feature(val)
            feature[key] = val
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()


def create_schema(complete_dict):
    feature_description = {}
    key_types = {}
    for key, val in complete_dict.items():
        feature, key_types[key], shape = convert_to_feature(val)
        feature = str(feature) # Look for a more robust option/method
        if 'int64_list' in feature:
            # feature_description[key] = tf.io.VarLenFeature(tf.int64)
            # feature_description[key] = tf.io.FixedLenSequenceFeature()
            feature_description[key] = tf.io.FixedLenFeature(shape, tf.int64, default_value=tf.zeros(shape, tf.int64))
        elif 'float_list' in feature:
            feature_description[key] = tf.io.FixedLenFeature(shape, tf.float32, default_value=tf.zeros(shape, tf.float32))
        elif 'bytes_list' in feature:
            if len(shape) == 0:
                feature_description[key] = tf.io.FixedLenFeature(shape, tf.string, default_value='')
            else:
                feature_description[key] = tf.io.FixedLenFeature([], tf.string, default_value=tf.io.serialize_tensor(tf.zeros(shape, key_types[key])))
        else:
            raise TypeError(f'Unexpected type {feature}.')
    return feature_description, key_types


def parse(schema, key_types, proto):
    example_dict = tf.io.parse_single_example(proto, schema)
    for key, val in example_dict.items():
        # if isinstance(val, tf.SparseTensor):
        #     example_dict[key] = tf.sparse.to_dense(val, default_value=0)
        if val.dtype == tf.string:
            try:
                example_dict[key] = tf.io.parse_tensor(val, key_types[key])
            except BaseException:   # This is too broad
                continue
    return example_dict


# All keys have a corresponding value datatype
complete_dict = {
    'key_1': 'String type',
    'key_2': 0,
    'key_3': 1.0,
    'key_4': [[1, 2, 3]],
    'key_5': np.ones((32, 32, 3)),
    'key_6': (1, 2, 3)
}

# Keys contain None values
incomplete_dict = {
    'key_1': None,
    'key_2': None,
    'key_3': None,
    'key_4': None,
    'key_5': None,
    'key_6': None
}


schema, key_types = create_schema(complete_dict)
print(schema)
print(key_types)
print(serialize(complete_dict))
print(serialize(incomplete_dict))

pprint.pprint(parse(schema, key_types, serialize(complete_dict)))
pprint.pprint(parse(schema, key_types, serialize(incomplete_dict)))
