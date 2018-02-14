import optparse
import pickle
import os
import sys


def standard_parser():
    parser = optparse.OptionParser()

    # Option parser
    parser.add_option('-c', '--comment',
                      action="store", dest="comment",
                      help="Comment about the model", type="string", default="")

    parser.add_option('-m', '--config',
                      action="store", dest="config",
                      help="Path to a config file", type="string",
                      default="/Users/kforest/workspace/toxiccomment/models/keras_config.json")

    options, args = parser.parse_args()

    return options, args


def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1

    input_size = os.path.getsize(filepath)
    bytes_in = bytearray(0)
    with open(filepath, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    obj = pickle.loads(bytes_in)

    return obj