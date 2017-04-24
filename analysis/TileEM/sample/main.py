import tempfile
import itertools as IT
import os
import json


def uniquify(path, sep = '_'):
    '''
	Create an new directory output_*/ if one already exist
	'''
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence 
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        filename = tempfile.mkdtemp(dir = dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    return filename
# if __main__():
DATA_DIR = uniquify('output')
print "Creating Directory:",DATA_DIR
os.chdir(DATA_DIR)

