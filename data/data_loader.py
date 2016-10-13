#import caffe
import leveldb
import numpy as np
#from caffe.proto import caffe_pb2
import data_pb2

def datum_to_array(datum):
    """Converts a datum to an array. Note that the label is not returned,
    as one can easily get it by calling datum.label.
    """
    if len(datum.data):
        return np.fromstring(datum.data, dtype=np.uint8).reshape(
            datum.channels, datum.height, datum.width)
    else:
        return np.array(datum.float_data).astype(float).reshape(
            datum.channels, datum.height, datum.width)

class DataLoader(object):

    @staticmethod
    def load_data(path):
        db = leveldb.LevelDB(path)
        #datum = caffe_pb2.Datum()
        datum = data_pb2.Datum()

        X = []
        Y = []
        for key, value in db.RangeIter():
            datum.ParseFromString(value)
            data = datum_to_array(datum)
            affordance = np.array(datum.float_data)[:-1]
            print affordance
            #CxHxW to HxWxC in cv2
            image = np.transpose(data, (1,2,0))
            print image.shape
            X.append(image)
            Y.append(affordance)
            break
        return np.array(X), np.array(Y)

    @staticmethod
    def generate_data(path):
        db = leveldb.LevelDB(path)
        #datum = caffe_pb2.Datum()
        datum = data_pb2.Datum()

        while True:
            X = []
            Y = []
            for key, value in db.RangeIter():
                datum.ParseFromString(value)
                data = datum_to_array(datum)
                affordance = np.array(datum.float_data)[:-1]
                image = np.transpose(data, (1,2,0))
                X.append(image)
                Y.append(affordance)

                if len(X) >= 100:
                    yield (np.array(X), np.array(Y))
                    X = []
                    Y = []
            if len(X) > 0:
                yield (np.array(X), np.array(Y))
