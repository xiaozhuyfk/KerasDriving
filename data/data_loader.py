import caffe
import leveldb
import numpy as np
from caffe.proto import caffe_pb2

class DataLoader(object):

    @staticmethod
    def load_data(path):
        db = leveldb.LevelDB(path)
        datum = caffe_pb2.Datum()

        X = []
        Y = []
        for key, value in db.RangeIter():
            datum.ParseFromString(value)
            data = caffe.io.datum_to_array(datum)
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
        datum = caffe_pb2.Datum()

        while True:
            X = []
            Y = []
            for key, value in db.RangeIter():
                datum.ParseFromString(value)
                data = caffe.io.datum_to_array(datum)
                affordance = np.array(datum.float_data)[:-1]
                image = np.transpose(data, (1,2,0))
                X.append(image)
                Y.append(affordance)

                if len(X) >= 1000:
                    yield (np.array(X), np.array(Y))
                    X = []
                    Y = []
            if len(X) > 0:
                yield (np.array(X), np.array(Y))
