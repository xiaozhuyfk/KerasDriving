#import caffe
import leveldb
import numpy as np
#from caffe.proto import caffe_pb2
import data_pb2
import random
#from matplotlib import pyplot as plt

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
    def get_db(path):
        db = leveldb.LevelDB(path)
        return db

    @staticmethod
    def get_data(db, key):
        value = db.Get(key)
        datum = data_pb2.Datum()
        datum.ParseFromString(value)
        data = datum_to_array(datum)

        affordance = np.array(datum.float_data)
        image = np.transpose(data, (1,2,0))

        return affordance, image


    @staticmethod
    def load_data(path):
        db = leveldb.LevelDB(path)
        #datum = caffe_pb2.Datum()
        datum = data_pb2.Datum()

        X = []
        Y = []
        for key, value in db.RangeIter():
            print key
            datum.ParseFromString(value)
            data = datum_to_array(datum)
            affordance = np.array(datum.float_data)[:-1]
            #print affordance
            indicator = np.array(datum.float_data)[-1]

            #CxHxW to HxWxC in cv2
            #print affordance
            #print indicator
            #image = np.transpose(data, (1,2,0))
            #plt.imshow(image)
            #plt.show()

            #X.append(image)
            #Y.append(affordance)

        return np.array(X), np.array(Y)

    @staticmethod
    def generate_data(path):
        db = leveldb.LevelDB(path)
        #datum = caffe_pb2.Datum()
        datum = data_pb2.Datum()
        weights = [1,
                   7, 3.5, 7, 75, 75,
                   9.5, 5.5, 5.5, 9.5, 75, 75, 75,
                   1]
        weights = np.array(weights, dtype=np.float32)

        while True:
            X = []
            Y = []

            """
            indices = random.sample(xrange(484815), 64)
            for idx in indices:
                key = str(idx)
                key = "0" * (8 - len(key)) + key
                value = db.Get(key)
                datum.ParseFromString(value)
                data = datum_to_array(datum)
                affordance = np.array(datum.float_data)
                image = np.transpose(data, (1,2,0))

                for i in xrange(len(affordance)):
                    affordance[i] = 0.1 + 0.8 * (affordance[i] / float(weights[i]))

                X.append(image)
                Y.append(affordance)
            yield (np.array(X), np.array(Y))
            """

            for key, value in db.RangeIter():
                datum.ParseFromString(value)
                data = datum_to_array(datum)
                affordance = np.array(datum.float_data) / weights
                image = np.transpose(data, (1,2,0))
                #for i in xrange(len(affordance)):
                    #affordance[i] = 0.1 + 0.8 * (affordance[i] / float(weights[i]))

                X.append(image)
                Y.append(affordance)

                if len(X) >= 32:
                    yield (np.array(X), np.array(Y))
                    X = []
                    Y = []
            if len(X) > 0:
                yield (np.array(X), np.array(Y))
