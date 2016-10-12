import leveldb

class DataLoader(object):

    @staticmethod
    def load_data(path):
        db = leveldb.LevelDB(path)