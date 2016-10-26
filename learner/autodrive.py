import logging
import globals
import numpy as np
from keras.callbacks import EarlyStopping
from data.data_loader import DataLoader
from model.cnn import CNN



logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def train(dataset):
    #X, Y = DataLoader.load_data(globals.config.get("Data", "training-data"))
    """
    db = DataLoader.get_db(globals.config.get("Data", "training-data"))

    X = []
    Y = []
    iterations = 10000

    aff, image = DataLoader.get_data(db, "00001439")
    """

    training_path = globals.config.get("Data", "training-data")
    shape = (210, 280, 3)

    model = CNN.model(64, 3, 3, shape)
    model.compile(optimizer='rmsprop', loss='mae', metrics=["accuracy"])
    earlyStopping = EarlyStopping(monitor='val_loss',
                                  patience=0,
                                  verbose=0,
                                  mode='auto')

    model.fit_generator(DataLoader.generate_data(training_path),
                        samples_per_epoch=8192,
                        nb_epoch=10,
                        callbacks=[earlyStopping])

    CNN.store_model(model)



def test(dataset):
    training_path = globals.config.get("Data", "training-data")
    db = DataLoader.get_db(globals.config.get("Data", "training-data"))
    aff, image = DataLoader.get_data(db, "00000001")

    X = np.array([np.array(image)])

    model = CNN.load_model()
    Y = model.predict(X)

    print Y
    print aff


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Choose to learn or test Keras AutoDrive')

    parser.add_argument('--config',
                        default='config.cfg',
                        help='The configuration file to use')
    subparsers = parser.add_subparsers(help='command help')
    train_parser = subparsers.add_parser('train', help='Train CCN to learn affordance')
    train_parser.add_argument('dataset',
                              help='The dataset to train.')
    train_parser.set_defaults(which='train')

    test_parser = subparsers.add_parser('test', help='Test affordance learning')
    test_parser.add_argument('dataset',
                             help='The dataset to test')
    test_parser.set_defaults(which='test')

    args = parser.parse_args()

    # Read global config
    globals.read_configuration(args.config)

    if args.which == 'train':
        train(args.dataset)
    elif args.which == 'test':
        test(args.dataset)


if __name__ == '__main__':
    main()