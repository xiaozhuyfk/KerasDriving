import logging
import globals
from data.data_loader import DataLoader
from model.cnn import CNN

logging.basicConfig(format='%(asctime)s : %(levelname)s '
                           ': %(module)s : %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)


def train(dataset):
    X, Y = DataLoader.load_data(globals.config.get("Data", "training-data"))
    training_path = globals.config.get("Data", "training-data")
    shape = (210, 280, 3)

    #model = CNN.model(64, 3, 3, shape)
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=["accuracy"])
    #model.fit_generator(DataLoader.generate_data(training_path), 100, 10)
    #CNN.store_model(model)


def test(dataset):
    pass


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