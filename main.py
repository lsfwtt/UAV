from trainer import Trainer
from utils.fix_random_seed import random_seed

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset-dir', type=str, default='/home/lsfwtt/UAV/datasets/dataset_test_cropped')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--epoch-size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)
    parser.add_argument('--save-dir', type=str, default='/home/lsfwtt/UAV/src/weight')

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--step', type=int, default=128)
    parser.add_argument('--multi-gpus', type=bool, default=False)
    parser.add_argument('--if-checkpoint', type=bool, default=False)
    parser.add_argument('--checkpoint-path', type=str, default='')

    parser.add_argument('--seed', type=int, default = 42)

    return parser.parse_args()

def main():
    args = parse_args()
    random_seed(args.seed)

    trainer = Trainer(args)
    
    trainer.load_checkpoint()
    for epoch in range(trainer.start_epoch, args.epochs):
        trainer.train(epoch)
        trainer.validate(epoch)

if __name__ == '__main__':
    main()