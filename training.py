# coding:utf-8
import os, sys, time, random
import numpy as np
import argparse, yaml, shutil
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions

import utils.ioFunctions as IO
from model import GraphCNN
from dataset import PointCloudDataset
from updater import GraphCnnUpdater
from evaluators import GraphCnnEvaluator

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--out', '-o', default= 'results/training_{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S')),
                        help='Directory to output the result')

    ##############################################################
    # training configs
    parser.add_argument('--batchsize', '-b', type=int, default=16,
                        help='Number of images in each mini-batch')
    parser.add_argument('--iteration', type=int, default=100000,
                        help='Number of iteration')
    parser.add_argument('--learning_rate', type=float, default=12e-4,
                        help='Number of iteration')

    parser.add_argument('--snapshot_interval', type=int, default=100)
    parser.add_argument('--display_interval', type=int, default=100)
    parser.add_argument('--evaluation_interval', type=int, default=100)

    ##############################################################
    # Graph configs
    parser.add_argument('--num_nearest_neighbor', type=int, default=40,
                        help='# nearest neighbor')
    parser.add_argument('--num_vertices', type=int, default=1024,
                        help='# vertices')


    parser.add_argument('--model', '-m', default='',
                        help='Load model data')
    parser.add_argument('--resume', '-res', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')
    parser.add_argument('--train_list', default='data/modelnet40_ply_hdf5_2048/train_files.txt',
                        help='training list')
    parser.add_argument('--val_list', default='data/modelnet40_ply_hdf5_2048/test_files.txt',
                        help='validation list')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    print('----- Build model -----')
    gcnn = GraphCNN()
    if args.model:
        chainer.serializers.load_npz(args.model, gcnn)

    if args.gpu >= 0:
        chainer.backends.cuda.set_max_workspace_size(2 * 512 * 1024 * 1024)
        chainer.global_config.autotune = True
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gcnn.to_gpu()  # Copy the model to the GPU
    xp = gcnn.xp

    print('----- Setup optimizer -----')
    optimizer = chainer.optimizers.Adam(alpha=args.learning_rate)
    optimizer.setup(gcnn)
    optimizer.add_hook(chainer.optimizer.WeightDecay(2e-4))

    print('----- Load dataset -----')
    # Load the dataset
    path = os.path.join(args.base, args.train_list)
    train = PointCloudDataset(root=args.root, list_path=path,
                                num_vertices=args.num_vertices,
                                num_nearest_neighbor=args.num_nearest_neighbor,
                                xp = xp,
                                augmentation=True)
    class_weights = train.class_weights
    path = os.path.join(args.base, args.val_list)
    val = PointCloudDataset(root=args.root, list_path=path,
                                num_vertices=args.num_vertices,
                                num_nearest_neighbor=args.num_nearest_neighbor,
                                xp = xp,
                                data_type='val')

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    val_iter = chainer.iterators.SerialIterator(val, args.batchsize,
                                                 repeat=False, shuffle=False)

    print('----- Make updater -----')
    updater = GraphCnnUpdater(
        model = gcnn,
        iterator = train_iter,
        optimizer = {'gcnn':optimizer},
        class_weights = class_weights,
        device = args.gpu
        )

    print('----- Make trainer -----')
    trainer = training.Trainer(updater,
                                (args.iteration, 'iteration'),
                                out=os.path.join(args.base, args.out))
    IO.save_args(os.path.join(args.base, args.out), args)

    snapshot_interval = (args.snapshot_interval, 'iteration')
    display_interval = (args.display_interval, 'iteration')
    evaluation_interval = (args.evaluation_interval, 'iteration')
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(gcnn, filename='gcnn_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)

    trainer.extend(GraphCnnEvaluator(val_iter, gcnn, class_weights=class_weights, device=args.gpu), trigger=evaluation_interval)

    trainer.extend(extensions.LogReport(trigger=display_interval))

    trainer.extend(extensions.ProgressBar(update_interval=10))

    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['train/loss', 'val/loss'], 'iteration', file_name='loss.png', trigger=display_interval))
        trainer.extend(extensions.PlotReport(['train/acc', 'val/acc'], 'iteration', file_name='accuracy.png', trigger=display_interval))


    trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    print('----- Run the training -----')
    reset_seed(0)
    trainer.run()


if __name__ == '__main__':
    main()
