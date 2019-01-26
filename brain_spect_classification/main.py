#coding:utf-8
import os, sys, time
import tensorflow as tf
import random
import numpy as np

sys.path.append(os.path.dirname(__file__))
import utils.ioFunctions as IO

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def main():
    flags = tf.flags
    flags.DEFINE_string('base', os.path.dirname(os.path.abspath(__file__)),
                        'base directory path of program files')
    flags.DEFINE_string('out', 'results/training_{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S')),
                        'Directory to output the result')
    flags.DEFINE_string('train_list', 'configs/train_list.txt',
                        'Training data list')
    flags.DEFINE_string('val_list', 'configs/val_list.txt',
                        'Validation data list')
    flags.DEFINE_string('gpu_index', '0',
                        'GPU index')
    flags.DEFINE_integer('batch_size', 12,
                        'Batch size')
    flags.DEFINE_integer('shuffle_buffer_size', 5,
                        'buffer size of shuffle')
    flags.DEFINE_integer('num_point', 1024,
                        'Number of sampling point')

    flags.DEFINE_string('root', os.path.dirname(os.path.abspath(__file__)),
                        'Root directory path of graph')
    FLAGS = flags.FLAGS
    reset_seed()

    output_dir = os.path.join(FLAGS.base, FLAGS.out)
    tensorboard_dir = os.path.join(output_dir, 'tensorboard')
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    print('----- Read data -----')
    train_data_list = IO.read_data_list(os.path.join(FLAGS.base, FLAGS.train_list))
    train_data_list = [os.path.join(FLAGS.root, i) for i in train_data_list]
    random.shuffle(train_data_list)
    val_data_list = IO.read_data_list(os.path.join(FLAGS.base, FLAGS.val_list))
    val_data_list = [os.path.join(FLAGS.root, i) for i in val_data_list]

    train_dataset = tf.data.TFRecordDataset(train_data_list, compression_type = 'GZIP')
    train_dataset = train_dataset.map(lambda x: IO._parse_function(x, num_point=FLAGS.num_point),
                              num_parallel_calls=os.cpu_count())
    train_dataset = train_dataset.shuffle(buffer_size = FLAGS.shuffle_buffer_size)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_iter = train_dataset.make_one_shot_iterator()
    train_data = train_iter.get_next()

    with tf.Session() as sess:
        (label, _, _) = sess.run(train_data)
        print(label)


if __name__ == '__main__':
    main()
