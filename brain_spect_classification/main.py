#coding:utf-8
import os, sys, time
import tensorflow as tf
import random
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.dirname(__file__))
import utils.ioFunctions as IO
import utils.utility_functions as UF
from model import GraphCNN

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def uniform_weight(trainLabel):
    weights = []
    [weights.append(1) for i in range(len(trainLabel))]
    return weights

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
    flags.DEFINE_integer('shuffle_buffer_size', 20,
                        'buffer size of shuffle')
    flags.DEFINE_integer('num_point', 1024,
                        'Number of sampling point')

    flags.DEFINE_integer('iteration', 100001,
                        'number of iteration')
    flags.DEFINE_integer('snapshot_interval', 1000,
                        'snapshot interval')
    flags.DEFINE_integer('display_interval', 1000,
                        'display interval')

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
    train_dataset = train_dataset.map(lambda x: UF._parse_function(x, num_point=FLAGS.num_point),
                                        num_parallel_calls=os.cpu_count())
    train_dataset = train_dataset.shuffle(buffer_size = FLAGS.shuffle_buffer_size)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_iter = train_dataset.make_one_shot_iterator()
    train_data = train_iter.get_next()

    print('----- Init variables -----')
    init_op = tf.group(tf.initializers.global_variables(),
                       tf.initializers.local_variables())

    with tf.Session(config = UF.gpu_config(index=FLAGS.gpu_index)) as sess:
        print('---- Build model ----')
        kwargs = {
            'sess' : sess,
            'output_dir' : output_dir,
            'batch_size': FLAGS.batch_size,
            'learning_rate': 1e-4,
            'num_sampling_point':1024,
            'num_feature':1
        }

        # print number of parmeters
        # UF.calc_parameter()
        gcnn = GraphCNN(**kwargs)
         # prepare tensorboard
        writer_train = tf.summary.FileWriter(os.path.join(tensorboard_dir, 'train'), sess.graph)

        value_loss = tf.Variable(0.0)
        tf.summary.scalar("loss", value_loss)
        merge_op = tf.summary.merge_all()

        # initialize
        sess.run(init_op)

        tbar = tqdm(range(FLAGS.iteration), ascii=True)
        train_total_loss = []
        train_loss_reg = []
        train_acc = []
        for i in tbar:
            batch_label, batch_graph, batch_intensity = sess.run(train_data)
            batch_binarize_label = label_binarize(batch_label, classes=[j for j in range(4)])
            batch_weight = uniform_weight(batch_label)
            total_loss, loss_reg, acc_train = gcnn.update(batch_intensity, batch_graph, batch_binarize_label, batch_weight)

            train_total_loss.append(total_loss)
            train_loss_reg.append(loss_reg)
            train_acc.append(acc_train)

            if i % FLAGS.display_interval is 0:
                s = 'Loss: {:.4f}'.format(np.mean(train_total_loss))
                tbar.set_description(s)

                summary_train_loss = sess.run(merge_op, {value_loss: total_loss})
                writer_train.add_summary(summary_train_loss, i)

                train_total_loss.clear()
                train_loss_reg.clear()
                train_acc.clear()


            if i % FLAGS.snapshot_interval is 0:
                # save model
                gcnn.save_model(i)

if __name__ == '__main__':
    main()
