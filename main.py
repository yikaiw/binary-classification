import numpy as np
import tensorflow as tf
from CNN import Network
from datetime import datetime
import os
from reader import Reader
import config
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string('checkpoint', None, 'Whether use a pre-trained checkpoint, default None.')


def main():
    checkpoint_dir = 'checkpoints'
    if FLAGS.checkpoint is not None:
        checkpoint_path = os.path.join(checkpoint_dir, FLAGS.checkpoint.lstrip('checkpoints/'))
    else:
        current_time = datetime.now().strftime('%Y%m%d-%H%M')
        checkpoint_path = os.path.join(checkpoint_dir, '{}'.format(current_time))
        try:
            os.makedirs(checkpoint_path)
        except os.error:
            print('Unable to make checkpoints direction: %s' % checkpoint_path)
    model_save_path = os.path.join(checkpoint_path, 'model.ckpt')

    nn = Network()
    dataset = Reader()

    saver = tf.train.Saver()
    print('Build session.')
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    if FLAGS.checkpoint is not None:
        print('Restore from pre-trained model.')
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        meta_graph_path = checkpoint.model_checkpoint_path + '.meta'
        restore = tf.train.import_meta_graph(meta_graph_path)
        restore.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
        step = int(meta_graph_path.split('-')[2].split('.')[0])
    else:
        print('Initialize.')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        step = 0

    loss_list = []
    accuracy_list = []
    step = 0

    # train_writer = tf.summary.FileWriter('log', sess.graph)
    # summary_op = tf.summary.merge_all()

    print('Start training:')
    for epoch in range(config.num_epochs):
        permutation = np.random.permutation(dataset.train_len)
        X_train_data = dataset.train_set_X[permutation]
        y_train_data = dataset.train_set_y[permutation]
        data_idx = 0
        while data_idx < dataset.train_len:
            X_train_batch = X_train_data[data_idx: np.clip(data_idx + config.batch_size, 0, dataset.train_len - 1)]
            y_train_batch = y_train_data[data_idx: np.clip(data_idx + config.batch_size, 0, dataset.train_len - 1)]
            data_idx += config.batch_size

            loss, train_batch_accuracy, _ = sess.run([nn.loss, nn.batch_accuracy, nn.optimizer],
                {nn.X_inputs: X_train_batch, nn.y_inputs: y_train_batch, nn.keep_prob: config.keep_prob, nn.training: True})
            loss_list.append(loss)

            if step % 10 == 0 and step > 0:
            print('>> At step %i: loss = %.3f, train_batch_accuracy = %.3f%%' % 
                    (step, loss, train_batch_accuracy * 100))

        
        accuracy = sess.run([nn.accuracy],  
            {nn.X_inputs: dataset.test_set_X, nn.y_inputs: dataset.test_set_X, nn.keep_prob: 1.0, nn.training: False})
        accuracy_list.append(accuracy)

        # train_writer.add_summary(summary, step)
        # train_writer.flush(

        print('For epoch %i: val_epoch_accuracy = %.3f%%\n' % (epoch, accuracy * 100))
        
        if step % 1000 == 0 and step > 0:
            save_path = saver.save(sess, model_save_path, global_step=step)
            print('Model saved in file: %s' % save_path)

    save_path = saver.save(sess, model_save_path, global_step=step)
    print('Model saved in file: %s' % save_path)
    sess.close()


if __name__ == '__main__':
    main()
