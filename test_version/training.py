import os
import tensorflow as tf
import numpy as np
import test_version.input_data
import test_version.model

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 10000
LEARNING_RATE = 0.0001
RATIO = 0.2


def run_training():

    train_dir = '/home/yuxin/PycharmProjects/cats_vs_dogs/data/train/'
    logs_train_dir = '/home/yuxin/PycharmProjects/cats_vs_dogs/test_version/logs/train/'
    logs_test_dir = '/home/yuxin/PycharmProjects/cats_vs_dogs/test_version/logs/test/'

    train, train_label, test, test_label = test_version.input_data.get_file(train_dir, RATIO)
    train_batch, train_label_batch = test_version.input_data.get_batch(train,
                                                                       train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
    test_batch, test_label_batch = test_version.input_data.get_batch(test,
                                                                     test_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    logits = test_version.model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    loss = test_version.model.losses(logits, train_label_batch)
    train_op = test_version.model.training(loss, LEARNING_RATE)
    acc = test_version.model.evaluation(logits, train_label_batch)

    x = tf.placeholder(tf.float32, [BATCH_SIZE, 208, 208, 3])
    y_ = tf.placeholder(tf.int16, [BATCH_SIZE])

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        test_writer = tf.summary.FileWriter(logs_test_dir, sess.graph)

        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                    break

                train_images, train_labels = sess.run([train_batch, train_label_batch])
                _, tra_loss, tra_acc = sess.run([train_op, loss, acc], feed_dict={x: train_images, y_: train_labels})

                if step % 50 == 0:
                    print('step: %d, train loss = %.2f, accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    train_writer.add_summary(summary_str, step)

                if step % 200 == 0:
                    test_images, test_labels = sess.run([test_batch, test_label_batch])
                    test_loss, test_acc = sess.run([loss, acc], feed_dict={x: test_images, y_: test_labels})

                    print('**  Step: %d, val loss = %.2f, val accuracy = %.2f%%  **' %
                          (step, test_loss, test_acc * 100.0))
                    summary_str = sess.run(summary_op)
                    test_writer.add_summary(summary_str, step)

                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)

        except tf.errors.OutOfRangeError:
            print('Done training, epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

