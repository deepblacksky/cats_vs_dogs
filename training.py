import os
import tensorflow as tf
import numpy as np
import input_data
import model

N_CLASSES = 2
IMG_W = 208
IMG_H = 208
BATCH_SIZE = 64
CAPACITY = 2000
MAX_STEP = 10000
LEARNING_RATE = 0.0001

##


def run_training():

    train_dir = '/home/yuxin/PycharmProjects/cats_vs_dogs/data/train/'
    logs_train_dir = '/home/yuxin/PycharmProjects/cats_vs_dogs/logs2/train/'

    train, train_label = input_data.get_file(train_dir)
    train_batch, train_label_batch = input_data.get_batch(train,
                                                          train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, LEARNING_RATE)
    train_acc = model.evaluation(train_logits, train_label_batch)

    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])

            if step % 50 == 0:
                print('step: %d, train loss: %.2f, accuracy: %.2f%%' % (step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training, epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


# Evaluate


# from PIL import Image
# import matplotlib.pyplot as plt


# def get_one_image(train):
#     '''Randomly pick one image from training data
#     Return: ndarray
#     '''
#     n = len(train)
#     ind = np.random.randint(0, n)
#     img_dir = train[ind]
#
#     image = Image.open(img_dir)
#     plt.imshow(image)
#     image = image.resize([208, 208])
#     image = np.array(image)
#     return image

#
# import math
#
#
# def evaluate_one_image():
#     """
#         Test one image against the saved models and parameters
#     """
#
#     test_dir = '/home/yuxin/PycharmProjects/cats_vs_dogs/data/test/'
#     N_CLASSES = 2
#     test_number = 12500
#     num_iter = int(math.ceil(test_number / BATCH_SIZE))
#     true_count = 0
#     total_test_number = num_iter * BATCH_SIZE
#     step = 0
#
#     with tf.Graph().as_default():
#
#         test, test_label = input_data.get_file(test_dir)
#         test_batch, test_label_batch = input_data.get_batch(test,
#                                                             test_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#
#         test_logit = model.inference(test_batch, BATCH_SIZE, N_CLASSES)
#
#         logit = tf.nn.softmax(test_logit)
#
#         image_holder = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 208, 208, 3])
#         label_holder = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
#
#         top_k_op = tf.nn.in_top_k(logit, label_holder, 1)
#
#         # you need to change the directories to yours.
#         logs_train_dir = '/home/yuxin/PycharmProjects/cats_vs_dogs/logs/train/'
#
#         saver = tf.train.Saver()
#
#         with tf.Session() as sess:
#
#             print("Reading checkpoints...")
#             ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#             if ckpt and ckpt.model_checkpoint_path:
#                 global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                 saver.restore(sess, ckpt.model_checkpoint_path)
#                 print('Loading success, global_step is %s' % global_step)
#             else:
#                 print('No checkpoint file found')
#             while step < num_iter:
#                 prediction = sess.run([top_k_op], feed_dict={image_holder: test_batch, label_holder: test_label_batch})
#                 true_count += np.sum(prediction)
#                 step += 1
#             precision = true_count / total_test_number
#             print('precision : %.3f' % precision)

