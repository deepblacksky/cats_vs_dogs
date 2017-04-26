import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt


train_dir = "/home/yuxin/PycharmProjects/cats_vs_dogs/data/train/"


def get_file(file_dir):
    """
    :param file_dir:file directory  
    :return:list(directory) of image and label 
    """
    cats = []
    dogs = []
    label_cats = []
    label_dogs = []
    for file in os.listdir(file_dir):
        name = file.split(sep='.')
        if name[0] == 'cat':
            cats.append(file_dir + file)
            label_cats.append(0)
        else:
            dogs.append(file_dir + file)
            label_dogs.append(1)

    print("There are %d cats/n There are %d dogs" % (len(cats), len(dogs)))
    image_list = np.hstack((cats, dogs))
    label_list = np.hstack((label_cats, label_dogs))

    temp = np.array([image_list, label_list])
    temp = np.transpose(temp)
    np.random.shuffle(temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def get_batch(image, label, image_W, image_H, batch_size, capacity):
    """
    :param image: list(directory) of image
    :param label: list(directory) of label
    :param image_W: image width
    :param image_H: image height
    :param batch_size: batch size
    :param capacity: the maximum elements in queue
    :return:
            image_batch: 4-D tensor [batch_size, width, height, 3], dtype = tf.float32
            label_batch: 1-D tensor [batch_size], tf.int32
    """
    image = tf.cast(image, tf.string)   # image is list of directory
    label = tf.cast(label, tf.int32)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_content = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_content, channels=3)

    ############################################
    # you can change data of image for feature #
    ############################################

    image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=64, capacity=capacity)

    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)

    return image_batch, label_batch


# if __name__ == '__main__':
#
#     BATCH_SIZE = 2
#     CAPACITY = 256
#     IMG_W = 208
#     IMG_H = 208
#
#     image_list, label_list = get_file(train_dir)
#     image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H,
#                                          batch_size=BATCH_SIZE, capacity=CAPACITY)
#
#     with tf.Session() as sess:
#         i = 0
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord=coord)
#
#         try:
#             while not coord.should_stop() and i < 5:
#                 img, label = sess.run([image_batch, label_batch])
#
#                 for j in np.arange(BATCH_SIZE):
#                     print("label: %d" % label[j])
#                     plt.imshow(img[j, :, :, :])
#                     plt.show()
#                 i += 1
#         except tf.errors.OutOfRangeError:
#             print("done")
#         finally:
#             coord.request_stop()
#         coord.join(threads)
#
