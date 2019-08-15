import tensorflow as tf
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np


def create_record(data_path, records_path):
    writer = tf.io.TFRecordWriter(records_path)
    for n in range(0, 30):
        img = os.path.join(data_path, 'image', str(n) + '.png')
        label = os.path.join(data_path, 'label', str(n) + '.png')
        img = Image.open(img)
        label = Image.open(label)
        img = img.resize((512, 512))
        label = label.resize((512, 512))
        img_raw = img.tobytes()
        label_raw = label.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_raw])),
                    'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }
            )
        )
        writer.write(example.SerializeToString())
    writer.close()


def read_record(filename, batch_size):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'img': tf.FixedLenFeature([], tf.string),
        }
    )
    img = features['img']
    label = features['label']
    img = tf.decode_raw(img, tf.uint8)
    label = tf.decode_raw(label, tf.uint8)

    img = tf.cast(img, dtype=tf.int32)
    label = tf.cast(label, dtype=tf.float32)

    label = label / 255
    label = tf.greater(label, 0.5)
    label = tf.cast(label, dtype=tf.int32)

    img = tf.reshape(img, [512, 512, 1])
    label = tf.reshape(label, [512, 512, 1])

    data = tf.concat([img, label], axis=2)

    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)

    data = tf.transpose(data, [2, 0, 1])
    img = data[0]
    label = data[1]

    img = tf.cast(img, dtype=tf.float32)
    img = tf.reshape(img, [512, 512, 1])
    img = tf.image.per_image_standardization(img)

    min_after_dequeue = 30
    capacity = min_after_dequeue + 3 * batch_size
    img, label = tf.train.shuffle_batch([img, label],
                                        batch_size=batch_size,
                                        num_threads=6,
                                        capacity=capacity,
                                        min_after_dequeue=min_after_dequeue)
    return img, label


if __name__ == '__main__':
    if not os.path.exists('./data/train/train.tfrecords'):
        create_record('data/train', './data/train/train.tfrecords')
    else:
        print('TFRecords already exists!')

    img, label = read_record('./data/train/train.tfrecords', 1)
    sess = tf.Session()
    tf.train.start_queue_runners(sess=sess)
    x, y = sess.run([img, label])

    x = np.reshape(x, (512, 512))
    y = np.reshape(y, (512, 512))
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(x)
    ax[0].axis('off')

    ax[1].imshow(y, cmap='gray')
    ax[1].axis('off')
    plt.show()
