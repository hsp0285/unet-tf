from utils import *
import time
from datetime import timedelta
from data import *


class UNet(object):

    def __init__(self, sess, conf):
        """
        本函数是UNet网络的初始化文件，用于构建网络结构、损失函数、优化方法。
        :param sess: tensorflow会话
        :param conf: 配置
        """
        self.sess = sess
        self.conf = conf
        # 输入图像
        self.images = tf.placeholder(tf.float32, shape=[None, 512, 512, 1], name='x')
        # 标注
        self.annotations = tf.placeholder(tf.int64, shape=[None, 512, 512], name='annotations')
        # 构建UNet网络结构
        self.predict = self.inference()
        # 损失函数，分类精度
        self.loss_op, self.accuracy_op = self.loss()
        # 优化方法
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.conf.learning_rate)\
            .minimize(self.loss_op, name='train_op')
        # 初始化参数
        self.sess.run(tf.global_variables_initializer())
        # 保存所有可训练的参数
        trainable_vars = tf.trainable_variables()
        # 模型保存和保存summary的工具
        self.saver = tf.compat.v1.train.Saver(var_list=trainable_vars, max_to_keep=0)
        self.writer = tf.compat.v1.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.train_summary = self.config_summary('train')

    def config_summary(self, name):
        summarys = [tf.summary.scalar(name + '/loss', self.loss_op),
                    tf.summary.scalar(name + '/accuracy', self.accuracy_op)]
        summary = tf.summary.merge(summarys)
        return summary

    def inference(self):
        conv1 = conv(self.images, shape=[3, 3, 1, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv1 = conv(conv1, shape=[3, 3, 64, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool1 = max_pool(conv1, size=2)

        conv2 = conv(pool1, shape=[3, 3, 64, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv2 = conv(conv2, shape=[3, 3, 128, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool2 = max_pool(conv2, size=2)

        conv3 = conv(pool2, shape=[3, 3, 128, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv3 = conv(conv3, shape=[3, 3, 256, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool3 = max_pool(conv3, size=2)

        conv4 = conv(pool3, shape=[3, 3, 256, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv4 = conv(conv4, shape=[3, 3, 512, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool4 = max_pool(conv4, size=2)

        conv5 = conv(pool4, shape=[3, 3, 512, 1024], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv5 = conv(conv5, shape=[3, 3, 1024, 1024], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up6 = deconv(conv5, shape=[2, 2, 512, 1024], stride=2, stddev=0.1)
        merge6 = concat(up6, conv4)
        conv6 = conv(merge6, shape=[3, 3, 512, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv6 = conv(conv6, shape=[3, 3, 512, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up7 = deconv(conv6, shape=[2, 2, 256, 512], stride=2, stddev=0.1)
        merge7 = concat(up7, conv3)
        conv7 = conv(merge7, shape=[3, 3, 256, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv7 = conv(conv7, shape=[3, 3, 256, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up8 = deconv(conv7, shape=[2, 2, 128, 256], stride=2, stddev=0.1)
        merge8 = concat(up8, conv2)
        conv8 = conv(merge8, shape=[3, 3, 128, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv8 = conv(conv8, shape=[3, 3, 128, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up9 = deconv(conv8, shape=[2, 2, 64, 128], stride=2, stddev=0.1)
        merge9 = concat(up9, conv1)
        conv9 = conv(merge9, shape=[3, 3, 64, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv9 = conv(conv9, shape=[3, 3, 64, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)

        predict = conv(conv9, shape=[3, 3, 64, 2], stddev=0.1, is_training=self.conf.is_training, stride=1)

        return predict

    def loss(self):
        """
        :return: 损失函数及分类精确度
        """
        # 标注 1通道-num_classes通道 one-hot
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.annotations, logits=self.predict)
        loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        # 预测的类别
        decoded_predictions = tf.argmax(self.predict, 3, name='accuracy/decode_pred')
        correct_prediction = tf.equal(self.annotations, decoded_predictions, name='accuracy/correct_pred')
        accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32, name='accuracy/cast'),
                                     name='accuracy/accuracy_op')
        return loss_op, accuracy_op

    def save(self, step):
        print('---->saving', step)
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        self.saver.save(self.sess, checkpoint_path, global_step=step)

    def reload(self, step):
        checkpoint_path = os.path.join(self.conf.modeldir, self.conf.model_name)
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('-------no such checkpoint', model_path)
            return
        self.saver.restore(self.sess, model_path)

    def save_summary(self, summary, step):
        print('---->summarizing', step)
        self.writer.add_summary(summary, step)

    def train(self):
        if self.conf.reload_step > 0:
            if not self.reload(self.conf.reload_step):
                return
            print('reload', self.conf.reload_step)

        images, labels = read_record('./data/train/train.tfrecords', 16)

        tf.train.start_queue_runners(sess=self.sess)

        print('Begin Train')
        for train_step in range(1, self.conf.max_step + 1):
            start_time = time.time()

            x, y = self.sess.run([images, labels])

            # summary
            if train_step == 1 or train_step % self.conf.summary_interval == 0:
                feed_dict = {self.images: x,
                             self.annotations: y}
                loss, acc, _, summary = self.sess.run(
                    [self.loss_op, self.accuracy_op, self.optimizer, self.train_summary],
                    feed_dict=feed_dict)
                print(str(train_step), '----Training loss:', loss, ' accuracy:', acc, end=' ')
                self.save_summary(summary, train_step + self.conf.reload_step)
            # print 损失和准确性
            else:
                feed_dict = {self.images: x,
                             self.annotations: y}
                loss, acc, _ = self.sess.run(
                    [self.loss_op, self.accuracy_op, self.optimizer], feed_dict=feed_dict)
                print(str(train_step), '----Training loss:', loss, ' accuracy:', acc, end=' ')
            # 保存模型
            if train_step % self.conf.save_interval == 0:
                self.save(train_step + self.conf.reload_step)
            end_time = time.time()
            time_diff = end_time - start_time
            print("Time usage: " + str(timedelta(seconds=int(round(time_diff)))))
