import tensorflow as tf 
import utils

class Reader():
    def __init__(self, tfrecords_file, image_width=320, image_height=200,
        min_queue_examples=1000, batch_size=1, num_threads=8, name=''):
        self.tfrecords_file = tfrecords_file
        self.image_height = image_height
        self.image_width = image_width
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.name = name
    def feed(self):
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])
            reader = tf.TFRecordReader()
            _, serialized_example = self.reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image/file_name': tf.FixedLenFeature([], tf.string),
                    'image/encoded_image': tf.FixedLenFeature([], tf.string),
                }
            )
            image_buffer = features['image/encoded_image']
            image = tf.image.decode_jpeg(image_buffer, channels=3)
            image = self._preprocess(image)
            images = tf.train.shuffle_batch(
                [image], batch_size=self.batch_size, num_threads=self.num_threads,
                capacity=self.min_queue_examples + 3 * self.batch_size,
                min_after_dequeue=self.min_queue_examples
            )
            tf.summary.image('_input', images)
        return images
    def _preprocess(self, image):
        image = tf.image.resize_images(image, size=(self.image_height, self.image_width))
        image = utils.convert2float(image)
        image.set_shape([self.image_height, self.image_width, 3])
        return image
    def test_reader():
        TRAIN_FILE_1 = 'pop1topop2/tfrecords/pop1.tfrecords'
        TRAIN_FILE_2 = 'pop1topop2/tfrecords/pop2.tfrecords'

        with tf.Graph().as_default():
            reader1 = Reader(TRAIN_FILE_1, batch_size=2)
            reader2 = Reader(TRAIN_FILE_2, batch_size=2)
            images_op1 = reader1.feed()
            images_op2 = reader2.feed()

            sess = tf.Session()
            init = tf.global_variables_initializer()
            sess.run(init)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                step  = 0
                while not coord.should_stop():
                    batch_images1, batch_images2 = sess.run([images_op1, images_op2])
                    print('Image shape: {}'.format(batch_images1))
                    print('Image shape: {}'.format(batch_images2))
                    print('='*10)
                    step += 1
            except KeyboardInterrupt:
                print('Interrupted')
                coord.request_stop()
            except Exception as e:
                print('Exception Occurred')
                print(e)
                coord.request_stop()
            finally:
                coord.request_stop()
                coord.join(threads)

if __name__ == '__main__':
    test_reader()