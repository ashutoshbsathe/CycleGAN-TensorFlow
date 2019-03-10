import tensorflow as tf
import random
import os

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('A_input_dir', 'pop1topop2/trainA',
        'trainA input directory, default: pop1topop2/trainA')
tf.flags.DEFINE_string('B_input_dir', 'pop1topop2/trainB',
        'trainB input directory, default: pop1topop2/trainB')
tf.flags.DEFINE_string('A_output_file', 'pop1topop2/tfrecords/pop1.tfrecords',
        'trainA output TFRecords File, default: pop1topop2/tfrecords/pop1.tfrecords')
tf.flags.DEFINE_string('B_output_file', 'pop1topop2/tfrecords/pop2.tfrecords',
        'trainB output TFRecords File, default: pop1topop2/tfrecords/pop2.tfrecords')

def data_reader(input_dir, shuffle=True):
    files = []
    for img_file in os.listdir(input_dir):
        if img_file.endswith('.jpg'):
            files.append(os.path.join(input_dir, img_file))

    if shuffle:
        shuffled_index = list(range(len(files)))
        random.seed(3141592)
        random.shuffle(shuffled_index)

        files = [files[i] for i in shuffled_index]

    return files

def _int64_feature(val):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convert_to_example(file_path, image_buffer):
    file_name = file_path.split('/')[-1]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/file_name': _bytes_feature(tf.compat.as_bytes(os.path.basename(file_name))),
        'image/encoded_image': _bytes_feature((image_buffer))
    }))
    return example

def data_writer(input_dir, output_file):
    files = data_reader(input_dir)

    output_dir = os.path.dirname(output_file)
    try:
        os.makedirs(output_dir)
    except os.error as e:
        print("Unable to make directories")
        print(e)
    
    num_images = len(files)

    writer = tf.python_io.TFRecordWriter(output_file)

    for i in range(num_images):
        file_path = files[i]
        with tf.gfile.FastGFile(file_path, 'rb') as f:
            image_data = f.read()
        example = _convert_to_example(file_path, image_data)
        writer.write(example.SerializeToString())

        if i % 100 == 0:
            print("Processed {}/{}...".format(i, num_images))
    print("Processing completed")
    print("------------------------------------------------------------")
    writer.close()

def main(unused_argv):
    print("------------------------------------------------------------")
    print("Converting trainA data to tfrecords...")
    data_writer(FLAGS.A_input_dir, FLAGS.A_output_file)
    print("------------------------------------------------------------")
    print("Converting trainB data to tfrecords...")
    data_writer(FLAGS.B_input_dir, FLAGS.B_output_file)
    print("------------------------------------------------------------")

if __name__ == '__main__':
    tf.app.run()