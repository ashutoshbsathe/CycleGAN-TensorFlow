import tensorflow as tf
from model import CycleGAN
from reader import Reader 
from datetime import datetime
import os
import logging
from utils import ImagePool

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('batch_size', 1, 'Batch Size, default: 1')
tf.flags.DEFINE_integer('image_width', 320, 'Image Width, default: 320')
tf.flags.DEFINE_integer('image_height', 200, 'Image Height, default: 200')
tf.flags.DEFINE_bool('use_lsgan', True,
    'Use LSGAN(MSE( or CrossEntropyLoss, default: True')
tf.flags.DEFINE_string('norm', 'instance',
    'Normalization [instance/batch], default: instance')
tf.flags.DEFINE_integer('lambda1', 10,
    'Weight of Forward Cycle Loss (A->B->A), default: 10')
tf.flags.DEFINE_integer('lambda2', 10, 
    'Weight of Backward Cycle Loss (B->A->B), default: 10')
tf.flags.DEFINE_float('learning_rate', 2e-4,
    'Learning rate of Adam, default: 2e-4')
tf.flags.DEFINE_float('beta1', 0.5,
    'Momentum of Adam, default: 0.5')
tf.flags.DEFINE_integer('pool_size', 50,
    'Size of Image Buffer that stores previously generated images, default: 50')
tf.flags.DEFINE_integer('ngf', 64,
    'Number of gen filters in first conv layer, default: 64')
tf.flags.DEFINE_string('A', 'pop1topop2/tfrecords/pop1.tfrecords',
    'A tfrecords file for training, default: pop1topop2/tfrecords/pop1.tfrecords')
tf.flags.DEFINE_string('B', 'pop1topop2/tfrecords/pop2.tfrecords',
    'B tfrecords file for training, default: pop1topop2/tfrecords/pop2.tfrecords')
tf.flags.DEFINE_string('load_model', None,
    'Folder of saved model for continuing the training, default: None')
tf.flags.DEFINE_bool('reset_model', False, 
    'Allows you to reset computational graph of tensorflow, default: False')
def train():
    if FLAGS.reset_model:
        tf.reset_default_graph()
    if FLAGS.load_model is not None:
        checkpoints_dir = 'checkpoints/' + FLAGS.load_model.lstrip('checkpoints/')
    else:
        current_time = datetime.now().strftime("%Y%m%d-%H%M")
        checkpoints_dir = 'checkpoints/{}'.format(current_time)
        try:
            os.makedirs(checkpoints_dir)
        except os.error as e:
            print("Unable to make directories for checkpoints")
            print(e)
    graph = tf.Graph()

    with graph.as_default():
        cycle_gan = CycleGAN(
            A_train_file=FLAGS.A,
            B_train_file=FLAGS.B,
            batch_size=FLAGS.batch_size,
            image_width=FLAGS.image_width,
            image_height=FLAGS.image_height,
            use_lsgan=FLAGS.use_lsgan,
            norm=FLAGS.norm,
            lambda1=FLAGS.lambda1,
            lambda2=FLAGS.lambda2,
            learning_rate=FLAGS.learning_rate,
            beta1=FLAGS.beta1,
            ngf=FLAGS.ngf
        )
        G_loss, D_B_loss, F_loss, D_A_loss, fake_b, fake_a = cycle_gan.model()
        optimizers = cycle_gan.optimize(G_loss, D_B_loss, F_loss, D_A_loss)
        
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(checkpoints_dir, graph)
        saver = tf.train.Saver()
    with tf.Session(graph=graph) as sess:
        if FLAGS.load_model is not None:
            checkpoint = tf.train.get_checkpoint_state(checkpoints_dir)
            meta_graph_path = checkpoint.model_checkpoint_path + '.meta'
            restore = tf.train.import_meta_graph(meta_graph_path)
            restore.restore(sess, tf.train.latest_checkpoint(checkpoints_dir))
            step = int(meta_graph_path.split('-')[2].split('.')[0])
            print("Resuming training from step" + str(step))
        else:
            print("Starting fresh training...")
            sess.run(tf.global_variables_initializer())
            step = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            fake_B_pool = ImagePool(FLAGS.pool_size)
            fake_A_pool = ImagePool(FLAGS.pool_size)

            while not coord.should_stop():
                fake_b_val, fake_a_val = sess.run([fake_b, fake_a])
                
                _, G_loss_val, D_B_loss_val, F_loss_val, D_A_loss_val, summary = (
                    sess.run(
                        [optimizers, G_loss, D_B_loss, F_loss, D_A_loss, summary_op],
                        feed_dict={cycle_gan.fake_b: fake_B_pool.query(fake_b_val),
                                   cycle_gan.fake_a: fake_A_pool.query(fake_a_val)}
                    )
                )

                train_writer.add_summary(summary, step)
                train_writer.flush()

                if(step % 10 == 0):
                    logging.info('==============(Step {})=============='.format(step))
                    logging.info('    G_loss    : {}'.format(G_loss_val))
                    logging.info('  D_B_loss    : {}'.format(D_B_loss_val))
                    logging.info('    F_loss    : {}'.format(F_loss_val))
                    logging.info('  D_A_loss    : {}'.format(D_A_loss_val))
                
                if(step % 100 == 0):
                    save_path = saver.save(sess, checkpoints_dir + '/model.ckpt', global_step=step)
                    logging.info('Model saved at {}'.format(save_path))
                step += 1
        except KeyboardInterrupt:
            logging.info('Keyboard Interrupt')
            coord.request_stop()
        except Exception as e:
            coord.request_stop(e)
        finally:
            save_path = saver.save(sess, checkpoints_dir + '/model.ckpt', global_step=step)
            logging.info('Model saved at {}'.format(save_path))
            coord.request_stop()
            coord.join(threads)

def main(unused_argv):
    train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run()
                    