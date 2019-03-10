import tensorflow as tf
import ops
import utils
from reader import Reader 
from discriminator import Discriminator 
from generator import Generator 

REAL_LABEL = 0.9

class CycleGAN:
    def __init__(self,
            A_train_file='',
            B_train_file='',
            batch_size=1,
            image_width=320,
            image_height=200,
            use_lsgan=True,
            norm='instance',
            lambda1=10,
            lambda2=10,
            learning_rate=2e-4,
            beta1=0.5,
            ngf=64
        ):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.use_lsgan = use_lsgan
        self.use_sigmoid = not use_lsgan
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.A_train_file = A_train_file
        self.B_train_file = B_train_file

        self.is_training = tf.placeholder_with_default(True, shape=[], name='is_training')

        self.G = Generator('G', self.is_training, ngf=ngf, norm=norm, image_width=image_width, image_height=image_height)
        self.D_B = Discriminator('D_B', self.is_training, norm=norm, use_sigmoid=self.use_sigmoid)
        self.F = Generator('F', self.is_training, norm=norm, image_width=image_width, image_height=image_height)
        self.D_A = Discriminator('D_A', self.is_training, norm=norm, use_sigmoid=self.use_sigmoid)

        self.fake_a = tf.placeholder(tf.float32,
            shape=[batch_size, image_width, image_height, 3])
        self.fake_b = tf.placeholder(tf.float32,
            shape=[batch_size, image_width, image_height, 3])
        
    def model(self):
        A_reader = Reader(self.A_train_file, name='A', image_width=self.image_width, image_height=self.image_height, batch_size=self.batch_size)
        B_reader = Reader(self.B_train_file, name='B', image_width=self.image_width, image_height=self.image_height, batch_size=self.batch_size)

        a = A_reader.feed()
        b = B_reader.feed()

        cycle_loss = self.cycle_consistency_loss(self.G, self.F, a, b)

        # A --> B
        fake_b = self.G(a)
        G_gan_loss = self.generator_loss(self.D_B, fake_b, use_lsgan=self.use_lsgan)
        G_loss = G_gan_loss + cycle_loss
        D_B_loss = self.discriminator_loss(self.D_B, b, self.fake_b, use_lsgan=self.use_lsgan)

        # B --> A
        fake_a = self.F(b)
        F_gan_loss = self.generator_loss(self.D_A, fake_a, use_lsgan=self.use_lsgan)
        F_loss = F_gan_loss + cycle_loss
        D_A_loss = self.discriminator_loss(self.D_A, a, self.fake_a, use_lsgan=self.use_lsgan) 

        tf.summary.histogram('D_B/true', self.D_B(b))
        tf.summary.histogram('D_B/fake', self.D_B(self.G(a)))
        tf.summary.histogram('D_A/true', self.D_A(a))
        tf.summary.histogram('D_A/fake', self.D_A(self.F(b)))

        tf.summary.scalar('loss/G', G_gan_loss)
        tf.summary.scalar('loss/D_B', D_B_loss)
        tf.summary.scalar('loss/F', F_gan_loss)
        tf.summary.scalar('loss/D_A', D_A_loss)
        tf.summary.scalar('loss/cycle', cycle_loss)

        tf.summary.image('A/generated', utils.batch_convert2int(self.G(a)))
        tf.summary.image('A/reconstruction', utils.batch_convert2int(self.F(self.G(a))))
        tf.summary.image('B/generated', utils.batch_convert2int(self.F(b)))
        tf.summary.image('B/reconstruction', utils.batch_convert2int(self.F(self.G(b))))

        return G_loss, D_B_loss, F_loss, D_A_loss, fake_b, fake_a
    
    def optimize(self, G_loss, D_B_loss, F_loss, D_A_loss):
        def make_optimizer(loss, variables, name='Adam'):
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.learning_rate
            end_learning_rate = 0.0
            start_decay_step = 100000
            decay_steps = 100000
            beta1 = self.beta1
            learning_rate = (
                tf.where(
                    tf.greater_equal(global_step, start_decay_step),
                    tf.train.polynomial_decay(starter_learning_rate, global_step-start_decay_step,
                                                decay_steps, end_learning_rate, power=1.0),
                    starter_learning_rate
                )
            )
            tf.summary.scalar('learning_rate/{}'.format(name), learning_rate)

            learning_step = (
                tf.train.AdamOptimizer(learning_rate, beta1-beta1, name=name)
                        .minimize(loss, global_step=global_step, var_list=variables)
            )
            return learning_step
        
        G_optimizer = make_optimizer(G_loss, self.G.variables, name='Adam_G')
        D_B_optimizer = make_optimizer(D_B_loss, self.D_B.variables, name='Adam_D_B')
        F_optimizer = make_optimizer(F_loss, self.F.variables, name='Adam_F')
        D_A_optimizer = make_optimizer(D_A_loss, self.D_A.variables, name='Adam_D_A')

        with tf.control_dependencies([G_optimizer, D_B_optimizer, F_optimizer, D_A_optimizer]):
            return tf.no_op(name='optimizers')
    
    def discriminator_loss(self, D, b, fake_b, use_lsgan=True):
        if use_lsgan:
            error_real = tf.reduce_mean(tf.squared_difference(D(b), REAL_LABEL))
            error_fake = tf.reduce_mean(tf.square(D(fake_b)))
        else:
            # Notice the negative sign carefully
            error_real = -tf.reduce_mean(ops.safe_log(D(b)))
            error_fake = -tf.reduce_mean(ops.safe_log(1 - D(fake_b)))
        loss = (error_real + error_fake) / 2
        return loss
    
    def generator_loss(self, D, fake_b, use_lsgan=True):
        if use_lsgan:
            loss = tf.reduce_mean(tf.squared_difference(D(fake_b), REAL_LABEL))
        else:
            # Notice the negative sign carefully
            loss = -tf.reduce_mean(ops.safe_log(D(fake_b))) / 2
        return loss

    def cycle_consistency_loss(self, G, F, a, b):
        forward_loss = tf.reduce_mean(tf.abs(F(G(a)) - a))
        backward_loss = tf.reduce_mean(tf.abs(F(G(b)) - b))
        loss = self.lambda1 * forward_loss + self.lambda2 * backward_loss
        return loss