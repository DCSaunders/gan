from __future__ import division
import argparse
import os
import numpy as np
import tensorflow as tf
import random
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import cPickle
FLAGS = None

def batchnormalize(X, eps=1e-8, g=None, b=None):
    if X.get_shape().ndims == 4:
        mean = tf.reduce_mean(X, [0,1,2])
        std = tf.reduce_mean( tf.square(X-mean), [0,1,2] )
        X = (X-mean) / tf.sqrt(std+eps)
        if g is not None and b is not None:
            g = tf.reshape(g, [1,1,1,-1])
            b = tf.reshape(b, [1,1,1,-1])
            X = X*g + b
    elif X.get_shape().ndims == 2:
        mean = tf.reduce_mean(X, 0)
        std = tf.reduce_mean(tf.square(X-mean), 0)
        X = (X-mean) / tf.sqrt(std+eps)

        if g is not None and b is not None:
            g = tf.reshape(g, [1,-1])
            b = tf.reshape(b, [1,-1])
            X = X*g + b
    else:
        raise NotImplementedError
    return X

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * X + f2 * tf.abs(X)

def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(o, t))

class DCGAN():
    def __init__(self, batch_size=200, image_shape=[28,28,1], dim_z=100,
                 dim_y=10, dim_W1=1024, dim_W2=128, dim_W3=64, dim_channel=1):
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.dim_z = dim_z
        self.dim_y = dim_y

        self.dim_W1 = dim_W1
        self.dim_W2 = dim_W2
        self.dim_W3 = dim_W3
        self.dim_channel = dim_channel

        self.gen_W1 = tf.Variable(
            tf.random_normal([dim_z+dim_y, dim_W1], stddev=0.02), name='gen_W1')
        self.gen_W2 = tf.Variable(
            tf.random_normal([dim_W1+dim_y, dim_W2*7*7], stddev=0.02), name='gen_W2')
        self.gen_W3 = tf.Variable(
            tf.random_normal([5,5,dim_W3,dim_W2+dim_y], stddev=0.02), name='gen_W3')
        self.gen_W4 = tf.Variable(
            tf.random_normal([5,5,dim_channel,dim_W3+dim_y], stddev=0.02), name='gen_W4')

        self.discrim_W1 = tf.Variable(
            tf.random_normal([5,5,dim_channel+dim_y,dim_W3], stddev=0.02), name='discrim_W1')
        self.discrim_W2 = tf.Variable(
            tf.random_normal([5,5,dim_W3+dim_y,dim_W2], stddev=0.02), name='discrim_W2')
        self.discrim_W3 = tf.Variable(
            tf.random_normal([dim_W2*7*7+dim_y,dim_W1], stddev=0.02), name='discrim_W3')
        self.discrim_W4 = tf.Variable(
            tf.random_normal([dim_W1+dim_y,1], stddev=0.02), name='discrim_W4')

    def build_model(self):
        Z = tf.placeholder(tf.float32, [self.batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [self.batch_size, self.dim_y])
        image_real = tf.placeholder(tf.float32, [self.batch_size]+self.image_shape)
        raw_image_gen = self.generate(Z,Y)
        image_gen = tf.nn.sigmoid(raw_image_gen)
        raw_real = self.discriminate(image_real, Y)
        p_real = tf.nn.sigmoid(raw_real)
        raw_gen = self.discriminate(image_gen, Y)
        p_gen = tf.nn.sigmoid(raw_gen)
        discrim_cost_real = bce(raw_real, tf.ones_like(raw_real))
        discrim_cost_gen = bce(raw_gen, tf.zeros_like(raw_gen))
        discrim_cost = discrim_cost_real + discrim_cost_gen
        gen_cost = bce(raw_gen, tf.ones_like(raw_gen))
        return Z, Y, image_real, discrim_cost, gen_cost, p_real, p_gen

    def discriminate(self, image, Y):
        yb = tf.reshape(Y, tf.pack([self.batch_size, 1, 1, self.dim_y]))
        X = tf.concat(3, [image, yb*tf.ones([self.batch_size, 28, 28, self.dim_y])])
        h1 = lrelu(
            tf.nn.conv2d(X, self.discrim_W1, strides=[1,2,2,1], padding='SAME'))
        h1 = tf.concat(3, [h1, yb*tf.ones([self.batch_size, 14, 14, self.dim_y])])
        h2 = lrelu(batchnormalize(
            tf.nn.conv2d(h1, self.discrim_W2, strides=[1,2,2,1], padding='SAME')))
        h2 = tf.reshape(h2, [self.batch_size, -1])
        h2 = tf.concat(1, [h2, Y])
        h3 = lrelu( batchnormalize(tf.matmul(h2, self.discrim_W3 )))
        out = tf.concat(1, [h3, Y])
        return out

    def generate(self, Z, Y):
        yb = tf.reshape(Y, [self.batch_size, 1, 1, self.dim_y])
        Z = tf.concat(1, [Z,Y])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z, self.gen_W1)))
        h1 = tf.concat(1, [h1, Y])
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [self.batch_size,7,7,self.dim_W2])
        h2 = tf.concat( 3, [h2, yb*tf.ones([self.batch_size, 7, 7, self.dim_y])])

        output_shape_l3 = [self.batch_size,14,14,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat( 3, [h3, yb*tf.ones([self.batch_size, 14,14,self.dim_y])] )

        output_shape_l4 = [self.batch_size,28,28,self.dim_channel]
        out = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        return out

    def samples_generator(self, batch_size):
        Z = tf.placeholder(tf.float32, [batch_size, self.dim_z])
        Y = tf.placeholder(tf.float32, [batch_size, self.dim_y])

        yb = tf.reshape(Y, [batch_size, 1, 1, self.dim_y])
        Z_ = tf.concat(1, [Z,Y])
        h1 = tf.nn.relu(batchnormalize(tf.matmul(Z_, self.gen_W1)))
        h1 = tf.concat(1, [h1, Y])
        h2 = tf.nn.relu(batchnormalize(tf.matmul(h1, self.gen_W2)))
        h2 = tf.reshape(h2, [batch_size,7,7,self.dim_W2])
        h2 = tf.concat( 3, [h2, yb*tf.ones([batch_size, 7, 7, self.dim_y])])

        output_shape_l3 = [batch_size,14,14,self.dim_W3]
        h3 = tf.nn.conv2d_transpose(h2, self.gen_W3, output_shape=output_shape_l3, strides=[1,2,2,1])
        h3 = tf.nn.relu( batchnormalize(h3) )
        h3 = tf.concat( 3, [h3, yb*tf.ones([batch_size, 14,14,self.dim_y])] )

        output_shape_l4 = [batch_size,28,28,self.dim_channel]
        h4 = tf.nn.conv2d_transpose(h3, self.gen_W4, output_shape=output_shape_l4, strides=[1,2,2,1])
        x = tf.nn.sigmoid(h4)
        return Z,Y,x

def regen(visualize_dim):
    with open(FLAGS.load_samples, 'rb') as f_in:
        gen_samples = cPickle.load(f_in)
        for idx, sample_set in enumerate(gen_samples):
            plot_generated(idx, sample_set, visualize_dim)

def plot_generated(idx, generated, visualize_dim=196):
    shape = int(np.sqrt(visualize_dim))
    fig, ax = plt.subplots(shape, shape, figsize=(1.2*shape, 1.2*shape))
    for index, im in enumerate(generated):
        if index < visualize_dim:
            row, col = index // shape, index % shape
            ax[row, col].imshow(np.reshape(im, (28, 28)))
            ax[row, col].axis('off')
    plt.suptitle('Epoch {}'.format(idx))
    plt.show()

def save_model(sess, saver, gen_samples):
    if FLAGS.save_model:
        saver.save(sess, FLAGS.save_model)
    if FLAGS.save_samples:
        with open(FLAGS.save_samples, 'wb') as f_out:
            cPickle.dump(gen_samples, f_out)

def get_noise_sample(batch_dim, dim_z):
    if FLAGS.normal_prior:
        return np.random.randn(batch_dim, dim_z).astype(np.float32)
    else:
        return np.random.uniform(-1, 1, size=(batch_dim,dim_z)).astype(np.float32)
    
def get_gen_label(batch_dim, label_in=None):
    label = np.array([np.zeros(10) for _ in range(batch_dim)])
    for arr in label:
        if label_in is not None:
            arr[label_in] = 1
        else:
            arr[np.random.randint(10)] = 1
    return label

def model(visualize_dim, train=True):
    n_epochs = FLAGS.max_epochs
    pretrain_batches = FLAGS.pretrain
    learning_rate = 0.0002
    batch_size = 200
    image_shape = [28, 28, 1]
    dim_z = FLAGS.noise_dim
    dim_W1 = 512
    dim_W2 = 128
    dim_W3 = 64
    dim_channel = 1
    mnist = input_data.read_data_sets('MNISt_Data', one_hot=True)

    dcgan_model = DCGAN(batch_size=batch_size, image_shape=image_shape,
            dim_z=dim_z, dim_W1=dim_W1, dim_W2=dim_W2, dim_W3=dim_W3)

    Z_tf, Y_tf, image_tf, d_cost_tf, g_cost_tf, p_real, p_gen = dcgan_model.build_model()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()

    discrim_vars = filter(lambda x: x.name.startswith('discrim'), tf.trainable_variables())
    gen_vars = filter(lambda x: x.name.startswith('gen'), tf.trainable_variables())
    discrim_vars = [i for i in discrim_vars]
    gen_vars = [i for i in gen_vars]

    train_op_discrim = tf.train.AdamOptimizer(
        learning_rate, beta1=0.5).minimize(d_cost_tf, var_list=discrim_vars)
    train_op_gen = tf.train.AdamOptimizer(
        learning_rate, beta1=0.5).minimize(g_cost_tf, var_list=gen_vars)

    Z_tf_sample, Y_tf_sample, image_tf_sample = dcgan_model.samples_generator(
        batch_size=visualize_dim)
    if FLAGS.load_model:
        saver.restore(sess, FLAGS.load_model)
    else:
        tf.initialize_all_variables().run()

    Z_np_sample = get_noise_sample(visualize_dim, dim_z)
    Y_np_sample = get_gen_label(visualize_dim)
    iterations = 0
    k = 2
    if train:
        for batch in range(pretrain_batches):
            Xs, Ys = mnist.train.next_batch(batch_size)
            Xs = Xs.reshape( [-1, 28, 28, 1]) 
            Zs = get_noise_sample(batch_size, dim_z)
            _, discrim_loss_val = sess.run([train_op_discrim, d_cost_tf],
                                           feed_dict={Z_tf:Zs, Y_tf:Ys, image_tf:Xs})
            print("batch: {}, discrim loss: {}".format(batch, discrim_loss_val))
    gen_samples=[]
    for epoch in range(n_epochs):
        batch_count = int(mnist.train.num_examples / batch_size)
        for iterations in range(0, batch_count):
            Xs, Ys = mnist.train.next_batch(batch_size)
            Xs = Xs.reshape( [-1, 28, 28, 1]) 
            Zs = get_noise_sample(batch_size, dim_z)
            if train:
                if np.mod(iterations, k ) != 0:
                    _, gen_loss_val = sess.run([train_op_gen, g_cost_tf],
                                               feed_dict={Z_tf:Zs,
                                                          Y_tf:Ys})
                    discrim_loss_val, p_real_val, p_gen_val = sess.run([d_cost_tf,p_real,p_gen],
                                                                       feed_dict={Z_tf:Zs,
                                                                                  image_tf:Xs, Y_tf:Ys})
                    print("Updating generator: iteration {}, g loss {}, d loss {}".format(
                        iterations, gen_loss_val, discrim_loss_val))
                else:
                    _, discrim_loss_val = sess.run([train_op_discrim, d_cost_tf],
                                                   feed_dict={Z_tf:Zs,
                                                              Y_tf:Ys,
                                                              image_tf:Xs})
                    gen_loss_val, p_real_val, p_gen_val = sess.run([g_cost_tf, p_real, p_gen],
                                                                   feed_dict={Z_tf:Zs,
                                                                              image_tf:Xs, Y_tf:Ys})
                    print("Updating discriminator: iteration {}, g loss {}, d loss {}".format(
                        iterations, gen_loss_val, discrim_loss_val))

                print("Average P(real)={}, P(gen)={}".format(p_real_val.mean(), p_gen_val.mean()))
            if np.mod(iterations, 200) == 0:
                if FLAGS.condition:
                    Y_np_sample = get_gen_label(visualize_dim, label_in=FLAGS.label)
                generated_samples = sess.run(image_tf_sample,
                                             feed_dict={Z_tf_sample:Z_np_sample,
                                                        Y_tf_sample:Y_np_sample})
                gen_samples.append(generated_samples)
                if FLAGS.plot:
                    plot_generated(epoch, generated_samples, visualize_dim)
    save_model(sess, saver, gen_samples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model', type=str, default='/tmp/model.ckpt',
                        help='Location for saving model')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Directory from which to load model')
    parser.add_argument('--load_samples', type=str, default=None,
                        help='Location from which to load pickled samples')
    parser.add_argument('--save_samples', type=str, default=None,
                        help='Location to pickle samples')
    parser.add_argument('--plot', default=False, action='store_true',
                        help='Set if plotting samples')
    parser.add_argument('--condition', default=False, action='store_true',
                        help='Set if conditioning on a label')
    parser.add_argument('--normal_prior', default=False, action='store_true',
                        help='Set if noise prior from normal distribution')
    parser.add_argument('--label', type=int, default=2, 
                        help='Sets label to generate from')
    parser.add_argument('--pretrain', type=int, default=50, 
                        help='Number of pretrain batches')
    parser.add_argument('--max_epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--noise_dim', type=int, default=50, 
                        help='Dimension of noise input to generator')
    parser.add_argument('--visualize_samples', type=int, default=100, 
                        help='Number of samples to plot each epoch')

    FLAGS = parser.parse_args()
    tf.set_random_seed(1234)
    np.random.seed(1234)
    random.seed(1234)
    if FLAGS.load_samples:
        regen(FLAGS.visualize_samples)
    elif FLAGS.load_model:
        model(FLAGS.visualize_samples, train=False)
    else:
        model(FLAGS.visualize_samples)
