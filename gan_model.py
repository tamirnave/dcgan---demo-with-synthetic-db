import tensorflow as tf
import numpy as np
import time
import scipy.misc
import os
import re
import math
import database

def conv2d(x, W, b, strides, with_bn='b'):
    x = tf.nn.conv2d(x, W, strides=strides, padding='SAME')
    x = tf.nn.bias_add(x, b)
    if not not with_bn:
        x = tf.contrib.layers.batch_norm(x, is_training=True, epsilon=1e-5, decay=0.9, scale=True,
                                         updates_collections=None, scope=with_bn)
    return lrelu(x)


def conv2d_tran(x, W, b, strides=[1, 1, 1, 1], with_bn='b'):
    out_size = [x.get_shape()[0]._value, 2 * x.get_shape()[1]._value, 2 * x.get_shape()[2]._value,
                W.get_shape()[2]._value]
    x = tf.nn.conv2d_transpose(x, W, out_size, strides, padding='SAME')
    x = tf.nn.bias_add(x, b)
    if not not with_bn:
        return tf.contrib.layers.batch_norm(x, is_training=True, epsilon=1e-5, decay=0.9, scale=True,
                                            updates_collections=None, scope=with_bn)
    else:
        return x

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

class gan:
    def __init__(self,batch_size=64,img_size=64,z_length=100,seed=0):
        self.batch_size = batch_size
        self.img_size = img_size
        self.z_length = z_length
        self.seed = seed

    def Gen_Input(self):
        im = np.random.uniform(-1, 1, (self.batch_size, self.z_length))
        return im

    def Build_Model(self,learning_rate_G=0.0002,learning_rate_D=0.0002,learning_rate_Gi=0.0002,learning_rate_z_opt=0.0002,beta1=0.5):
        tf.set_random_seed(self.seed)
        self.x = tf.placeholder("float32", [self.batch_size, self.img_size, self.img_size, 3])
        self.z = tf.placeholder("float32", [self.batch_size, self.z_length])
        with tf.variable_scope('Inv_Generator') as scope:
            self.Gi_x,_ = self.G_inv(self.x)
            #scope.reuse_variables()
            #self.GiGz, GiGz_logits=self.G_inv(self.Gz)

        with tf.variable_scope('Generator') as scope:
            self.Gz = self.Generator(self.z)
            scope.reuse_variables()
            self.GGi_x= self.Generator(self.Gi_x)

        with tf.variable_scope('Discriminator') as scope:
            self.Dx, Dx_logits = self.Discriminator(self.x)
            scope.reuse_variables()
            self.DGz, DGz_logits = self.Discriminator(self.Gz)

        D_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')
        G_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
        G_inv_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Inv_Generator')

        # When training G we want to minimize: -log(D(G(z))
        # D convinced that Gz is real        => G_cost=0
        # D convinced that Gz is faked       => G_cost=inf
        # D is confused about Gz             => G_cost=0.69
        # When training D we want to minimize: -log(D(x)) - log(1-D(G(z)))
        # D is convinced that x is real and Gz is faked => D_cost=0
        # D is convinced that x is faked and Gz is real => D_cost=inf
        # D is confused about x and Gz                  => D_cost=1.38
        # When training GiGz we want to minimize: log(Ginv(G(z)-z))
        self.G_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DGz_logits, labels=tf.ones_like(self.DGz)))
        self.D_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx_logits, labels=tf.ones_like(self.Dx))) + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=DGz_logits, labels=tf.zeros_like(self.DGz)))
        #self.GiGz_cost = tf.reduce_mean(tf.abs(self.z-self.GiGz))
        self.GGi_x_cost = tf.reduce_mean(tf.square(self.x - self.GGi_x))

        # starter_learning_rate = 0.1
        # global_step_G = tf.Variable(0, trainable=False)
        # global_step_D = tf.Variable(0, trainable=False)
        # learning_rate_G = tf.train.exponential_decay(starter_learning_rate, global_step_G,training_epochs, 0.96)
        # learning_rate_D = tf.train.exponential_decay(starter_learning_rate, global_step_D,training_epochs, 0.96)

        self.G_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_G, beta1=beta1).minimize(self.G_cost,var_list=G_params)  # AdamOptimizer #GradientDescentOptimizer
        self.D_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_D, beta1=beta1).minimize(self.D_cost,var_list=D_params)
        #self.GiGz_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_Gi, beta1=beta1).minimize(self.GiGz_cost,var_list=G_inv_params)
        self.GGi_x_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_Gi, beta1=beta1).minimize(self.GGi_x_cost, var_list=G_inv_params)

    def Generator(self,z):
        w1 = tf.get_variable('w1', [self.z_length, 8192], initializer=tf.random_normal_initializer(stddev=0.02))
        w2 = tf.get_variable('w2', [5, 5, 256, 512], initializer=tf.random_normal_initializer(stddev=0.02))
        w3 = tf.get_variable('w3', [5, 5, 128, 256], initializer=tf.random_normal_initializer(stddev=0.02))
        w4 = tf.get_variable('w4', [5, 5, 64, 128], initializer=tf.random_normal_initializer(stddev=0.02))
        w5 = tf.get_variable('w5', [5, 5, 3, 64], initializer=tf.random_normal_initializer(stddev=0.02))

        b1 = tf.get_variable('b1', [8192], initializer=tf.random_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', [256], initializer=tf.random_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('b3', [128], initializer=tf.random_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('b4', [64], initializer=tf.random_normal_initializer(stddev=0.02))
        b5 = tf.get_variable('b5', [3], initializer=tf.random_normal_initializer(stddev=0.02))

        layer1 = tf.add(tf.matmul(z, w1), b1)
        layer1 = tf.reshape(layer1, [-1, 4, 4, 512])
        layer1 = tf.nn.relu(tf.contrib.layers.batch_norm(layer1, is_training=True, epsilon=1e-5, decay=0.9, scale=True,
                                                         updates_collections=None, scope='bn1'))
        layer2 = tf.nn.relu(conv2d_tran(layer1, w2, b2, [1, 2, 2, 1], 'bn2'))
        layer3 = tf.nn.relu(conv2d_tran(layer2, w3, b3, [1, 2, 2, 1], 'bn3'))
        layer4 = tf.nn.relu(conv2d_tran(layer3, w4, b4, [1, 2, 2, 1], 'bn4'))
        G = tf.nn.tanh(conv2d_tran(layer4, w5, b5, [1, 2, 2, 1], ''))
        return G

    def Discriminator(self,x):
        w1 = tf.get_variable('w1', [5, 5, 3, 64], initializer=tf.random_normal_initializer(stddev=0.02))
        w2 = tf.get_variable('w2', [5, 5, 64, 128], initializer=tf.random_normal_initializer(stddev=0.02))
        w3 = tf.get_variable('w3', [5, 5, 128, 256], initializer=tf.random_normal_initializer(stddev=0.02))
        w4 = tf.get_variable('w4', [5, 5, 256, 512], initializer=tf.random_normal_initializer(stddev=0.02))
        w5 = tf.get_variable('w5', [8192, 1], initializer=tf.random_normal_initializer(stddev=0.02))

        b1 = tf.get_variable('b1', [64], initializer=tf.random_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', [128], initializer=tf.random_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('b3', [256], initializer=tf.random_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('b4', [512], initializer=tf.random_normal_initializer(stddev=0.02))
        b5 = tf.get_variable('b5', [1], initializer=tf.random_normal_initializer(stddev=0.02))

        layer1 = conv2d(x, w1, b1, [1, 2, 2, 1], '')
        layer2 = conv2d(layer1, w2, b2, [1, 2, 2, 1], 'bn2')
        layer3 = conv2d(layer2, w3, b3, [1, 2, 2, 1], 'bn3')
        layer4 = conv2d(layer3, w4, b4, [1, 2, 2, 1], 'bn4')
        layer4 = tf.reshape(layer4, [self.batch_size, -1])
        layer5 = tf.add(tf.matmul(layer4, w5), b5)
        D = tf.nn.sigmoid(layer5)
        return D, layer5

    def G_inv(self,x):
        w1 = tf.get_variable('w1', [5, 5, 3, 64], initializer=tf.random_normal_initializer(stddev=0.02))
        w2 = tf.get_variable('w2', [5, 5, 64, 128], initializer=tf.random_normal_initializer(stddev=0.02))
        w3 = tf.get_variable('w3', [5, 5, 128, 256], initializer=tf.random_normal_initializer(stddev=0.02))
        w4 = tf.get_variable('w4', [5, 5, 256, 512], initializer=tf.random_normal_initializer(stddev=0.02))
        w5 = tf.get_variable('w5', [8192, self.z_length], initializer=tf.random_normal_initializer(stddev=0.02))

        b1 = tf.get_variable('b1', [64], initializer=tf.random_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', [128], initializer=tf.random_normal_initializer(stddev=0.02))
        b3 = tf.get_variable('b3', [256], initializer=tf.random_normal_initializer(stddev=0.02))
        b4 = tf.get_variable('b4', [512], initializer=tf.random_normal_initializer(stddev=0.02))
        b5 = tf.get_variable('b5', [self.z_length], initializer=tf.random_normal_initializer(stddev=0.02))

        c1 = tf.constant(-0.5,dtype='float32',shape=[1])
        c2 = tf.constant(2,dtype='float32',shape=[1]) # self.batch_size,self.z_length

        layer1 = conv2d(x, w1, b1, [1, 2, 2, 1], '')
        layer2 = conv2d(layer1, w2, b2, [1, 2, 2, 1], 'bn2')
        layer3 = conv2d(layer2, w3, b3, [1, 2, 2, 1], 'bn3')
        layer4 = conv2d(layer3, w4, b4, [1, 2, 2, 1], 'bn4')
        layer4 = tf.reshape(layer4, [self.batch_size, -1])
        layer5 = tf.add(tf.matmul(layer4, w5), b5)
        G_i = tf.multiply(tf.add(tf.nn.sigmoid(layer5),c1),c2)
        return G_i, layer5

    def Sample_Gz(self, sess,trial_z=None):
        if trial_z is None:
            trial_z=self.Gen_Input()
        return sess.run(self.Gz, {self.z: trial_z})

    def Sample_G_inv(self, sess,x):
        return sess.run(self.Gi_x, {self.x: x})

    def Train_G_Inv(self, sess, dat, bench_name, training_epochs=1, display_step=1, save_step=5,saving_obj=None,from_epoch=0):
        start = time.time()
        costs = np.zeros((training_epochs*(math.ceil(dat.DB_size/dat.Batch_size)), 2))
        ind=0
        for epoch in range(from_epoch+1,from_epoch+1+training_epochs):  # Loop on all epochs
            eoe = False  # end of epoch flag
            while not eoe:  # Loop on all batches
                x_in, eoe = dat.Get_Next_Batch()
                dict_po = {self.x: x_in}
                Gi_c, _ = sess.run([self.GGi_x_cost, self.GGi_x_optimizer], dict_po)
                costs[ind] = Gi_c
                ind=ind+1

            if epoch % display_step == 0:
                end = time.time()
                print("Epoch:", '%04d' % (epoch), "cost Gi=", "{:.4f}".format(Gi_c),"Time: ", "{:2.2f}".format(end - start))
                z = self.Sample_G_inv(sess, x_in)
                test_image_restored = self.Sample_Gz(sess, z)
                scipy.misc.imsave(bench_name + '/samples/train_Gi' + str(epoch) + '.png', np.concatenate((database.Rest_Img(x_in[0, :, :, :]), database.Rest_Img(test_image_restored[0, :, :, :]))))
                start = time.time()

            if (epoch % save_step == 0 or epoch==from_epoch+training_epochs) and not not saving_obj:
                saving_obj.save(sess, bench_name + '/checkpoints/G_inv/G_inv', epoch)
                # End of Training

        return costs

    def Train_GD(self,sess,dat,bench_name,training_epochs,D_train_epochs=1,G_train_epochs=1,display_step=1,save_step=5,saving_obj=None,from_epoch=0):
        start = time.time()
        costs = np.zeros((training_epochs*(math.ceil(dat.DB_size/dat.Batch_size)), 2))
        ind=0
        for epoch in range(from_epoch+1,from_epoch+1+training_epochs):  # Loop on all epochs
            eoe = False  # end of epoch flag

            while not eoe:  # Loop on all batches
                x_in, eoe = dat.Get_Next_Batch()
                z_in = self.Gen_Input()
                dict_po = {self.x: x_in, self.z: z_in}
                D_c = 0
                G_c = 0

                # Train D
                for train_ind in range(D_train_epochs):
                    D_c, _ = sess.run([self.D_cost, self.D_optimizer], dict_po)

                # Train G
                for train_ind in range(G_train_epochs):
                    G_c, _, gz = sess.run([self.G_cost, self.G_optimizer, self.Gz], dict_po)

                costs[ind,:]=[D_c, G_c]
                ind=ind+1

            if epoch % display_step == 0:
                end = time.time()
                print("Epoch:", '%04d' % (epoch), "cost D=", "{:.4f}".format(D_c), "cost G=", "{:.4f}".format(G_c),
                      "Time: ", "{:2.2f}".format(end - start))
                scipy.misc.imsave(bench_name + '/samples/train_' + str(epoch) + '.png', database.Rest_Img(gz[0, :, :, :]))
                start = time.time()

            if (epoch % save_step == 0 or epoch==from_epoch+training_epochs) and not not saving_obj:
                saving_obj.save(sess, bench_name + '/checkpoints/GD/GD', epoch)
                # End of Training

        return costs

    def Restore_Checkpoint(self,sess,sav,checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            sav.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            return counter
        else:
            return -1