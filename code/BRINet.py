import numpy as np
import tensorflow as tf
import sys
sys.path.append('./external/TF-resnet')
sys.path.append('./external/TF-deeplab')
import resnet_model
import deeplab_model

from util import data_reader
from util.processing_tools import *
from util import im_processing, text_processing, eval_tools
from util import loss


class BRI_model(object):

    def __init__(self,  batch_size = 1, 
                        num_steps = 20,
                        vf_h = 40,
                        vf_w = 40,
                        H = 320,
                        W = 320,
                        vf_dim = 512,
                        vf_dim2 = 1024,
                        vf_dim3 = 2048,
                        vocab_size = 12112,
                        w_emb_dim = 1000,
                        v_emb_dim = 1000,
                        conv_dim = 1000,
                        atrous_dim = 512,
                        mlp_dim = 500,
                        start_lr = 0.00025,
                        lr_decay_step = 750000,
                        lr_decay_rate = 1.0,
                        rnn_size = 1000,
                        keep_prob_rnn = 1.0,
                        keep_prob_emb = 1.0,
                        keep_prob_mlp = 1.0,
                        num_rnn_layers = 1,
                        optimizer = 'adam',
                        weight_decay = 0.0005,
                        mode = 'eval',
                        weights = 'resnet'):
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.vf_h = vf_h
        self.vf_w = vf_w
        self.H = H
        self.W = W
        self.vf_dim = vf_dim
        self.vf_dim2 = vf_dim2
        self.vf_dim3 = vf_dim3
        self.start_lr = start_lr
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        self.vocab_size = vocab_size
        self.w_emb_dim = w_emb_dim
        self.v_emb_dim = v_emb_dim
        self.conv_dim = conv_dim
        self.atrous_dim = atrous_dim
        self.mlp_dim = mlp_dim
        self.rnn_size = rnn_size
        self.keep_prob_rnn = keep_prob_rnn
        self.keep_prob_emb = keep_prob_emb
        self.keep_prob_mlp = keep_prob_mlp
        self.num_rnn_layers = num_rnn_layers
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.mode = mode
        self.weights = weights
        self.words = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
        self.im = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 3])
        self.target_fine = tf.placeholder(tf.float32, [self.batch_size, self.H, self.W, 1])

        if self.weights == 'resnet':
            resmodel = resnet_model.ResNet(batch_size=self.batch_size, 
                                        atrous=True,
                                        images=self.im,
                                        labels=tf.constant(0.))
            self.visual_feat = resmodel.logits

        elif self.weights == 'deeplab':
            resmodel = deeplab_model.DeepLab(batch_size=self.batch_size,
                                        images=self.im,
                                        labels=tf.constant(0.))
            self.visual_feat3 = resmodel.res3c
            self.visual_feat4 = resmodel.res4c
            self.visual_feat5 = resmodel.res5c

        with tf.variable_scope("text_objseg"):
            self.build_graph()
            if self.mode == 'eval':
                return
            self.train_op()

    def build_graph(self):

        if self.weights == 'deeplab':

            visual_feat3 = self._conv("conv_3", self.visual_feat3, 1, self.vf_dim, self.v_emb_dim, [1, 1, 1, 1])
            visual_feat4 = self._conv("conv_4", self.visual_feat4, 1, self.vf_dim2, self.v_emb_dim, [1, 1, 1, 1])
            visual_feat5 = self._conv("conv_5", self.visual_feat5, 1, self.vf_dim3, self.v_emb_dim, [1, 1, 1, 1])

        elif self.weights == 'resnet':
            visual_feat = self.visual_feat

        #12112-->1000
        embedding_mat = tf.get_variable("embedding", [self.vocab_size, self.w_emb_dim],
                                        initializer=tf.random_uniform_initializer(minval=-0.08, maxval=0.08))
        embedded_seq = tf.nn.embedding_lookup(embedding_mat, tf.transpose(self.words))
        #word_reshape = tf.reshape(embedded_seq, [self.num_steps, self.rnn_size])
        rnn_cell_w = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size, state_is_tuple=False)
        cell_w = tf.nn.rnn_cell.MultiRNNCell([rnn_cell_w] * self.num_rnn_layers, state_is_tuple=False)

        #LSTM
        state_w = cell_w.zero_state(self.batch_size, tf.float32)
        state_w_shape = state_w.get_shape().as_list()
        state_w_shape[0] = self.batch_size
        state_w.set_shape(state_w_shape)
        h_a = tf.zeros([self.batch_size, self.rnn_size])
        ht_all = []
        def f1():
            return state_w, h_a

        def f2():
            # Word input to embedding layer
            w_emb = embedded_seq[n, :, :]
            with tf.variable_scope("WLSTM"):
                h_w, state_w_ret = cell_w(w_emb, state_w)
            return state_w_ret, h_w

        with tf.variable_scope("RNN"):
            for n in range(self.num_steps):
                if n > 0:
                    tf.get_variable_scope().reuse_variables()
                state_w, h_a = tf.cond(tf.equal(self.words[0, n], tf.constant(0)), lambda: f1(), lambda: f2())
                ht_all.append(h_a)
        ht_all = tf.reshape(ht_all, [self.num_steps, self.rnn_size])
        q_final = tf.tile(tf.reshape(h_a, [1, 1, 1, self.rnn_size]), [1, self.vf_h, self.vf_w, 1])
        q_word = tf.transpose(ht_all)
        visual_feat3 = tf.nn.l2_normalize(visual_feat3, 3)
        visual_feat4 = tf.nn.l2_normalize(visual_feat4, 3)
        visual_feat5 = tf.nn.l2_normalize(visual_feat5, 3)
        spatial = tf.convert_to_tensor(generate_spatial_batch(self.batch_size, self.vf_h, self.vf_w))
        #  BCAM
        #  Res_5
        three_fuse_5 = tf.concat([q_final, visual_feat5, spatial], 3)
        B_5 = self.BCAM('x5', three_fuse_5, q_word)
        up5 = self._conv("conv_sum5", B_5, 1, B_5.shape[3], 1, [1, 1, 1, 1])
        self.up1_5 = up5

        #  Res_4
        three_fuse_4 = tf.concat([q_final, visual_feat4, spatial], 3)
        B_4 = self.BCAM('x4', three_fuse_4, q_word)
        up4 = self._conv("conv_sum4", B_4, 1, B_4.shape[3], 1, [1, 1, 1, 1])
        self.up1_4 = up4

        #  Res_3
        three_fuse_3 = tf.concat([q_final, visual_feat3, spatial], 3)
        B_3 = self.BCAM('x3', three_fuse_3, q_word)
        up3 = self._conv("conv_sum3", B_3, 1, B_3.shape[3], 1, [1, 1, 1, 1])
        self.up1_3 = up3

        #GBFM
        out = self.GBFM(B_3, B_4, B_5)
        conv2 = self._conv("conv2_g", out, 1, out.shape[3], 1, [1, 1, 1, 1])
        self.pred = conv2
        self.up = tf.image.resize_bilinear(self.pred, [self.H, self.W])
        self.sigm = tf.sigmoid(self.up)

    def GBFM(self, B_3, B_4, B_5):
        # left
        l_34 = self.get_gate('l_34', B_4, B_3)
        l_45 = self.get_gate('l_45', B_5, B_4)
        left = self.get_gate('left', l_45, l_34)
        # right
        r_34 = self.get_gate('r_34', B_3, B_4)
        r_45 = self.get_gate('r_45', B_4, B_5)
        right = self.get_gate('right', r_34, r_45)
        # fuse
        out = left + right
        out = self._conv("out_feature", out, 3, out.shape[3], self.conv_dim, [1, 1, 1, 1])
        out = tf.nn.relu(out)
        return out

    def get_gate(self, name, x1, x2):
        with tf.variable_scope(name):
            c_12 = tf.concat([x1, x2], 3)
            c_12 = self._conv("feature", c_12, 1, c_12.shape[3], self.conv_dim, [1, 1, 1, 1])
            c_12 = tf.nn.relu(c_12)
            gate = self._conv("gate", c_12, 3, c_12.shape[3], self.conv_dim, [1, 1, 1, 1])
            gate = tf.nn.sigmoid(gate)
            out = gate * x1 + x2
        return out

    def BCAM(self, name, v, q):
        with tf.variable_scope(name):
            W_q = tf.get_variable('W_q', [self.conv_dim, 500], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            W_v = tf.get_variable('W_v', [self.conv_dim, 500], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            W_a = tf.get_variable('W_a', [self.vf_h * self.vf_w, 500], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            #VLAM
            v_1 = self._conv("conv_v1", v, 1, v.shape[3], self.rnn_size, [1, 1, 1, 1])
            v_1 = tf.nn.relu(v_1)
            re_v1 = tf.reshape(v_1, [self.vf_h * self.vf_w, self.conv_dim])
            re_q = tf.reshape(q, [self.conv_dim, self.num_steps])
            a_t = tf.matmul(re_v1, re_q)
            a_t = tf.nn.softmax(a_t, axis=1)
            key_word_mat = tf.matmul(a_t, tf.transpose(re_q))
            key_word_mat_r = tf.reshape(key_word_mat, [1, self.vf_h, self.vf_w, self.conv_dim])
            #LVAM
            v_2 = self._conv("conv_v2", v, 1, v.shape[3], self.rnn_size, [1, 1, 1, 1])
            v_2 = tf.nn.relu(v_2)
            re_v2 = tf.reshape(v_2, [self.vf_h * self.vf_w, self.conv_dim])
            A = tf.tanh(tf.matmul(key_word_mat, W_q) + tf.matmul(re_v2, W_v))
            rel_map = tf.nn.softmax(tf.matmul(W_a, tf.transpose(A)), axis=1)
            v_3 = self._conv("conv_v3", v, 1, v.shape[3], self.rnn_size, [1, 1, 1, 1])
            v_3 = tf.nn.relu(v_3)
            re_v3 = tf.transpose(tf.reshape(v_3, [self.vf_h * self.vf_w, self.conv_dim]))
            out = tf.matmul(re_v3, rel_map)
            out = tf.reshape(tf.transpose(out), [1, self.vf_h, self.vf_w, self.conv_dim])
            C_con = tf.concat([out, key_word_mat_r], 3)
            C_out = self._conv("conv_C1", C_con, 1, C_con.shape[3], self.rnn_size, [1, 1, 1, 1])
            C_out = tf.nn.relu(C_out)
            #fuse
            v_4 = self._conv("conv_v4", v, 1, v.shape[3], self.rnn_size, [1, 1, 1, 1])
            v_4 = tf.nn.relu(v_4)
            C_fuse = tf.nn.relu(C_out) + v_4
            #ASPP
            atrous_C_1 = self._atrous_conv("atrous_C_1", C_fuse, 3, C_fuse.shape[3], self.atrous_dim, 1)
            atrous_C_3 = self._atrous_conv("atrous_C_3", C_fuse, 3, C_fuse.shape[3], self.atrous_dim, 3)
            atrous_C_5 = self._atrous_conv("atrous_C_5", C_fuse, 3, C_fuse.shape[3], self.atrous_dim, 5)
            atrous_C_7 = self._atrous_conv("atrous_C_7", C_fuse, 3, C_fuse.shape[3], self.atrous_dim, 7)
            atrous_con = tf.concat([atrous_C_1, atrous_C_3, atrous_C_5, atrous_C_7, C_fuse], 3)
            final_out = self._conv("conv_final_out", atrous_con, 1, atrous_con.shape[3], self.conv_dim, [1, 1, 1, 1])
            final_out = tf.nn.relu(final_out)

            return final_out

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters], 
                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.conv2d(x, w, strides, padding='SAME') + b

    def _atrous_conv(self, name, x, filter_size, in_filters, out_filters, rate):
        with tf.variable_scope(name):
            w = tf.get_variable('DW', [filter_size, filter_size, in_filters, out_filters],
                initializer=tf.random_normal_initializer(stddev=0.01))
            b = tf.get_variable('biases', out_filters, initializer=tf.constant_initializer(0.))
            return tf.nn.atrous_conv2d(x, w, rate=rate, padding='SAME') + b

    def train_op(self):
        # define loss
        self.target = tf.image.resize_bilinear(self.target_fine, [self.vf_h, self.vf_w])
        tvars = [var for var in tf.trainable_variables() if var.op.name.startswith('text_objseg') or var.op.name.startswith('ResNet/fc1000')]
        reg_var_list = [var for var in tvars if var.op.name.find(r'DW') > 0]
        self.mid_loss_5 = loss.weighed_logistic_loss(self.up1_5, self.target)
        self.mid_loss_4 = loss.weighed_logistic_loss(self.up1_4, self.target)
        self.mid_loss_3 = loss.weighed_logistic_loss(self.up1_3, self.target)

        self.cls_loss = loss.weighed_logistic_loss(self.pred, self.target)
        self.reg_loss = loss.l2_regularization_loss(reg_var_list, self.weight_decay)
        self.cost = self.cls_loss + self.reg_loss + self.mid_loss_5 + self.mid_loss_4 + self.mid_loss_3

        # learning rate
        lr = tf.Variable(0.0, trainable=False)
        self.learning_rate = tf.train.polynomial_decay(self.start_lr, lr, self.lr_decay_step, end_learning_rate=0.00001, power=0.9)

        # optimizer
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise ValueError("Unknown optimizer type %s!" % self.optimizer)

        # learning rate multiplier
        grads_and_vars = optimizer.compute_gradients(self.cost, var_list=tvars)
        var_lr_mult = {var: (2.0 if var.op.name.find(r'biases') > 0 else 1.0) for var in tvars}
        grads_and_vars = [((g if var_lr_mult[v] == 1 else tf.multiply(var_lr_mult[v], g)), v) for g, v in grads_and_vars]

        # training step
        self.train_step = optimizer.apply_gradients(grads_and_vars, global_step=lr)