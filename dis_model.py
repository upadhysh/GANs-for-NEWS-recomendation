import tensorflow as tf
import cPickle
import numpy as np
def assign_as_blocks_v2(a,b):
        shape1 = tf.shape(a)
        shape2 = tf.shape(b)
        m1 = shape1[0]
        n1 = shape1[1]
        m2 = shape2[0]
        n2 = shape2[1]
        p1 = tf.tile(a,[1,n2])
        p2 = tf.reshape(tf.tile(tf.expand_dims(b, 1),[1,1,n1]), [m2,-1])
        out = tf.concat((p1,p2),axis=0)
        return out

class DIS():
    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=0.05, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []
        self.hidden_num_units = 5

        with tf.variable_scope('discriminator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])


        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        self.x = tf.concat([self.u_embedding, self.i_embedding],1)

        self.weights = {
                'hidden': tf.Variable(tf.random_normal([2*self.emb_dim, self.hidden_num_units], seed=4350)),
                'output': tf.Variable(tf.random_normal([self.hidden_num_units, 1], seed=4350))
        }

        self.biases = {
                'hidden': tf.Variable(tf.random_normal([self.hidden_num_units], seed=4350)),
                'output': tf.Variable(tf.random_normal([1], seed=4350))
        }
        
        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias, self.weights['hidden'], self.weights['output'], self.biases['hidden'], self.biases['output']]

        self.hidden_layer = tf.add(tf.matmul(self.x, self.weights['hidden']), self.biases['hidden'])

        self.hidden_layer = tf.nn.relu(self.hidden_layer)

        self.pre_logits = tf.matmul(self.hidden_layer, self.weights['output']) + self.biases['output']

        # self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
                                                                logits=self.pre_logits) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        )

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)
        
        #reward_logits calculation
        temp_embedding = tf.reshape(self.u_embedding, [-1, 1])
        shape = tf.shape(self.i_embedding)
        conc_embedding = tf.transpose(tf.tile(temp_embedding, [1, shape[0]]))
        self.input_embed = tf.concat([conc_embedding, self.i_embedding], 1)
        
        hidden_layer = tf.add(tf.matmul(self.input_embed, self.weights['hidden']), self.biases['hidden'])

        hidden_layer = tf.nn.relu(hidden_layer)

        self.reward_logits = tf.matmul(hidden_layer, self.weights['output']) + self.biases['output']

        # self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
                                           # 1) + self.i_bias
        #reward_logits calculation



        self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)

        #all_rating calculation
        # for test stage, self.u: [batch_size]
        # self.all_pairs = [[ tf.concat(x,y) for x in self.u_embedding ] for y in self.item_embeddings]
        self.all_pairs = tf.transpose(assign_as_blocks_v2(tf.transpose(self.u_embedding), tf.transpose(self.item_embeddings)))
        hidden_layer = tf.add(tf.matmul(self.all_pairs, self.weights['hidden']), self.biases['hidden'])
        hidden_layer = tf.nn.relu(hidden_layer)
        self.all_rating = tf.matmul(hidden_layer, self.weights['output']) + self.biases['output']
        # self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    # transpose_b=True) + self.item_bias
        #all_rating calculation
        
        #all_logits calculation
        shape2 = tf.shape(self.item_embeddings)
        conc_embedding2 = tf.transpose(tf.tile(temp_embedding, [1, shape2[0]]))
        self.input_embed2 = tf.concat([conc_embedding2, self.item_embeddings], 1)

        hidden_layer = tf.add(tf.matmul(self.input_embed2, self.weights['hidden']), self.biases['hidden'])

        hidden_layer = tf.nn.relu(hidden_layer)

        self.all_logits = tf.matmul(hidden_layer, self.weights['output']) + self.biases['output']
        # self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        #all_logits calculation


        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        )
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'w'))
