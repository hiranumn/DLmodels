import tensorflow as tf
import numpy as np

class VAE:
    def __init__(self, dimensions, fn=tf.nn.elu, dropout=1.0):
        # Reset all existing tensors
        tf.reset_default_graph()
        
        # Define parameters of the encoder
        self.dimensions = dimensions
        self.fn = fn
        self._lambda = 0.0
        self.learning_rate = 0.001
        self._dropout = 1.0
        self.built = False
        self.sesh = tf.Session()
        self.e = 0
        
        # Tracking data
        self.learning_curve = []
        self.latent_record = {"z":[], "y":[]}
        
        # Building the graph
        self.ops = self.build()
        self.sesh.run(tf.global_variables_initializer())
    
    def build(self):
        # Placeholders for input and dropout probs.
        if self.built:
            return -1
        else:
            self.built = True
        x = tf.placeholder(tf.float32, shape=[None, self.dimensions[0]], name="x")
        dropout = tf.placeholder(tf.float32, shape=[], name="dropout_keepprob")
        
        # Fully connected encoder.
        with tf.variable_scope("encoder"):
            dense = x
            for dim in self.dimensions[1:-1]:
                dense = tf.contrib.slim.fully_connected(dense, dim, activation_fn=self.fn)
                dense = tf.contrib.slim.dropout(dense, keep_prob=dropout)
                
        with tf.name_scope("latent"):
            # Latent distribution defined.
            z_mean = tf.contrib.slim.fully_connected(dense, self.dimensions[-1], activation_fn=tf.identity)
            z_logsigma = tf.contrib.slim.fully_connected(dense, self.dimensions[-1], activation_fn=tf.identity)
            
        z = self.sample(z_mean, z_logsigma)
        
        # Fully connected decoder.
        with tf.variable_scope("decoder"):
            dense = z
            for i in range(len(self.dimensions[1:-1])):
                dim = self.dimensions[1:-1][-1*i-1]
                dense = tf.contrib.slim.fully_connected(dense, dim, activation_fn=self.fn)
                dense = tf.contrib.slim.dropout(dense, keep_prob=dropout)
            reconstructed = tf.contrib.slim.fully_connected(dense, self.dimensions[0], activation_fn=tf.nn.sigmoid)
        
        # Defining the loss components.
        rec_loss = self.crossEntropy(reconstructed, x)
        kl_loss = self.kullbackLeibler(z_mean, z_logsigma)
        
        # Regularize weights by l2 if necessary
        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(v) for v in tf.trainable_variables() if "weights" in v.name]
            l2_reg = self._lambda * tf.add_n(regularizers)
            
        # Define cost as the sum of KL and reconstrunction ross with BinaryXent.
        with tf.name_scope("cost"):
            # average over minibatch
            cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
            cost += l2_reg
        
        # Defining optimization procedure.
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) for grad, tvar in grads_and_vars]
            train = optimizer.apply_gradients(clipped, name="minimize_cost")
            
        # Graph for reconstrunction from latent input.
        # Variables are shared with the regular decoder. Hence, it will be trained simultaneously.
        with tf.variable_scope("decoder", reuse=True):
            z_ = tf.placeholder_with_default(tf.random_normal([1, self.dimensions[-1]]),
                                            shape=[None, self.dimensions[-1]],
                                            name="latent_input")
            dense = z_
            for i in range(len(self.dimensions[1:-1])):
                dim = self.dimensions[1:-1][-1*i-1]
                dense = tf.contrib.slim.fully_connected(dense, dim, activation_fn=self.fn)
                dense = tf.contrib.slim.dropout(dense, keep_prob=dropout)
                
            reconstructed_ = tf.contrib.slim.fully_connected(dense, self.dimensions[0], activation_fn=tf.nn.sigmoid)
            
        # Exporting out the operaions as dictionary
        return dict(
            dropout_keepprob = dropout,
            x = x,  
            z_mean = z_mean, 
            z_logsigma = z_logsigma,
            z = z,
            latent_input = z_,
            reconstructed = reconstructed,
            reconstructed_ = reconstructed_,
            rec_loss = rec_loss,
            kl_loss = kl_loss,
            cost = cost,
            train = train
        )
    
    # Closing session
    def close(self):
        self.sesh.close()
    
    # ReparameterizationTrick
    def sample(self, mu, log_sigma):
        with tf.name_scope("sample_reparam"):
            epsilon = tf.random_normal(tf.shape(log_sigma), name="0mean1varGaus")
            return mu + epsilon * tf.exp(log_sigma)

    # Binary cross-entropy (Adapted from online source)
    def crossEntropy(self, obs, actual, offset=1e-7):
        with tf.name_scope("BinearyXent"):
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) +
                                  (1 - actual) * tf.log(1 - obs_), 1)
        
    # KL divergence between Gaussian with mu and log_sigma, q(z|x) vs 0-mean 1-variance Gaussian p(z).
    def kullbackLeibler(self, mu, log_sigma):
        with tf.name_scope("KLD"):
            return -0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu**2 - tf.exp(2 * log_sigma), 1)
    
    # training procedure.
    def train(self, X, epochs, valid=None):
        # Making the saver object.
        saver = tf.train.Saver()
        
        # Defining the number of batches per epoch
        batch_num = int(np.ceil(X.n*1.0/X.batch_size))
        if valid != None:
            val_batch_num = int(np.ceil(valid.n*1.0/valid.batch_size))
        
        e = 0
        while e < epochs:
            epoch_cost = {"kld":[], "rec":[], "cost":[], "validcost":[]}
            
            if e == epochs-1: self.latent_record = {"z":[], "y":[]}
            
            for i in range(batch_num):
                #Training happens here.
                batch = X.next()
                feed_dict = {self.ops["x"]: batch[0], self.ops["dropout_keepprob"]: self._dropout}
                ops_to_run = [self.ops["reconstructed"], self.ops["z_mean"], self.ops["cost"],\
                              self.ops["kl_loss"], self.ops["rec_loss"], self.ops["train"]]
                reconstruction, z, cost, kld, rec, _= self.sesh.run(ops_to_run, feed_dict)
                
                if e == epochs-1: self.latent_record["z"] = self.latent_record["z"] + [_ for _ in z]
                if e == epochs-1: self.latent_record["y"] = self.latent_record["y"] + [_ for _ in batch[1]]
                epoch_cost["kld"].append(np.mean(kld))
                epoch_cost["rec"].append(np.mean(rec))
                epoch_cost["cost"].append(cost)
            
            if valid != None:
                for i in range(val_batch_num):
                    batch = valid.next()
                    feed_dict = {self.ops["x"]: batch[0], self.ops["dropout_keepprob"]: 1.0}
                    cost = self.sesh.run(self.ops["cost"], feed_dict)
                    epoch_cost["validcost"].append(cost)
            self.e+=1
            e+= 1
                
            print "Epoch:"+str(self.e), "train_cost:", np.mean(epoch_cost["cost"]),
            if valid != None: print "valid_cost:", np.mean(epoch_cost["validcost"]),
            print "(", np.mean(epoch_cost["kld"]), np.mean(epoch_cost["rec"]), ")"
            self.learning_curve.append(epoch_cost)
    
    # Encode examples
    def encode(self, x):
        feed_dict = {self.ops["x"]: x, self.ops["dropout_keepprob"]: 1.0}
        return self.sesh.run([self.ops["z_mean"], self.ops["z_logsigma"]], feed_dict=feed_dict)

    # Decode latent examples. Other_wise, draw from N(0,1)
    def decode(self, zs=None):
        # (np.array | tf.Variable) -> np.array
        feed_dict = dict()
        if zs is not None:
            feed_dict = {self.ops["latent_input"]: zs, self.ops["dropout_keepprob"]: 1.0}
        return self.sesh.run(self.ops["reconstructed_"], feed_dict)