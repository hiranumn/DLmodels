import tensorflow as tf
import numpy as np

class VAE3D:
    def __init__(self, latent_dim=200):
        
        tf.reset_default_graph()
        
        # Define parameters of the encoder
        self.input_dim = (32,32,32,1)
        
        # Dimension of the sub-netowrks.]
        self.latent_dim = latent_dim
        self.generator_dim = [512, 256, 64, 1]
        self.discriminator_dim = [64, 256, 512, self.latent_dim]
        self.batchsize = 32
        
        # Activation function is tf.nn.elu
        self.gen_fn = tf.nn.relu
        self.dis_fn = tf.nn.elu
        
        # Other parameters
        self._lambda = 0.0
        self.learning_rate = 0.001
        self._dropout = 1.0
        self.training = True
        
        # Build netowrk
        self.built = False
        self.sesh = tf.Session()
        self.e = 0
        
        # Tracking data
        self.learning_curve = []
        self.record = {"z":[], "reconstructed":[]}
        
        # Building the graph
        self.ops = self.build()
        self.sesh.run(tf.global_variables_initializer())
        
    def build(self):
        # Placeholders for input and dropout probs.
        if self.built:
            return -1
        else:
            self.built = True
            
        x = tf.placeholder(tf.float32, shape=[self.batchsize]+list(self.input_dim), name="x")
        bn = tf.placeholder(tf.bool, shape=[])
        
        # Fully connected encoder.
        dense = self.encoder(x, bn)
        dense = tf.reshape(dense, (self.batchsize, self.latent_dim))
                
        with tf.name_scope("latent"):
            # Latent distribution defined.
            z_mean = tf.contrib.slim.fully_connected(dense, self.discriminator_dim[-1], activation_fn=tf.identity)
            z_logsigma = tf.contrib.slim.fully_connected(dense, self.discriminator_dim[-1], activation_fn=tf.identity)
            
        z = self.sample(z_mean, z_logsigma)
        
        reconstructed = self.decoder(z, bn)
        
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
        
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
        # Defining optimization procedure.
            with tf.name_scope("Adam_optimizer"):

                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                tvars = tf.trainable_variables()
                grads_and_vars = optimizer.compute_gradients(cost, tvars)
                clipped = [(tf.clip_by_value(grad, -5, 5), tvar) for grad, tvar in grads_and_vars]
                train = optimizer.apply_gradients(clipped, name="minimize_cost")
        
        # Latent reconstruction
        z_ = tf.placeholder_with_default(tf.random_normal([self.batchsize, self.latent_dim]),
                                            shape=[self.batchsize, self.latent_dim],
                                            name="latent_input")
        reconstructed_ = self.decoder(z_, bn, reuse=True)
            
        # Exporting out the operaions as dictionary
        return dict(
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
            train = train,
            bn = bn
        )
    
    def encoder(self, _input, bn, reuse=None):
        with tf.variable_scope("encoder", reuse=reuse):
            
            # layer 1
            n_filt = self.discriminator_dim[0]
            layer1 = tf.layers.conv3d(_input, n_filt, (4, 4, 4), strides=(2,2,2), padding="same", activation=None)
            layer1 = tf.contrib.layers.batch_norm(layer1, is_training=bn)
            layer1 = self.dis_fn(layer1)
            
            # layer 2
            n_filt = self.discriminator_dim[1]
            layer2 = tf.layers.conv3d(layer1, n_filt, (4, 4, 4), strides=(2,2,2), padding="same", activation=None)
            layer2 = tf.contrib.layers.batch_norm(layer2, is_training=bn)
            layer2 = self.dis_fn(layer2)
            
            # layer 3
            n_filt = self.discriminator_dim[2]
            layer3 = tf.layers.conv3d(layer2, n_filt, (4, 4, 4), strides=(2,2,2), padding="same", activation=None)
            layer3 = tf.contrib.layers.batch_norm(layer3, is_training=bn)
            layer3 = self.dis_fn(layer3)
            
            # layer 4
            n_filt = self.discriminator_dim[3]
            layer4 = tf.layers.conv3d(layer3, n_filt, (4, 4, 4), strides=(1,1,1), padding="valid", activation=None)
            layer4 = tf.contrib.layers.batch_norm(layer4, is_training=bn)
            layer4 = tf.nn.sigmoid(layer4)
            
            return layer4
        
    def decoder(self, _input, bn, reuse=None):
        with tf.variable_scope("decoder", reuse=reuse):
            _input = tf.reshape(_input, (self.batchsize, 1, 1, 1, self.latent_dim))
            
            # layer 1 (outputs: 512x4x4x4)
            n_filt = self.generator_dim[0]
            layer1 = tf.layers.conv3d_transpose(_input, n_filt, (4, 4, 4), strides=(1,1,1), padding="valid", activation=None)
            layer1 = tf.contrib.layers.batch_norm(layer1, is_training=bn)
            layer1 = self.gen_fn(layer1)
            
            # layer 2 (outputs: 256x8x8x8)
            n_filt = self.generator_dim[1]
            layer2 = tf.layers.conv3d_transpose(layer1, n_filt, (4, 4, 4), strides=(2,2,2), padding="same", activation=None)
            layer2 = tf.contrib.layers.batch_norm(layer2, is_training=bn)
            layer2 = self.gen_fn(layer2)
            
            # layer 3 (outputs: 128x16x16x16)
            n_filt = self.generator_dim[2]
            layer3 = tf.layers.conv3d_transpose(layer2, n_filt, (4, 4, 4), strides=(2,2,2), padding="same", activation=None)
            layer3 = tf.contrib.layers.batch_norm(layer3, is_training=bn)
            layer3 = self.gen_fn(layer3)
            
            # layer 4 (outputs: 64x32x32x32)
            n_filt = self.generator_dim[3]
            layer4 = tf.layers.conv3d_transpose(layer3, n_filt, (4, 4, 4), strides=(2,2,2), padding="same", activation=None)
            layer4 = tf.contrib.layers.batch_norm(layer4, is_training=bn)
            layer4 = tf.nn.sigmoid(layer4)
            
            return layer4
    
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
                                  (1 - actual) * tf.log(1 - obs_), reduction_indices=[1,2,3, 4])
        
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
            
            if e == epochs-1: self.record = {"z":[], "reconstructed":[]}
            
            for i in range(batch_num):
                #Training happens here.
                batch = X.next()
                feed_dict = {self.ops["x"]: batch, self.ops["bn"]: True}
                ops_to_run = [self.ops["reconstructed"], self.ops["z_mean"], self.ops["cost"],\
                              self.ops["kl_loss"], self.ops["rec_loss"], self.ops["train"]]
                reconstruction, z, cost, kld, rec, _= self.sesh.run(ops_to_run, feed_dict)
                
                if e == epochs-1: self.record["z"] = self.record["z"] + [_ for _ in z]
                if e == epochs-1: self.record["reconstructed"] = self.record["reconstructed"] + [_ for _ in reconstruction]
                epoch_cost["kld"].append(np.mean(kld))
                epoch_cost["rec"].append(np.mean(rec))
                epoch_cost["cost"].append(cost)
            
            if valid != None:
                for i in range(val_batch_num):
                    batch = valid.next()
                    feed_dict = {self.ops["x"]: batch[0], self.ops["bn"]: False}
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
        feed_dict = {self.ops["x"]: x, self.ops["bn"]: False}
        return self.sesh.run([self.ops["z_mean"], self.ops["z_logsigma"]], feed_dict=feed_dict)

    # Decode latent examples. Other_wise, draw from N(0,1)
    def decode(self, zs=None):
        feed_dict = dict()
        if zs is not None:
            feed_dict = {self.ops["latent_input"]: zs, self.ops["bn"]: False}
        return self.sesh.run(self.ops["reconstructed_"], feed_dict)