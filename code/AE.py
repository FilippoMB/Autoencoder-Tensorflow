import tensorflow as tf
import numpy as np
import time
from tensorflow.contrib.layers import maxout


def fc_layer(input_, in_dim, size, init, w_l2):
    '''
    yield a fully connected layer
    '''
    
    if init == 'he':
        init_type = tf.keras.initializers.he_normal(seed=None)
    elif init == 'xav':
        init_type = tf.contrib.layers.xavier_initializer()
    else:
        raise RuntimeError('Unknown initializer, can be {"he", "xav"}')

    W = tf.get_variable('W', shape=(in_dim, size), 
                        dtype=tf.float32,
                        initializer=init_type, 
                        regularizer=tf.contrib.layers.l2_regularizer(w_l2)
                        )

    b = tf.get_variable('bias', shape=(size,),
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer)

    fc_output = tf.add(tf.matmul(input_, W), b)

    return fc_output


def build_network(input_, net_layout, keep_prob, nonlinearity='relu', init='he', w_l2=0.0001):
    '''
    build a feed-forward network
    '''

    in_dim = net_layout[0]

    for i, neurons in enumerate(net_layout,1):
        with tf.variable_scope('h{}'.format(i)):
            if nonlinearity == 'relu':
                layer_out = fc_layer(input_, in_dim, neurons, init, w_l2)
                layer_out = tf.nn.relu(layer_out)
            elif nonlinearity == 'maxout':
                K = 5
                layer_out = fc_layer(input_, in_dim, neurons*K, init, w_l2)
                layer_out = maxout(layer_out, neurons)
            elif nonlinearity == 'sigmoid':
                layer_out = fc_layer(input_, in_dim, neurons, init, w_l2)
                layer_out = tf.nn.sigmoid(layer_out)
            elif nonlinearity == 'tanh':
                layer_out = fc_layer(input_, in_dim, neurons, init, w_l2)
                layer_out = tf.nn.tanh(layer_out)
            elif nonlinearity == 'linear':
                layer_out = fc_layer(input_, in_dim, neurons, init, w_l2)
            else:
                raise RuntimeError('Unknown nonlinearity, can be {"relu", "maxout", "sigmoid", "tanh", "linear"}')

            layer_out = tf.nn.dropout(layer_out, keep_prob=keep_prob)

            input_ = layer_out
            in_dim = neurons

    return layer_out


def next_batch(X, batch_size=1, shuffle=True):
    '''
    Generator that supplies mini batches
    '''
    
    n_data = len(X)

    if shuffle:
        idx = np.random.permutation(n_data)
    else:
        idx = range(n_data)
    X = X[idx,:]

    n_batches = n_data//batch_size
    for i in range(n_batches):
        X_batch = X[i*batch_size:(i+1)*batch_size,:]

        yield X_batch


class customAE():

    def make_graph(self,
                   ae_layout=[2,1,2], # must be list of odd length (usually symmetric). First and last numbers are the input size. The central is code size.
                   lin_dec=True,
                   nonlinearity='relu',
                   init='he',
                   learning_rate=0.001,
                   w_l2=0.001,
                   max_gradient_norm=1.0,
                   seed=None,
                   model_id=None,
                   ):
        
        # initialize computational graph
        g = tf.Graph()
        with g.as_default():
            
            tf.set_random_seed(seed) # only affects default graph
        
            # placeholders
            self.encoder_inputs = tf.placeholder(shape=(None,ae_layout[0]), dtype=tf.float32, name='encoder_inputs')
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
                    
            if len(ae_layout)%2 == 0:
                raise RuntimeError('Bad AE layout!')
            enc_size = len(ae_layout)//2
            enc_layout = ae_layout[:enc_size+1]
            dec_layout = ae_layout[enc_size:]
    
            # encoder
            with tf.variable_scope('encoder'):
                self.codes = build_network(self.encoder_inputs,
                                           enc_layout,
                                           self.keep_prob,
                                           nonlinearity,
                                           init,
                                           w_l2)
    
    
            # decoder
            if lin_dec:
                dec_nonlin = 'linear'
            else:
                dec_nonlin = nonlinearity
            with tf.variable_scope('decoder'):
                self.dec_out = build_network(self.codes,
                                             dec_layout,
                                             self.keep_prob,
                                             dec_nonlin,
                                             init,
                                             w_l2)
    
           
            # reconstruction loss
            self.reconstruct_loss = tf.losses.mean_squared_error(labels=self.dec_out, predictions=self.encoder_inputs)
    
            # L2 loss
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
            # tot loss
            self.tot_loss = self.reconstruct_loss + self.reg_loss 
    
            # Calculate and clip gradients
            parameters = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gradients = tf.gradients(self.tot_loss, parameters)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.update_step = optimizer.apply_gradients(zip(clipped_gradients, parameters))
    
            # save the graph        
            if model_id is not None:  
                print('export graph')
                # put in a collection tensors and operations to call later 
                g.add_to_collection('AE_collection', self.encoder_inputs) #0
                g.add_to_collection('AE_collection', self.keep_prob) #1
                g.add_to_collection('AE_collection', self.codes) #2
                g.add_to_collection('AE_collection', self.dec_out) #3
                g.add_to_collection('AE_collection', self.update_step) #4
                g.add_to_collection('AE_collection', self.reconstruct_loss) #5
                g.add_to_collection('AE_collection', self.reg_loss) #6 
                g.add_to_collection('AE_collection', self.tot_loss) #7
                
                saver = tf.train.Saver()                                    
                saver.export_meta_graph('../models/AE_model_'+model_id+'.meta')
                            
        print('graph building is done')
        
        return g


    def fit(self, 
            X, 
            Xval, 
            batch_size=25, 
            num_epochs=1000, 
            dropout_prob=0.0,
            model_id=None,
            input_graph=None):
        
        if input_graph is None:
            raise RuntimeError('computational graph not provided')
                
        with tf.Session(graph=input_graph) as sess:
                                     
            # restore a meta-graph 
            if model_id is not None:       
                print('restoring graph')    
                model_name = '../models/AE_model_'+model_id
                saver = tf.train.import_meta_graph(model_name+'.meta', clear_devices=True)                
                self.encoder_inputs = input_graph.get_collection('AE_collection')[0]
                self.keep_prob = input_graph.get_collection('AE_collection')[1]
                self.update_step = input_graph.get_collection('AE_collection')[4]
                self.reconstruct_loss = input_graph.get_collection('AE_collection')[5]
                self.reg_loss = input_graph.get_collection('AE_collection')[6]
                self.tot_loss = input_graph.get_collection('AE_collection')[7]
                
            else:
                model_name = '../models/AE_model_default'
                saver = tf.train.Saver()          
                                    
            # initialize trainable variables 
            print('init vars')
            sess.run(tf.global_variables_initializer())

            # initialize training stuff
            time_tr_start = time.time()
            loss_track = []
            min_val_loss = np.infty
            
            print('training start')            
            try:
                for t in range(num_epochs):
                    for X_batch in next_batch(X, batch_size, True):
                        fdtr = {self.encoder_inputs: X_batch,
                                self.keep_prob: 1.0 - dropout_prob}

                        _, train_loss = sess.run([self.update_step, self.tot_loss], fdtr)                        
                        loss_track.append(train_loss)

                    # check training progress on the validation set
                    if t % 100 == 0:
                        fdvs = {self.encoder_inputs: Xval, 
                                self.keep_prob: 1.0}
                        
                        (val_tot_loss,
                         val_rec_loss,
                         val_reg_loss) = sess.run([  self.tot_loss,
                                                 self.reconstruct_loss,
                                                 self.reg_loss], fdvs)

                        print("totL: %.3f, recL: %.3f, reg_loss: %.3f"%(val_tot_loss, val_rec_loss, val_reg_loss))

                        # Save model yielding the lowest loss on validation
                        if val_tot_loss < min_val_loss:
                            min_val_loss = val_tot_loss
                            saver.save(sess, model_name)

            except KeyboardInterrupt:
                print('training interrupted')

        ttime = (time.time()-time_tr_start)/60
        print("tot time:{}".format(ttime))
        
        return loss_track


    def transform(self, 
                  Xte, 
                  model_id=None,
                  input_graph=None):

        if input_graph is None:
            raise RuntimeError('computational graph not provided')
                
        with tf.Session(graph=input_graph) as sess:
            
            # restore a meta-graph 
            if model_id is not None:  
                print('restoring graph')               
                model_name = '../models/AE_model_'+model_id
                saver = tf.train.import_meta_graph(model_name+'.meta', clear_devices=True)                
                self.encoder_inputs = tf.get_collection('AE_collection')[0]
                self.keep_prob = tf.get_collection('AE_collection')[1]
                self.codes = tf.get_collection('AE_collection')[2]
                self.dec_out = tf.get_collection('AE_collection')[3]
                self.reconstruct_loss = tf.get_collection('AE_collection')[5]
                
            else:
                model_name = '../models/AE_model_default'
                saver = tf.train.Saver()
                
            print('restoring weights')                    
            saver.restore(sess, model_name)
                
            # evaluate model on the test set
            fdte = {self.encoder_inputs: Xte, self.keep_prob: 1.0}
            te_rec_loss, te_code, te_reconstr = sess.run([self.reconstruct_loss, self.codes, self.dec_out], fdte)        
        
        return te_rec_loss, te_code, te_reconstr
