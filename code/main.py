import numpy as np
from AE import make_AEgraph, trainAE, testAE
import matplotlib.pyplot as plt
from sklearn import datasets, preprocessing, model_selection

seed = 0
np.random.seed(seed)

# load data
data = datasets.load_wine()
X = preprocessing.StandardScaler().fit_transform(data['data']) # z-score
Y = preprocessing.OneHotEncoder(sparse=False).fit_transform(data['target'].reshape(-1, 1)) # one-hot
input_size = X.shape[1]

# split labelled/unlabelled
Xtr, Xte, Ytr, Yte = model_selection.train_test_split(X, Y, test_size=0.2, stratify=Y)
Xval = Xtr

# build first AE graph
g1 = make_AEgraph(ae_layout=[input_size,5,2,5,input_size],
                         lin_dec=False,
                         nonlinearity='maxout', # {relu, sigmoid, tanh, maxout}
                         init='he', # {he, xav}
                         learning_rate=0.01,
                         w_l2=0.01,
                         max_gradient_norm=1.0,
                         seed=seed) 

# build second AE graph
g2 = make_AEgraph(ae_layout=[input_size,5,2,5,input_size],
                         lin_dec=False,
                         nonlinearity='maxout', # {relu, sigmoid, tanh, maxout}
                         init='he', # {he, xav}
                         learning_rate=0.001,
                         w_l2=0.01,
                         max_gradient_norm=1.0,
                         seed=seed) 

# train first AE
trainAE(Xtr,
       Xval,
       batch_size=25,
       num_epochs=300,
       dropout_prob = 0.0,
       save_id='id1',
       input_graph=g1)

# test first AE
te_rec_loss, te_code, te_reconstr = testAE(Xte, save_id='id1')
plt.scatter(te_code[:,0], te_code[:,1], c=np.argmax(Yte,axis=1), cmap="tab10")
plt.show()

# train second AE
trainAE(Xtr,
       Xval,
       batch_size=25,
       num_epochs=2000,
       dropout_prob = 0.0,
       save_id='id2',
       input_graph=g2)

# test second AE
te_rec_loss, te_code, te_reconstr = testAE(Xte, save_id='id2')
plt.scatter(te_code[:,0], te_code[:,1], c=np.argmax(Yte,axis=1), cmap="tab10")
plt.show()

# test a pretrained model
te_rec_loss, te_code, te_reconstr = testAE(Xte, save_id='pretrained')
plt.scatter(te_code[:,0], te_code[:,1], c=np.argmax(Yte,axis=1), cmap="tab10")
plt.show()