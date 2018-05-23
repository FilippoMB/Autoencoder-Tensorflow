import numpy as np
from AE import customAE
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

# create AE object
AE = customAE()

# ----- NO MODEL SAVING ------

# build graph
g = AE.make_graph(ae_layout=[input_size,5,2,5,input_size],
                         lin_dec=False,
                         nonlinearity='maxout', # {relu, sigmoid, tanh, maxout}
                         init='he', # {he, xav}
                         learning_rate=0.001,
                         w_l2=0.01,
                         max_gradient_norm=1.0,
                         seed=seed,
                         model_id=None) 

# train AE
AE.fit(Xtr,
       Xval,
       batch_size=25,
       num_epochs=1000,
       dropout_prob = 0.0,
       model_id=None,
       input_graph=g)

# test AE
te_rec_loss, te_code, te_reconstr = AE.transform(Xte, model_id=None, input_graph=g)
plt.scatter(te_code[:,0], te_code[:,1], c=np.argmax(Yte,axis=1), cmap="tab10")
plt.show()


# ----- HANDLE MULTIPLE MODELS AT ONCE -----

# build first AE
g1 = AE.make_graph(ae_layout=[input_size,10,2,10,input_size],
                             lin_dec=True,
                             nonlinearity='relu', # {relu, sigmoid, tanh, maxout}
                             init='xav', # {he, xav}
                             learning_rate=0.001,
                             w_l2=0.001,
                             max_gradient_norm=1.0,
                             seed=seed,
                             model_id='id1') 

# build second AE
g2 = AE.make_graph(ae_layout=[input_size,2,input_size],
                             lin_dec=False,
                             nonlinearity='relu', # {relu, sigmoid, tanh, maxout}
                             init='he', # {he, xav}
                             learning_rate=0.001,
                             w_l2=0.1,
                             max_gradient_norm=1.0,
                             seed=seed,
                             model_id='id2') 

# train first AE
AE.fit(Xtr,
       Xval,
       batch_size=25,
       num_epochs=1000,
       dropout_prob = 0.0,
       model_id='id1',
       input_graph=g1)


# train second AE
AE.fit(Xtr,
       Xval,
       batch_size=25,
       num_epochs=1000,
       dropout_prob = 0.0,
       model_id='id2',
       input_graph=g2)
    
# test first AE
te_rec_loss, te_code, te_reconstr = AE.transform(Xte, model_id='id1', input_graph=g1)
plt.scatter(te_code[:,0], te_code[:,1], c=np.argmax(Yte,axis=1), cmap="tab10")
plt.show()

# test second AE
te_rec_loss, te_code, te_reconstr = AE.transform(Xte, model_id='id2',input_graph=g2)
plt.scatter(te_code[:,0], te_code[:,1], c=np.argmax(Yte,axis=1), cmap="Set1")
plt.show()
