import time
import random
from model import GCNModel
from opt import Optimizer
import numpy as np
import numpy.linalg as LA
import math
import tensorflow as tf
def solve_l1l2(W,lambda1):
    nv=W.shape[1]
    F=W.copy()
    for p in range(nv):
        nw=LA.norm(W[:,p],"fro")
        if nw>lambda1:
            F[:,p]=(nw-lambda1)*W[:,p]/nw
        else:F[:,p]=np.zeros((W[:,p].shape[0],1))
    return F

def PredictScore(train_drug_dis_matrix, drug_matrix, dis_matrix, seed, epochs, emb_dim, dp, lr,  adjdp):
    np.random.seed(seed)
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    adj = constructHNet(train_drug_dis_matrix, drug_matrix, dis_matrix)
    adj = sp.csr_matrix(adj)
    association_nam = train_drug_dis_matrix.sum()
    X = constructNet(train_drug_dis_matrix)
    features = sparse_to_tuple(sp.csr_matrix(X))
    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    adj_orig = train_drug_dis_matrix.copy()
    adj_orig = sparse_to_tuple(sp.csr_matrix(adj_orig))

    adj_norm = preprocess_graph(adj)
    adj_nonzero = adj_norm[1].shape[0]
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'adjdp': tf.placeholder_with_default(0., shape=())
    }
    model = GCNModel(placeholders, num_features, emb_dim,
                     features_nonzero, adj_nonzero, train_drug_dis_matrix.shape[0], name='LAGCN')
    with tf.name_scope('optimizer'):
        opt = Optimizer(
            preds=model.reconstructions,
            labels=tf.reshape(tf.sparse_tensor_to_dense(
                placeholders['adj_orig'], validate_indices=False), [-1]),
            model=model,
            lr=lr, num_u=train_drug_dis_matrix.shape[0], num_v=train_drug_dis_matrix.shape[1], association_nam=association_nam)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        feed_dict = dict()
        feed_dict.update({placeholders['features']: features})
        feed_dict.update({placeholders['adj']: adj_norm})
        feed_dict.update({placeholders['adj_orig']: adj_orig})
        feed_dict.update({placeholders['dropout']: dp})
        feed_dict.update({placeholders['adjdp']: adjdp})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 100 == 0:
            feed_dict.update({placeholders['dropout']: 0})
            feed_dict.update({placeholders['adjdp']: 0})
            res = sess.run(model.reconstructions, feed_dict=feed_dict)
            print("Epoch:", '%04d' % (epoch + 1),
                  "train_loss=", "{:.5f}".format(avg_cost))
    print('Optimization Finished!')
    feed_dict.update({placeholders['dropout']: 0})
    feed_dict.update({placeholders['adjdp']: 0})
    res = sess.run(model.reconstructions, feed_dict=feed_dict)
    sess.close()
    return res

time1 = time.time()
a=np.mat(np.loadtxt('KNN2.txt'))#Define a 541*831 matrix of zeros
AA=a.astype(int)#Convert matrix AA to integer

#Read the known SM-miRNA associations
b=np.loadtxt('known SM-miRNA interaction2.txt')
B=b.astype(int)
SSM=np.mat(np.loadtxt('Integrated Similarity of SM2.txt'))
#Read the integrated similarity of miRNA
SMR=np.mat(np.loadtxt('Integrated similarity of miRNA2.txt'))
#Normalize the SSM
SSM1=np.mat(SSM.copy())
for nn1 in range(39):
    for nn2 in range(39):
        SSM[nn1,nn2]=SSM[nn1,nn2]/(np.sqrt(np.sum(SSM1[nn1,:]))*np.sqrt(np.sum(SSM1[nn2,:])))
#Normalize the SMR
SMR1=np.mat(SMR.copy())
for mm1 in range(286):
    for mm2 in range(286):
        SMR[mm1,mm2]=SMR[mm1,mm2]/(np.sqrt(np.sum(SMR1[mm1,:]))*np.sqrt(np.sum(SMR1[mm2,:])))
#GCN
        miRNA_sim = SMR
        SM_sim = SSM
        SM_miRNA_matrix = A
        epoch = 600
        emb_dim = 64
        lr = 0.00725
        adjdp = 0.6
        dp = 0.4
        simw = 6
        miRNA_matrix = miRNA_sim * simw
        SM_matrix = SM_sim * simw
        train_matrix = np.matrix(SM_miRNA_matrix, copy=True)
        circle_time = 0
        miRNA_len = SM_miRNA_matrix.shape[0]
        SM_len = SM_miRNA_matrix.shape[1]
        SM_miRNA_res = PredictScore(train_matrix, miRNA_matrix, SM_matrix, circle_time, epoch, emb_dim, dp, lr,
                                    adjdp)
        S = np.mat(SM_miRNA_res.reshape(miRNA_len, SM_len))
        print('--------------------------------------')