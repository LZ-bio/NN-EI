#!/usr/bin/env python
# coding: utf-8

# In[5]:

# # Model construction

# In[30]:

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--project", type=str, default="PBRCA", help="Cancer project")
    parser.add_argument(
        "--alpha", type=int, default = 8000, help="The number of chose genes"
    )
    parser.add_argument(
        "--garma", type=float, default = 0.25, help="The p-value in IORA"
    )
    parser.add_argument(
        "--delta", type=float, default = 0.5, help="The number of permutations"
    )
    parser.add_argument(
        "--dr", type=float, default = 0.5, help="The dropout ratio of"
    )
    parser.add_argument(
        "--lr", type=float, default = 0.0005, help="The learning ratio"
    )
    parser.add_argument(
        "--sel", type=float, default = 0.9, help="The selta of focal loss"
    )
    parser.add_argument(
        "--bel", type=float, default = 3, help="The belta of focal loss"
    )
    parser.add_argument(
        "--n2", type=int, default = 36, help="The number of neurons in IGSEA-driven modules"
    )
    parser.add_argument(
        "--n3", type=int, default = 36, help="The number of neurons in the hierarchy module"
    )
    parser.add_argument(
        "--n4", type=int, default = 8, help="The number of neurons in the first fully connected layers"
    )
    parser.add_argument(
        "--b1", type=int, default = 1, help="The weight of loss function(class1)"
    )
    parser.add_argument(
        "--b2", type=int, default = 4, help="The weight of loss function(class2)"
    )
    parser.add_argument(
        "--b3", type=int, default = 10, help="The weight of loss function(class2)"
    )

    args = parser.parse_args()
    return args

args = parse_args()

cancer = args.project
chosefea = args.alpha
chosefea1 = chosefea
threa = args.garma
threa1 = args.delta
dr = args.dr
lr = args.lr
sel = args.sel
bel = args.bel

n2 = args.n2
n3 = args.n3
n4 = args.n4

ba1 = args.b1
ba2 = args.b2
ba3 = args.b3

lossw = [ba1,ba2,ba1,ba2,ba3]
n_hidden_layers_PtH = [n2]


#n_hidden_layers_PtH = [36,8,8] 
#n_hidden_layers_PtH = [36,8,8] 


from keras.layers import Dense, Dropout, Activation, BatchNormalization, multiply
from keras.regularizers import l2,Regularizer
from keras import Input
from keras.engine import Model,Layer
from keras import backend as K
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2,f_regression
from sklearn.model_selection import StratifiedKFold,RepeatedKFold
import scipy.stats as ss
import keras
import numpy as np
from keras import regularizers
# from keras import initializations
from keras.initializers import glorot_uniform, Initializer
from keras.layers import activations, initializers, constraints,Reshape
# our layer will take input shape (nb_samples, 1)
from keras.regularizers import Regularizer
import tensorflow as tf
import re
from Mnes import mgsea,imputation,createNetwork13,createNetwork14,cal_mgsea,createNetwork4,createNetwork5
from scipy.spatial.distance import pdist, squareform
from coms import focal_loss,f1


# In[32]:


class M_Nets(Layer):   
    def __init__(self, units, activation=None,
                 use_bias=True,
                 kernel_initializer='lecun_uniform',
                 bias_initializer='zeros',
                 W_regularizer=None,
                 b_regularizer=None,
                 **kwargs):
        self.units = units
        self.activation = activation
        self.activation_fn = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularize = regularizers.get(b_regularizer)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(W_regularizer)
        
        super(M_Nets, self).__init__(**kwargs)


    def build(self, input_shape):  

        input_dimension = input_shape[1]
        self.kernel_shape = (input_dimension, self.units)  
        self.n_inputs_per_node = input_dimension / self.units

        rows = np.arange(input_dimension) 
        cols = np.arange(self.units)    
        cols = np.repeat(cols, self.n_inputs_per_node) 
        self.nonzero_ind = np.column_stack((rows, cols)) 

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_dimension,),  
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      trainable=True)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        trainable=True
                                        
                                       )
        else:
            self.bias = None

        super(M_Nets, self).build(input_shape)  

    def call(self, x, mask=None):
        
        n_features = x.shape[1]


        kernel = K.reshape(self.kernel, (1, n_features))
        mult = x * kernel
        mult = K.reshape(mult, (-1, int(self.n_inputs_per_node)))
        mult = K.sum(mult, axis=1)
        output = K.reshape(mult, (-1, self.units))

        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation_fn is not None:
            output = self.activation_fn(output)
        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'kernel_initializer' : self.kernel_initializer,
            'bias_initializer' : self.bias_initializer,
            'use_bias': self.use_bias
        }
        base_config = super(M_Nets, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)


# In[33]:


class Nets(Layer):
    def __init__(self, units, mapp=None, nonzero_ind=None, kernel_initializer='glorot_uniform',
                 activation='elu', use_bias=True,bias_initializer='glorot_uniform', bias_regularizer=None,
                 bias_constraint=None,**kwargs):
        
        self.units = units
        self.activation = activation
        self.mapp = mapp
        self.nonzero_ind = nonzero_ind
        self.use_bias = use_bias
        
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.l2(0.0001)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.l2(0.0001)
        self.activation_fn = activations.get(activation)
        super(Nets, self).__init__(**kwargs)

        
    def build(self, input_shape):
        
        input_dim = input_shape[1]
   

        if not self.mapp is None:
            self.mapp = self.mapp.astype(np.float32)

   
        if self.nonzero_ind is None:
            nonzero_ind = np.array(np.nonzero(self.mapp)).T
            self.nonzero_ind = nonzero_ind

        self.kernel_shape = (input_dim, self.units)
        

        nonzero_count = self.nonzero_ind.shape[0]  


        self.kernel_vector = self.add_weight(name='kernel_vector',
                                             shape=(nonzero_count,),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer
                                        )
        else:
            self.bias = None

        super(Nets, self).build(input_shape)  
      

    def call(self, inputs):
        
        
        temp_t = tf.scatter_nd(tf.constant(self.nonzero_ind, tf.int32), self.kernel_vector,
                           tf.constant(list(self.kernel_shape)))
    
        output = K.dot(inputs, temp_t)
        
    
        if self.use_bias:
            output = K.bias_add(output, self.bias)
            
        if self.activation_fn is not None:
            output = self.activation_fn(output)

        return output

    def get_config(self):
        config = {
            'units': self.units,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'nonzero_ind': np.array(self.nonzero_ind),
          
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),


            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            #'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),

            #'kernel_initializer': initializers.serialize(self.kernel_initializer),
            #'W_regularizer': regularizers.serialize(self.kernel_regularizer),

        }
        base_config = super(Nets, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
      
        return (input_shape[0], self.units)



#no_mask and no_attention
def create_models_pheno17(Omics_data):
    #mrna
    M_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_m')

    #m0 = Nets(Get_Node_relation_mrna[0].shape[1],mapp =Get_Node_relation_mrna[0].values, name = 'm0')(M_inputs)
    m0 = keras.layers.Dense(Get_Node_relation_mrna[0].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.0001),use_bias=True,bias_regularizer=regularizers.l2(0.0001))(M_inputs)
    drop0 = keras.layers.Dropout(dr)(m0)
    drop0 = BatchNormalization()(drop0)
    
    #m1 = keras.layers.Dense(Get_Node_relation_mrna[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.0001),use_bias=True,bias_regularizer=regularizers.l2(0.0001))(drop0)
    m1 = Nets(Get_Node_relation_mrna[1].shape[1],mapp =Get_Node_relation_mrna[1].values, name = 'm00')(drop0)
    drop_m1 = keras.layers.Dropout(0.5)(m1)
    drop_m1 = BatchNormalization()(drop_m1)

    m2 = Nets(Get_Node_relation_mrna[2].shape[1],mapp =Get_Node_relation_mrna[2].values, name = 'm10')(drop_m1)
    drop_m2 = keras.layers.Dropout(0.5)(m2)
    drop_m2 = BatchNormalization()(drop_m2)

    m3 = Nets(Get_Node_relation_mrna[3].shape[1],mapp =Get_Node_relation_mrna[3].values, name = 'm20')(drop_m2)
    drop_m3 = keras.layers.Dropout(0.5)(m3)
    drop_m3 = BatchNormalization()(drop_m3)

    output1 = keras.layers.Dense(1,activation='sigmoid')(drop_m1)

    output2 = keras.layers.Dense(1,activation='sigmoid')(drop_m3)
    
    #cnv_amp
    h_inputs = Input(shape=(Omics_data.shape[1],), dtype='float32',name= 'inputs_h')

    h0 = keras.layers.Dense(Get_Node_relation_amp[0].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.0001),use_bias=True,bias_regularizer=regularizers.l2(0.0001))(h_inputs)
    #h0 = Nets(Get_Node_relation_amp[0].shape[1],mapp =Get_Node_relation_amp[0].values, name = 'h0')(h_inputs)
    drop_h0 = keras.layers.Dropout(dr)(h0)
    drop_h0 = BatchNormalization()(drop_h0)

    #h1 = keras.layers.Dense(Get_Node_relation_amp[1].shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.0001),use_bias=True,bias_regularizer=regularizers.l2(0.0001))(drop_h0)
    h1 = Nets(Get_Node_relation_amp[1].shape[1],mapp =Get_Node_relation_amp[1].values, name = 'm01')(drop_h0)
    drop_h1 = keras.layers.Dropout(0.5)(h1)
    drop_h1 = BatchNormalization()(drop_h1)

    h2 = Nets(Get_Node_relation_amp[2].shape[1],mapp =Get_Node_relation_amp[2].values, name = 'm11')(drop_h1)
    drop_h2 = keras.layers.Dropout(0.5)(h2)
    drop_h2 = BatchNormalization()(drop_h2)

    h3 = Nets(Get_Node_relation_amp[3].shape[1],mapp =Get_Node_relation_amp[3].values, name = 'm21')(drop_h2)
    drop_h3 = keras.layers.Dropout(0.5)(h3)
    drop_h3 = BatchNormalization()(drop_h3)

    output3 = keras.layers.Dense(1,activation='sigmoid')(drop_h1)

    output4 = keras.layers.Dense(1,activation='sigmoid')(drop_h3)

    a1 = keras.layers.concatenate([drop_m3,drop_h3])
    a1 = BatchNormalization()(a1)
    
    #a2 = keras.layers.Dense(Get_Node_relation_next.shape[1],activation='elu',kernel_regularizer=regularizers.l2(0.0001),use_bias=True,bias_regularizer=regularizers.l2(0.0001))(a1)
    a2 = Nets(Get_Node_relation_next.shape[1],mapp =Get_Node_relation_next.values,name = 'a1')(a1)
    drop_a2 = keras.layers.Dropout(0.5)(a2)
    drop_a2 = BatchNormalization()(drop_a2)

    a5 = keras.layers.Dense(n3,activation='elu',kernel_regularizer=regularizers.l2(0.0001),use_bias=True,bias_regularizer=regularizers.l2(0.0001))(drop_a2)
    a5 = keras.layers.Dropout(0.5)(a5)
    a5 = BatchNormalization()(a5)
    a6 = keras.layers.Dense(n4,activation='elu',kernel_regularizer=regularizers.l2(0.0001),use_bias=True,bias_regularizer=regularizers.l2(0.0001))(a5)
    a6 = BatchNormalization()(a6)

    Output = keras.layers.Dense(1,activation='sigmoid')(a6)
    model = Model(inputs=[M_inputs,h_inputs], outputs=[output1,output2,output3,output4,Output])

    model.summary()

    #opt = keras.optimizers.Adamax(lr = 0.002)
    opt = keras.optimizers.Adam(lr = lr) #,decay=-0.0001 ,decay=0.0001
    model.compile(optimizer=opt,
                  #loss=['binary_crossentropy']*5,
                  loss=[focal_loss(alpha=sel,gamma=bel)]*5, #PRAD(0.7,2.5)
                  loss_weights = lossw,   #[1,1,1,1,40,54,20,2000] #[1,4,1,4,10]
                  metrics=['acc'])
    return model



def combine(x_list, y_list, rows_list, cols_list, data_type_list, combine_type, use_coding_genes_only=False):    
    cols_list_set = [set(list(c)) for c in cols_list]      
    print('cols_list_set',len(cols_list_set))
    if combine_type == 'intersection':    
        cols = set.intersection(*cols_list_set)
    else:
        cols = set.union(*cols_list_set)     
    print('intersection_cols',len(cols))
    if use_coding_genes_only: #true
        coding_genes_df = pd.read_csv('./Data/protein-coding_gene_with_coordinate_minimal.txt', sep='\t', header=None)
        coding_genes_df.columns = ['chr', 'start', 'end', 'name']
        coding_genes = set(coding_genes_df['name'].unique())     
        cols = cols.intersection(coding_genes)  
        print('protein-coding_genes',len(coding_genes))   
    print('finally_cols',len(cols))   
    all_cols = list(cols)
    all_cols_df = pd.DataFrame(index=all_cols) 
    df_list = []
    for x, y, r, c in zip(x_list, y_list, rows_list, cols_list):
        df = pd.DataFrame(x, columns=c, index=r)
        df = df.T.join(all_cols_df, how='right')  
        df = df.T
        df = df.fillna(0)
        df_list.append(df)
    all_data = pd.concat(df_list, keys=data_type_list, join='inner', axis=1, )    
    all_data = all_data.swaplevel(i=0, j=1, axis=1)
    order = all_data.columns.levels[0] 
    all_data = all_data.reindex(columns=order, level=0)  
    x = all_data
    reordering_df = pd.DataFrame(index=all_data.index)  
    y = reordering_df.join(y, how='left')   
    y = y.values   
    cols = all_data.columns   
    rows = all_data.index      
    print(
        'After combining, loaded data %d samples, %d variables, %d responses ' % (x.shape[0], x.shape[1], y.shape[0]))
    return x, y, rows, cols

# In[44]:


import sklearn
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
   

from sklearn.metrics import precision_recall_curve
def evaluates(y_test, y_pred):
    
    auc = metrics.roc_auc_score(y_test,y_pred)
    
    aupr = average_precision_score(y_test, y_pred)
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)    
    auprc  = metrics.auc(recall, precision)
    
    pp = [1 if index>=0.5  else 0 for index in  y_pred ]
    
    pre = metrics.precision_score(y_test,pp)
    
    f1 = metrics.f1_score(y_test,pp)
    
    rec = metrics.recall_score(y_test,pp)
    
    acc = metrics.accuracy_score(y_test,pp)
    
    print(confusion_matrix(y_test,pp))
    
    return pre,acc,rec,f1,auc,aupr,auprc


# In[37]:


from deepexplain.model_utils import get_layers, get_coef_importance

def get_coef_importances(model, X_train, y_train, target=-1, feature_importance='deepexplain_grad*input'):

    coef_ = get_coef_importance(model, X_train, y_train, target, feature_importance, detailed=False)
    return coef_


# In[38]:


from keras.callbacks import LearningRateScheduler
def myScheduler(epoch):

    if epoch % 150 == 0 and epoch != 0:

        lr = K.get_value(model.optimizer.lr)

        K.set_value(model.optimizer.lr, lr * 0.5)
    return K.get_value(model.optimizer.lr)
 
myReduce_lr = LearningRateScheduler(myScheduler)


import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import copy
import os
import random
import itertools
import logging
random.seed(555)  


def get_nodes_at_level(net, distance):
    nodes = set(nx.ego_graph(net, 'root', radius=distance))
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))
    return list(nodes)

def get_nodes(net,num):
    net_nodes = [] 
    for i in range(1,num+1):
        net_nodes.append(get_nodes_at_level(net,i))    
    return net_nodes

def add_node(net,net_nodes):        
    for i in range(len(net_nodes)-2,-1,-1):
        data_temp = copy.deepcopy(net_nodes[i])
        for n in net_nodes[i]:
            nexts = net.successors(n)         
            temp = [ nex  for nex in nexts ] 
            if len(temp)==0:
                data_temp.remove(n)  # If the node of the current layer has no successor node, remove the node
            elif len(set(temp).intersection(set(net_nodes[i+1])))==0:   #if the subsequent node of the node of the current layer is not on the next layer, delete the node
                data_temp.remove(n)
            else:
                continue
        net_nodes[i] = data_temp
    return net_nodes


def get_note_relation(net_nodes):
    node_mat = []
    for i in range(len(net_nodes)-1):
        dicts = {}
        for n in net_nodes[i]:
            nexts = net.successors(n)  
            x = [ nex   for nex in nexts if nex in net_nodes[i+1] ]
            dicts[n] = x
        mat = np.zeros((len(net_nodes[i]), len(net_nodes[i+1]))) 
        for p, gs in dicts.items():     
            g_inds = [net_nodes[i+1].index(g) for g in gs]
            p_ind = net_nodes[i].index(p)
            mat[p_ind, g_inds] = 1
        df = pd.DataFrame(mat, index=net_nodes[i], columns=net_nodes[i+1])
        node_mat.append(df.T)
    return node_mat

def sigmoid(Z):
    return 1/(1+np.exp(-Z))

def add_edges(netx1,name):
    node1 = [node for node, indeg in netx1.in_degree() if indeg == 0]
    node2 = [node for node, indeg in netx1.out_degree() if indeg == 0]
    nodes = list(set(node1).intersection(set(node2)))
    edges = [(node,name) for node in nodes]
    return edges

def get_layers(network_genes,selected_genes,chosedPathways,netx1,netx2,netx3):
    layers = []
    nodes = network_genes
    dic = {}
    for n  in nodes:
        next = netx1.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    dic = {}
    nodes = selected_genes
    for n  in nodes:
        next = netx2.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    dic = {}
    nodes = chosedPathways
    for n  in nodes:
        next = netx3.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    return layers


def get_map_from_layer(layer_dict):
    genes = list(layer_dict.keys())
    print ('genes', len(genes))
    pathways = list(itertools.chain.from_iterable(layer_dict.values()))
    pathways = list(np.unique(pathways))
    print ('pathways', len(pathways))
    #print(pathways)
    n_pathways = len(pathways)
    n_genes = len(genes)
    mat = np.zeros((n_genes, n_pathways))
    for g, ps in layer_dict.items():
        p_inds = [pathways.index(p) for p in ps]
        g_ind = genes.index(g)
        mat[g_ind, p_inds] = 1

    df = pd.DataFrame(mat, index=genes, columns=pathways)
    return df

def get_layer_maps(genes,layers,names):
    PK_layers = layers
    filtering_index = genes
    maps = []
    for i, layer in enumerate(PK_layers):
        print ('layer #', i)
        mapp = get_map_from_layer(layer)
        #print ('filtered_map', mapp.index)
        filter_df = pd.DataFrame(index=filtering_index)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        #print(filtered_map)
        #if i ==0:
        #    filtered_map = filtered_map[list(filtering_index)]
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        print ('filtered_map', filter_df.shape)
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        filtered_map = filtered_map.fillna(0)
        print ('filtered_map', filter_df.shape)
        if i==2:
            filtered_map = filtered_map[names]
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps


def get_layers_first(network_genes,chosedgenes,chosedPathways,netx1,netx2,netx3,netx4):
    layers = []
    nodes = network_genes
    dic = {}
    for n  in nodes:
        next = netx1.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    dic = {}
    nodes = chosedgenes
    for n  in nodes:
        next = netx2.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    dic = {}
    nodes = chosedgenes
    for n  in nodes:
        next = netx3.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    dic = {}
    nodes = chosedPathways
    for n  in nodes:
        next = netx4.successors(n)
        dic[n] = [nex for nex in next]
    layers.append(dic)
    return layers


def get_layer_maps_first(genes,layers,names):
    PK_layers = layers
    filtering_index = genes
    maps = []
    for i, layer in enumerate(PK_layers):
        print ('layer #', i)
        mapp = get_map_from_layer(layer)
        #print ('filtered_map', mapp.index)
        filter_df = pd.DataFrame(index=filtering_index)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        #print(filtered_map)
        #if i ==0:
        #    filtered_map = filtered_map[list(filtering_index)]
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        print ('filtered_map', filter_df.shape)
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        filtered_map = filtered_map.fillna(0)
        print ('filtered_map', filter_df.shape)
        if i==3:
            filtered_map = filtered_map[names]
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps

def get_layers_map_sec(subname,netx1,names):
    nodes = subname
    dic = {}
    for n  in nodes:
        next = netx1.successors(n)
        dic[n] = [nex for nex in next]
    filtering_index = subname
    mapp = get_map_from_layer(dic)
    print
    filter_df = pd.DataFrame(index=filtering_index)
    filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
    filtered_map = filtered_map.fillna(0)
    filtered_map = filtered_map[names]
    return filtered_map

def load_data_dict(filename):
    data_dict_list = []
    dict = {}
    with open( filename) as gmt:
        data_list = gmt.readlines()
        # print data_list[0]
        for row in data_list:
            genes = row.split('\t')
            genes = [ i.replace('\n','') for i in genes]
            dict[genes[2]] = genes[3:]
    return dict
# # pathways process and network generation



# In[6]:
kfscore = []
#cancer = 'BRCAB'
#cancer1 = 'BRCA'
#cancer = 'PPRAD'  #('PBRCA','BRCA')
#cancer1 = 'PRAD'

omics_num = 4
n_knn = 11

# loading mrna data
file = "./Data/cc/" + cancer + '_mut_matrix.csv'
#mrna_data = pd.read_csv("./Data/TCGA-LUAD.varscan2_snv.csv",index_col = 0)
mrna_data = pd.read_csv(file,index_col = 0)

file = "./Data/cc/response_paper_" + cancer + '.csv'
response  = pd.read_csv(file,index_col=0)
#response  = pd.read_csv('./Data/response_paper.csv',index_col=0)

#introduce prior knowledges
file = "./Data/cc/prior_mrna_zscores_" + cancer + '_1.txt'
mrna_prior = pd.read_csv(file, sep='\t',index_col=0)
mrna_prior =mrna_prior.transpose()

file = "./Data/cc/prior_clinical_" + cancer + '_1.csv'
clin_prior  = pd.read_csv(file,index_col=0)


#read network genes
network_genes = pd.read_csv('./Data/genes/HN_genes.csv', sep='\t',dtype=str)
#network_genes = pd.read_csv('./Data/genes/HN_genes_0.1.csv', sep='\t',dtype=str)


#ead gene-gene relationships 
network_edges = pd.read_csv('./Data/networks/HumanNet.txt', sep='\t',names=['start','end'],dtype=str) 
#network_edges = pd.read_csv('./Data/networks/HumanNet_0.1.txt', sep='\t',names=['start','end'],dtype=str)
network_edges_copy = copy.deepcopy(network_edges)
network_edges_copy.columns = ['end','start']
network_edges_copy = network_edges_copy[['start','end']]
network_edges = pd.concat([network_edges, network_edges_copy], ignore_index=True)
network_edges = network_edges.drop_duplicates()

#Disrupted data set
response = response.sample(frac=1)
mrna_data = mrna_data.sample(frac=1)

#split copy number variation data
import copy

# In[16]:
print(np.array(mrna_data.values.nonzero()).shape)

#
#print(cnv_amp)
mrna_data = mrna_data.loc[response.index]
mrna_data_genes = mrna_data.columns
mrna_data_genenum = len(mrna_data_genes)

re_mrna_data = copy.deepcopy(mrna_data)
re_response = copy.deepcopy(response)

# Read gene-pathway annotation relationships
pathway_genes = pd.read_csv('./Data/pathways/PK/PU.txt', sep='\t',names=['gene','group'],dtype=str)
pathways = list(set(pathway_genes['group']))
pathway_num = len(pathways)
pathway_genes_num = {}
pathway_mrna = {}
pathway_amp = {}
pathway_del = {}
pathway_dmeth = {}
for h in range(pathway_num):
    pathway = pathways[h]
    aa = []
    aa.append(pathway)
    pathway_gene = pathway_genes[pathway_genes['group'].isin(aa)]
    genes = pathway_gene['gene']
    genes_inter_mrna = list(set(genes).intersection(set(mrna_data_genes)))
    pathway_genes_num[pathway] = len(genes_inter_mrna)
    pathway_mrna[pathway] = genes_inter_mrna

tol_snv = mrna_prior.join(clin_prior,how='inner')
model = SelectKBest(f_regression, k=chosefea)
#print(tol_snv.values[:,-1])
temp = tol_snv.values[:,0:-1]
temp = np.nan_to_num(temp, nan=0.0)
x_data1 = model.fit_transform(temp,tol_snv.values[:,-1])
#x_data1 = model.fit_transform(tol_snv.values[:,4300:4632],tol_snv.values[:,-1])
fea = model.get_support()
mrna_prior_sub = mrna_prior.loc[:,fea]
mrna_prior_chosegenes = list(mrna_prior_sub.columns)

mrna_pvalues = []
for h in range(pathway_num):
    pathway = pathways[h]
    genes_inter_mrna = pathway_mrna[pathway]
    record_num = pathway_genes_num[pathway]
    pathway_inter_gene =  set(mrna_prior_chosegenes).intersection(set(genes_inter_mrna))
    a1 =record_num
    a2 = len(pathway_inter_gene)
    a3 = a1 - a2
    a4 = chosefea - a2
    a5 = mrna_data_genenum - a1 - a4
    table = np.array([[a2, a3],[a4, a5]])
    #odds_ratio, p_value = ss.fisher_exact(table,alternative='greater')
    odds_ratio, p_value = ss.fisher_exact(table)
    mrna_pvalues.append(p_value)
        
mrna_prior_pathways =list(np.array(pathways)[np.array(mrna_pvalues)<=threa])
print('The number of prior pathway: {}'.format(len(mrna_prior_pathways))) 
cnv_amp_chosegenes = copy.deepcopy(mrna_prior_chosegenes)
amp_pathways = copy.deepcopy(mrna_prior_pathways)

pset =  np.array(mrna_pvalues)[np.array(mrna_pvalues)<=threa]
ora_pset = {}
ora_pset['names'] = amp_pathways
ora_pset['pvalues'] = pset
data = pd.DataFrame(ora_pset)
directory_path = './Data/coef/{}'.format(cancer)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

data.to_csv('./Data/coef/{}/prior_pathway_pvalues.csv'.format(cancer),index=False,encoding='UTF-8')


pathway_network = pd.read_csv('./Data/pathways/PK/pathway_network.txt', sep='\t',names=['start','end'],dtype=str)
#network_edges = pd.read_csv('./Data/networks/HumanNet_0.1.txt', sep='\t',names=['start','end'],dtype=str)
pathway_network_copy = copy.deepcopy(pathway_network)
pathway_network_copy.columns = ['end','start']
pathway_network_copy = pathway_network_copy[['start','end']]
pathway_network = pd.concat([pathway_network, pathway_network_copy], ignore_index=True)
df = {}
df['start'] = pathways
df['end'] = pathways
df = pd.DataFrame(df)
pathway_network = pd.concat([pathway_network, df], ignore_index=True)
pathway_network = pathway_network.drop_duplicates()

x11 = mrna_data.values
y1 = response['response'].values
y1 = y1.reshape(-1)
istrain = True
num_runs = 1
Save_Res = []
Save_Res1 = []
Save_Res2 = []

if istrain:
    is_flag = 1
else:
    is_flag = 0


for rs in range(0,num_runs):
    kfscore = []

    p = 0
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=rs+1)
    #skf = RepeatedKFold(n_splits=5,n_repeats=2,shuffle=True,random_state=rs+1)
    for train_index, test_index in skf.split(x11, y1):
        #***
        mrna_data = copy.deepcopy(re_mrna_data)
        response = copy.deepcopy(re_response)
        if istrain:
            skf1 = StratifiedKFold(n_splits=5,shuffle=True,random_state=rs+2)
            y12 = y1[train_index]
            for train_index1, test_index1 in skf1.split(train_index, y12):
                break
            tra1 = train_index[train_index1]
            tes1 = train_index[test_index1]
            response1 =  response.iloc[tra1,:]
            response2 =  response.iloc[tes1,:]
        else:
            response1 =  response.iloc[train_index,:]
            response2 =  response.iloc[test_index,:]
        tol_snv = mrna_data.join(response1,how='inner')
        model = SelectKBest(chi2, k=chosefea1)
        #print(tol_snv.values[:,-1])
        temp = tol_snv.values[:,0:-1]
        temp = np.nan_to_num(temp, nan=0.0)
        x_data1 = model.fit_transform(temp,tol_snv.values[:,-1])
        #x_data1 = model.fit_transform(tol_snv.values[:,4300:4632],tol_snv.values[:,-1])
        fea = model.get_support()
        mrna_data_sub = mrna_data.loc[:,fea]
        mrna_data_chosegenes = list(mrna_data_sub.columns)
        union_gene = mrna_data_genes
        ##
        pset = model.pvalues_
        mrna_data_chosegenes_pvalues = pset[fea]
        ora_pset = {}
        ora_pset['names'] = mrna_data_chosegenes
        ora_pset['pvalues'] = mrna_data_chosegenes_pvalues
        data = pd.DataFrame(ora_pset)

        directory_path = './Data/coef/{}/h{}/'.format(cancer,p)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        data.to_csv('./Data/coef/{}/h{}/snv_chi2_pvalues.csv'.format(cancer,p),index=False,encoding='UTF-8')
        
        mrna_pvalues = []
        for h in range(pathway_num):
            pathway = pathways[h]
            genes_inter_mrna = pathway_mrna[pathway]
            record_num = pathway_genes_num[pathway]
            pathway_inter_gene =  set(mrna_data_chosegenes).intersection(set(genes_inter_mrna))
            a1 =record_num
            a2 = len(pathway_inter_gene)
            a3 = a1 - a2
            a4 = chosefea - a2
            a5 = mrna_data_genenum - a1 - a4
            table = np.array([[a2, a3],[a4, a5]])
            #odds_ratio, p_value = ss.fisher_exact(table,alternative='greater')
            odds_ratio, p_value = ss.fisher_exact(table)
            mrna_pvalues.append(p_value)
                       
        mrna_pathways =list(np.array(pathways)[np.array(mrna_pvalues)<=threa1])
        print('The number of the chosed pathway_mrna: {}'.format(len(mrna_pathways))) 
        print('The number of the chosed pathway_amp_1: {}'.format(len(amp_pathways)))
        common_mrna_pathways = list(set(mrna_pathways).intersection(set(mrna_prior_pathways)))
        print('The number of the common pathway_amp_1: {}'.format(len(common_mrna_pathways)))

        pset =  np.array(mrna_pvalues)[np.array(mrna_pvalues)<=threa1]
        ora_pset = {}
        ora_pset['names'] = mrna_pathways
        ora_pset['pvalues'] = pset
        data = pd.DataFrame(ora_pset)
        data.to_csv('./Data/coef/{}/h{}/pathway_pvalues.csv'.format(cancer,p),index=False,encoding='UTF-8')
       
        cnv_amp_chosegenes_diff = list(set(cnv_amp_chosegenes).difference(set(mrna_data_chosegenes)))
        amp_pathways_diff = list(set(amp_pathways).difference(set(mrna_pathways)))
        print('The number of the chosed pathway_amp: {}'.format(len(amp_pathways_diff))) 
        #common_mrna_genes = list(set(mrna_data_chosegenes).union(set(mrna_prior_chosegenes)))
        #common_mrna_pathways = list(set(mrna_pathways).union(set(mrna_prior_pathways)))
        #print('The number of pathways:',len(common_mrna_pathways))

        # In[18]:
        print(mrna_data.shape)
        mrna_data = mrna_data[union_gene]
        rows = mrna_data.index      
        y = response.values
        y = y.reshape(-1)
        x = mrna_data.values

        #GBM(0.61,2.61);LGG(1.54,0.74);KIRC(1.49,0.715)
        ind_train = rows.isin(response1.index)
        ind_test = rows.isin(response2.index)
        X_train1 = x[ind_train]
        y_train = y[ind_train]
        X_test1 = x[ind_test]
        y_test = y[ind_test]

        X_train2 = copy.deepcopy(X_train1)
        X_test2 = copy.deepcopy(X_test1)
    
        network_edges_new = copy.deepcopy(network_edges)
        union_network_gene =list(set(union_gene).intersection(set(network_genes['genes'])))
        if len(union_network_gene)!=len(union_gene):
            nodes = set(union_gene).difference(union_network_gene)
            df = {}
            df['start'] = list(nodes)
            df['end'] = list(nodes)
            df = pd.DataFrame(df)
            network_edges_new = pd.concat([network_edges_new, df], ignore_index=True)

        # mrna_data:
        network_edges_mrna = network_edges_new[(network_edges_new['start'].isin(union_gene)) & (network_edges_new['end'].isin(mrna_data_chosegenes))]
        net1 = nx.from_pandas_edgelist(network_edges_mrna, 'start', 'end', create_using=nx.DiGraph())
        net1.name = 'HumanNet'
        nodes = set(union_gene).difference(set(net1.nodes))
        if len(nodes)>0:
            root_node = 'mrna_0'
            edges = [(node,root_node) for node in nodes]
            net1.add_edges_from(edges)
            mrna_data_chosegenes.append('mrna_0')
        network_edges_mrna = network_edges_new[(network_edges_new['start'].isin(mrna_data_chosegenes)) & (network_edges_new['end'].isin(mrna_data_chosegenes))]
        net2 = nx.from_pandas_edgelist(network_edges_mrna, 'start', 'end', create_using=nx.DiGraph())
        net2.name = 'GeneToGene'
        root_node = 'mrna_0'
        edges = [(root_node, node) for node in mrna_data_chosegenes]
        net2.add_edges_from(edges)
        edges = [(node,root_node) for node in mrna_data_chosegenes]
        net2.add_edges_from(edges)
        # Read gene-pathway annotation relationships
        #pathway_genes = pd.read_csv('./Data/pathways/PK/PU.txt', sep='\t',names=['gene','group'],dtype=str)
        pathway_genes_mrna = pathway_genes[(pathway_genes['gene'].isin(mrna_data_chosegenes)) & (pathway_genes['group'].isin(mrna_pathways))]
        mrna_pathways_update = list(set(pathway_genes_mrna['group']))
        common_mrna_pathways_update = list(set(mrna_pathways_update).intersection(set(common_mrna_pathways)))
        print('The number of the screened pathway_mrna: {}'.format(len(mrna_pathways_update))) 
        net3 = nx.from_pandas_edgelist(pathway_genes_mrna, 'gene', 'group', create_using=nx.DiGraph())
        net3.name = 'GeneToPathway_mrna'
        nodes = set(mrna_data_chosegenes).difference(set(pathway_genes_mrna['gene']))
        if len(nodes)>0:
            root_node = 'mrna_1'
            edges = [(node,root_node) for node in nodes]
            net3.add_edges_from(edges)
            mrna_pathways_update.append(root_node)
        #
        pathway_network_mrna = pathway_network[(pathway_network['start'].isin(mrna_pathways_update)) & (pathway_network['end'].isin(mrna_pathways_update))]
        net4 = nx.from_pandas_edgelist(pathway_network_mrna, 'start', 'end', create_using=nx.DiGraph())
        net4.name = 'PathwayToPathway'
        nodes = set(mrna_pathways_update).difference(set(net4.nodes))
        if len(nodes)>0:
            root_node = 'mrna_1'
            edges = [(node,root_node) for node in nodes]
            net4.add_edges_from(edges)
            edges = [(root_node,node) for node in nodes]
            net4.add_edges_from(edges)

        layers = get_layers_first(union_gene,mrna_data_chosegenes,mrna_pathways_update,net1,net2,net3,net4)
        Get_Node_relation_mrna = get_layer_maps(union_gene, layers,mrna_pathways_update)
        #snv_amp
        # In[24]:
        network_edges_amp = network_edges_new[(network_edges_new['start'].isin(union_gene)) & (network_edges_new['end'].isin(cnv_amp_chosegenes_diff))]
        cnv_amp_chosegenes_diff = list(set(network_edges_amp['end']))
        net1 = nx.from_pandas_edgelist(network_edges_amp, 'start', 'end', create_using=nx.DiGraph())
        net1.name = 'HumanNet'
        nodes = set(union_gene).difference(set(net1.nodes))
        if len(nodes)>0:
            root_node = 'amp_0'
            edges = [(node, root_node) for node in nodes]
            net1.add_edges_from(edges)
            cnv_amp_chosegenes_diff.append('amp_0')
        network_edges_amp = network_edges_new[(network_edges_new['start'].isin(cnv_amp_chosegenes_diff)) & (network_edges_new['end'].isin(cnv_amp_chosegenes_diff))]
        net2 = nx.from_pandas_edgelist(network_edges_amp, 'start', 'end', create_using=nx.DiGraph())
        net2.name = 'GeneToGene'
        root_node = 'amp_0'
        edges = [(root_node, node) for node in cnv_amp_chosegenes_diff]
        net2.add_edges_from(edges)
        edges = [(node,root_node) for node in mrna_data_chosegenes]
        net2.add_edges_from(edges)
        # Read gene-pathway annotation relationships
        #pathway_genes = pd.read_csv('./data/pathways/PK/PU.txt', sep='\t',names=['gene','group'],dtype=str)
        pathway_genes_amp = pathway_genes[(pathway_genes['gene'].isin(cnv_amp_chosegenes_diff)) & (pathway_genes['group'].isin(amp_pathways_diff))]
        amp_pathways_diff = list(set(pathway_genes_amp['group']))
        print('The number of the screened pathway_amp: {}'.format(len(amp_pathways_diff)))
        net3 = nx.from_pandas_edgelist(pathway_genes_amp, 'gene', 'group', create_using=nx.DiGraph())
        net3.name = 'GeneToPathway_mrna'
        nodes = set(cnv_amp_chosegenes_diff).difference(set(pathway_genes_amp['gene']))
        if len(nodes)>0:
            root_node = 'amp_1'
            edges = [(node,root_node) for node in nodes]
            net3.add_edges_from(edges)
            amp_pathways_diff.append(root_node)
        
        pathway_network_amp = pathway_network[(pathway_network['start'].isin(amp_pathways_diff)) & (pathway_network['end'].isin(amp_pathways_diff))]
        net4 = nx.from_pandas_edgelist(pathway_network_amp, 'start', 'end', create_using=nx.DiGraph())
        net4.name = 'PathwayToPathway'
        nodes = set(amp_pathways_diff).difference(set(net4.nodes))
        if len(nodes)>0:
            root_node = 'amp_1'
            edges = [(node,root_node) for node in nodes]
            net4.add_edges_from(edges)
            edges = [(root_node,node) for node in nodes]
            net4.add_edges_from(edges)

        layers = get_layers_first(union_gene,cnv_amp_chosegenes_diff,amp_pathways_diff,net1,net2,net3,net4)
        Get_Node_relation_amp = get_layer_maps_first(union_gene,layers,amp_pathways_diff)
        #
        all_pathways = copy.deepcopy(mrna_pathways_update)
        all_pathways.extend(amp_pathways_diff)

        amp_pathways_update = copy.deepcopy(common_mrna_pathways_update)
        amp_pathways_update.extend(amp_pathways_diff)

        net5,subnames = createNetwork14(mrna_pathways_update,amp_pathways_update,n_hidden_layers_PtH[0])  
        Get_Node_relation_next = get_layers_map_sec(all_pathways,net5,subnames)

        model = create_models_pheno17(x)

        history = model.fit([X_train1,X_train2],[y_train]*5,epochs=30,batch_size =30,shuffle=True)
        y_pred = model.predict([X_test1,X_test2])
    
        y_pred1 = np.mean(np.array(y_pred),axis=0)
        kfscore.append(evaluates(y_test, y_pred1))
        results = evaluates(y_test, y_pred1)
        print("results : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(results[0],3),round(results[1],3),round(results[2],3),round(results[3],3),round(results[4],3),round(results[5],3),round(results[6],3)))
        
        # feature importance
        '''
        explain_x1 = X_train1[np.where(y_train!=0)]
        explain_x2 = X_train2[np.where(y_train!=0)]
        explain_y1 = y_train[np.where(y_train!=0)]
        explain_x = np.array([explain_x1,explain_x2])
        explain_y = explain_y1
        coef_ = get_coef_importance(model,explain_x, explain_y, target=-1,feature_importance='deepexplain_deeplift')
        cof_values = ['m00','m01','m10','m11','m20','m21']
        name = [Get_Node_relation_mrna[2].index,Get_Node_relation_amp[2].index,Get_Node_relation_mrna[3].index,Get_Node_relation_amp[3].index,
        Get_Node_relation_mrna[3].columns,Get_Node_relation_amp[3].columns]
        #os.mkdir('./data/coef/{}/h{}/'.format(cancer,p))
        for i in range(0,6):
            X = pd.DataFrame()
            X['name'] = name[i]
            X['values'] = coef_[0][cof_values[i]]
            X.to_csv('./Data/coef/{}/h{}/{}.csv'.format(cancer,p,cof_values[i]),index=False,encoding='UTF-8')
        '''
        p = p + 1
        del model
        
       
    #avrrage
    #kfscores = np.array(kfscore).sum(axis= 0)/5.0
    #print("average value : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(kfscores[0],3),round(kfscores[1],3),round(kfscores[2],3),round(kfscores[3],3),round(kfscores[4],3),round(kfscores[5],3),round(results[6],3)))
    #Save_Res.append(kfscores)
    #resu = pd.DataFrame(kfscore)
    #resu.to_csv('./Data/result/{}/{}_{}_{}_{}.csv'.format(cancer,rs,threa,threa1,0))

    
    #  average  five result
    '''
    file_name = ['m00.csv','m01.csv','m10.csv','m11.csv','m20.csv','m21.csv']
    directory_path = './Data/coef/{}/average/'.format(cancer)
    if not os.path.isdir(directory_path):
        os.mkdir(directory_path)
    for j in file_name:
        result0 =pd.DataFrame()
        for i in range(0,5):
            result  = pd.read_csv('./Data/coef/{}/h{}/{}'.format(cancer,i,j))
            if i==0:
                result0 = result
            else:
                result0 = pd.merge(result,result0, on='name', how='outer')
    #    
        result1 = result0.set_index('name')
        #result2 = result1.apply(lambda col: col.fillna(col.min()))
        result2 = result1.apply(lambda col: col.fillna(0))
        result3 = result2.mean(axis=1)
        result3 = result3.to_frame()
        result3.columns = ['values']
        results = result3.sort_values('values',ascending=False)
        results.to_csv('./Data/coef/{}/average/{}'.format(cancer,j),index = True)

        result21 = result2.applymap(abs)
        result3 = result21.mean(axis=1)
        result3 = result3.to_frame()
        result3.columns = ['values']
        results = result3.sort_values('values',ascending=False)
        results.to_csv('./Data/coef/{}/average/avg_{}'.format(cancer,j),index = True)
    '''  
    #avrrage
    '''
    kfscores = np.array(kfscore).sum(axis= 0)/5.0
    print(kfscores)
    print("average value : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(kfscores[0],3),round(kfscores[1],3),round(kfscores[2],3),round(kfscores[3],3),round(kfscores[4],3),round(kfscores[5],3),round(kfscores[6],3)))
    resu = pd.DataFrame(kfscore)
    resu.to_csv('./Data/result/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(cancer,chosefea,threa,threa1,dr,lr,sel,bel,is_flag,0),index = False)
    resu = pd.DataFrame(kfscores)
    resu.to_csv('./Data/result/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_avg.csv'.format(cancer,chosefea,threa,threa1,dr,lr,sel,bel,is_flag,0),index = False)
    '''
    #
    kfscores = np.array(kfscore).sum(axis= 0)/5.0
    print(kfscores)
    print("average value : pre = {}, acc = {},rec = {},f1 = {},auc = {},aupr = {},auprc = {}".format(round(kfscores[0],3),round(kfscores[1],3),round(kfscores[2],3),round(kfscores[3],3),round(kfscores[4],3),round(kfscores[5],3),round(kfscores[6],3)))
    resu = pd.DataFrame(kfscore)
    resu.to_csv('./Data/result/{}/{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(cancer,n2,n3,n4,ba1,ba2,ba3,is_flag,0),index = False)
    resu = pd.DataFrame(kfscores)
    resu.to_csv('./Data/result/{}/{}_{}_{}_{}_{}_{}_{}_{}_avg.csv'.format(cancer,n2,n3,n4,ba1,ba2,ba3,is_flag,0),index = False)    

 