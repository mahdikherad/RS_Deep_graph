# In[0] Imports
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from keras.models import Sequential, Model
from keras.layers import Dense
import keras.backend as Kb
import matplotlib.pyplot as plt
import networkx as nx
from node2vec import Node2Vec
from scipy.sparse import csr_matrix
import community
import igraph as ig
# In[1] load data epinions
ratings = pd.read_csv('ratings_data.txt', delim_whitespace=True,header=None,skiprows=1)
trusts = pd.read_csv('trust_data.txt', delim_whitespace=True,header=None,skiprows=1)
ratings=ratings.values
trusts=trusts.values
num_users=49290
num_items = 139738
ratings=ratings.astype(int)
trusts=trusts.astype(int)
ratings=np.delete(ratings, np.where(ratings[:,0]<0), 0)
ratings=np.delete(ratings, np.where(ratings[:,0]==40163), 0)
trusts=np.delete(trusts, np.where(trusts[:,0]<0), 0)
trusts=np.delete(trusts, np.where(trusts[:,1]<0), 0)
trusts=np.delete(trusts, np.where(trusts[:,0]==40163), 0)
trusts=np.delete(trusts, np.where(trusts[:,1]==40163), 0)
# In[2] load data Ciao
ratings = pd.read_csv('rating.csv',header=None)
trusts = pd.read_csv('trustnetwork.csv',header=None)
ratings=ratings.values
trusts=trusts.values
num_users=7375
num_items = 106797
ratings=ratings.astype(int)
trusts=trusts.astype(int)
ratings=np.delete(ratings, np.where(ratings[:,2]<1), 0)
# In[3] init
R=csr_matrix((ratings[:,2], (ratings[:,0]-1, ratings[:,1]-1)), shape=(num_users, num_items))
tru=csr_matrix((np.ones((np.shape(trusts)[0])), (trusts[:,0]-1, trusts[:,1]-1)), shape=(num_users, num_users))

rn=R.getnnz(1)>0;
R = R[rn][:,R.getnnz(0)>0]
tru=tru[rn]
tru=tru[:,rn]
num_users=R.shape[0]
num_items =R.shape[1]
K=10
alpha=0.1
beta=0.01

averagebias=R.sum(1)/(R!=0).sum(1).astype(float)
mybias=R.sum(0)/(R!=0).sum(0).astype(float)
avg=np.mean(mybias)
nnt=np.nonzero(tru)
mys=[]
for i in range(num_users):
    mys.append(np.nonzero(tru[i,:]))
Rn=R.nonzero();
Rm=R[Rn[0],Rn[1]];
Rm=np.array(Rm)
Rm=Rm.reshape((Rn[0].shape))
samples=[Rn[0],Rn[1],Rm]
samples=np.stack(samples).T
# In[4] functions
def cosinesim(vA,vB):
    return ((np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB)))+1)/2
def mytrust(P,u,v):
    cosinesimsum=0
    cosinesimsum=[cosinesim(P[:,u],P[:,ss]) for ss in mys[u][1]]
    cosinesimsum=np.sum(cosinesimsum)
    if cosinesimsum>0:
        return (cosinesim(P[:,u],P[:,v]))/cosinesimsum
    else:
        return 0
def rmse(y_true, y_pred):
	mask_true = Kb.cast(Kb.not_equal(y_true, 0), Kb.floatx())
	masked_squared_error = mask_true * Kb.square((y_true - y_pred))
	masked_mse = Kb.sum(masked_squared_error, axis=-1) / Kb.sum(mask_true, axis=-1)
	return Kb.sqrt(masked_mse)
def nn_batch_generator(X_data, batch_size):
    samples_per_epoch = X_data.shape[0]
    number_of_batches = samples_per_epoch/batch_size
    counter=0
    index = np.arange(np.shape(X_data)[0])
    while 1:
        index_batch = index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X_data[index_batch,:].todense()
        counter += 1
        yield np.array(X_batch),np.array(X_batch)
        if (counter > number_of_batches):
            counter=0
def myAE(X,bs):
    m = Sequential()
    m.add(Dense(128,  activation='selu', input_shape=(X.shape[1],)))
    m.add(Dense(64,  activation='selu'))
    m.add(Dense(32,  activation='selu'))
    m.add(Dense(K,  activation='sigmoid', name="bottleneck"))
    m.add(Dense(32,  activation='selu'))
    m.add(Dense(64,  activation='selu'))
    m.add(Dense(128,  activation='selu'))
    m.add(Dense(X.shape[1],  activation='selu'))
    m.compile(loss=rmse, optimizer ='Adam')
    m.fit_generator(nn_batch_generator(X, bs),
                    steps_per_epoch=X.shape[0] // bs, epochs=1, verbose=1)
    encoderA = Model(m.input, m.get_layer('bottleneck').output)
    ZencX = encoderA.predict_generator(nn_batch_generator(X, bs),
                    steps=X.shape[0] // bs)
    return ZencX
def countN(comu,N):
    CN=np.zeros((N,1))
    for j in range(N):
        CN[j,0]=[i for i in range(len(comu)) if j in comu[i]][0]
    return CN
def mostimportance(CN,bet_cen):
    mi=np.zeros((np.max(CN)+1,1),dtype=int)
    for j in range(np.max(CN)+1):
        argm=np.argmax(bet_cen[np.where(CN==j)[0],:][:,1])
        mi[j,0]=int(bet_cen[np.where(CN==j)[0],:][argm,0])
    return mi  
def SGD(train,test,N,M,eta,K,lambda_C,lambda_T,lambda_P,lambda_Q,lambda_W,Step):
    # train: train data
    # test: test data
    # N:the number of user
    # M:the number of item
    # eta: the learning rata
    # K: the number of latent factor
    # lambda: regularization parameters
    # Step: the max iteration
    W = np.random.normal(0, 0.1, (K, N))
    P = myAE(R,43)
    Q = myAE(csr_matrix.transpose(R),109)
    P=P.T
    Q=Q.T
    P=minmax_scale(P,feature_range=(0,0.1),axis=1)
    Q=minmax_scale(Q,feature_range=(0,0.1),axis=1)
    rmse=[]
    for ste in range(Step):
        for data in train:
            i=data[0]
            j=data[1]
            r=data[2]
            prediction = np.dot(P[:,i],Q[:,j])+np.dot(W[:,i].T,N2V[:,i])+avg+(averagebias[i,0]-avg)+(mybias[0,j]-avg)
            e = (r - prediction)
            communityterm=P[:,i]-P[:,mi[CN[i,0],0]]
            mysum=np.dot((P[:,i].reshape(K,1)-P[:,myt[i]]),T[i,myt[i]].todense().T)
            mysum=np.array(mysum)
            mysum=mysum.reshape((K,))
            P[:,i]=P[:,i]+eta * (e*(Q[:,j])- lambda_C*communityterm- lambda_T*mysum- lambda_P*P[:,i])
            Q[:,j]=Q[:,j]+eta * (e*(P[:,i]) - lambda_Q*Q[:,j])
            W[:,i]=W[:,i]+eta * (e*N2V[:,i]-lambda_W*W[:,i])
        rms=RMSE(P,Q,W,test)
        rmse.append(rms)
        print(ste,rms)
    return rmse,P,Q,W
           
def RMSE(P,Q,W,test):
    count=len(test)
    sum_rmse=0.0
    for t in test:
        i=t[0]
        j=t[1]
        r=t[2]
        pr = np.dot(P[:,i],Q[:,j])+np.dot(W[:,i].T,N2V[:,i])+avg+(averagebias[i,0]-avg)+(mybias[0,j]-avg)
        sum_rmse+=np.square(r-pr)
    rmse=np.sqrt(sum_rmse/count)
    return rmse

def Figure(rmse):
    x = range(len(rmse))
    plt.plot(x, rmse, color='r',linewidth=3)
    plt.title('Convergence curve')
    plt.xlabel('Iterations')
    plt.ylabel('RMSE')
    plt.show()
# In[5] node2vec
Gtru = nx.from_scipy_sparse_matrix(tru)
node2vec = Node2Vec(Gtru, dimensions=K, walk_length=10, num_walks=20, workers=1)  # Use temp_folder for big graphs
model = node2vec.fit(window=10, min_count=1, batch_words=4)
N2V=model.wv.vectors
N2V=N2V.T
N2V=minmax_scale(N2V, axis = 1)
# In[6] my trust
T=csr_matrix((num_users, num_users))
nntsh=nnt[0].shape[0]
for ii in range(nntsh):
    T[nnt[0][ii],nnt[1][ii]]=mytrust(N2V,nnt[0][ii],nnt[1][ii])
myt=[]
for i in range(num_users):
    myt.append(T[i,:].nonzero()[1])
# In[7] community detection
partition = community.best_partition(Gtru)
comu=list(partition.values())
CN=np.array(comu)
CN=CN.reshape((num_users,1))
CN=CN.astype(int)
# In[8] HITS
nx.write_pajek(Gtru,"Gtru.net")
G=ig.Graph.Read_Pajek("Gtru.net")
au=G.authority_score()
au_score=[(i, au[i]) for i in range(0, len(au))]
au_score=np.array(au_score)
mi=mostimportance(CN,au_score)
# In[9] data split to train and test
np.random.shuffle(samples)
train=samples[0:int(len(samples)*0.8)]
test=samples[int(len(samples)*0.8):]
# In[10] set parameters
N=num_users
M=num_items
eta=0.005
K=10
lambda_C=0.1
lambda_T=0.1
lambda_P=0.1
lambda_Q=0.1
lambda_W=0.1
Step=10
# In[11] run SGD
rmsee,P,Q,W=SGD(train,test,N,M,eta,K,lambda_C,lambda_T,lambda_P,lambda_Q,lambda_W,Step)
print(rmsee[-1])
Figure(rmsee)