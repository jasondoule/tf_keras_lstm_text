'''
    Script to classify neg/pos comments with LSTM

'''
from __future__ import absolute_import
from __future__ import print_function

import pandas as pd
import numpy as np
import jieba

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import load_model

neg=pd.read_excel('neg.xls',header=None,index=None)
pos=pd.read_excel('pos.xls',header=None,index=None) #读取训练语料完毕

pos['mark']=1
neg['mark']=0

pn=pd.concat([pos,neg],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos)

cw = lambda x: list(jieba.cut(x)) #定义分词函数

pn['words'] = pn[0].apply(cw)


comment = pd.read_excel('sum.xls') #读入评论内容
comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw) #评论分词

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True)

w = [] #将所有词语整合在一起
for i in d2v_train:
    w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train

dict['id']=list(range(1,len(dict)+1))

get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent) #速度太慢


'''
pn['sent']:
21103            [212, 19, 1, 211, 11, 19, 1, 223, 181, 1]
21104    [209, 223, 88, 658, 1, 224, 223, 16, 5, 1832, ...
'''

maxlen = 50 #timestep

print("Pad sequences (samples x time/timesteps)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))  #keras func to format input data


x = np.array(list(pn['sent']))[::2] #训练集  start from 0 till end, every other element
#print('x',x.shape)
y = np.array(list(pn['mark']))[::2]
#print('y',y.shape)
xt = np.array(list(pn['sent']))[1::2] #测试集 start from 1 till end, every other element
#print('xt',xt.shape)
yt = np.array(list(pn['mark']))[1::2]
#print('yt',yt.shape)
xa = np.array(list(pn['sent'])) #全集
#print('xa',xa.shape)
ya = np.array(list(pn['mark']))
#print('ya',ya.shape)


# build or load model
model_path = 'lstm_chinese_classify_model.h5'

def buildModel():
    print('Build model...')

    model = Sequential()
    model.add(Embedding(len(dict) + 1, 256))  # 256 = 词向量长度  This embedding layer turns the input data to 3D
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

try:
    model = load_model(model_path)
except:
    model = buildModel()


model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

print(model.summary())

# train the model, output generated text
for iteration in range(1, 20):
    print()
    print('-' * 80)
    print('Iteration', iteration)
    model.fit(x, y, batch_size=32, epochs=3, verbose=1)
    print('Save model...')
    model.save(model_path)

    classes = model.predict(xt,batch_size=32)
    print('predicted classes:')
    print(classes)

    results = model.evaluate(xt,yt,batch_size=32,verbose=1)
    print('predicted results:')
    print(results)
    #print('Test accuracy:', acc)