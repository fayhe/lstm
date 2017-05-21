# coding=utf-8
from __future__ import absolute_import #导入3.x的特征函数
from __future__ import print_function

import pandas as pd #导入Pandas
import numpy as np #导入Numpy
import jieba #导入结巴分词

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU


from keras.models import model_from_json



print('start model...')
neg=pd.read_excel('neg.xls',header=None,index=None)
pos=pd.read_excel('pos.xls',header=None,index=None) #读取训练语料完毕
pos['mark']=1
neg['mark']=0 #给训练语料贴上标签
pn=pd.concat([pos,neg],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目

cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)
print (pn['words'])

comment = pd.read_excel('sum.xls') #读入评论内容
#comment = pd.read_csv('a.csv', encoding='utf-8')
comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
comment['words'] = comment['rateContent'].apply(cw) #评论分词 

d2v_train = pd.concat([pn['words'], comment['words']], ignore_index = True) 
print (d2v_train)

w = [] #将所有词语整合在一起
for i in d2v_train:
  w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train
dict['id']=list(range(1,len(dict)+1))

get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent) #速度太慢

maxlen = 50

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))

x = np.array(list(pn['sent']))[::2] #训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) #全集
ya = np.array(list(pn['mark']))

print('test model...')
#model = Sequential()
#model.add(Embedding(len(dict)+1, 256))
#model.add(LSTM(256)) # try using a GRU instead, for fun

#model.add(Dropout(0.5))
#model.add(Dense( 1))
#model.add(Activation('sigmoid'))

#model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")


print('test tain...')
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(xa, ya, verbose=0)

print(score)

#model.fit(xa, ya, batch_size=16, nb_epoch=10) #训练时间为若干个小时

#classes = model.predict_classes(xa)
#acc = np_utils.accuracy(classes, ya)
#print('Test accuracy:', acc)

# evaluate the model

# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
# serialize weights to HDF5
#model.save_weights("model.h5")
#print("Saved model to disk")

#scores = model.evaluate(xa, ya, verbose=0)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
