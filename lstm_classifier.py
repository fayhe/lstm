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

# coding=utf-8
import urllib2
import json
from elasticsearch import Elasticsearch
import re


print('start model...')


def filer_str(temp):
	string = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),temp)
	print(string)
	return string

def save_raw_data(file,data):	
	file.write( data.encode('utf8').replace('\n','').replace('\'','').replace('"','').replace(',','')  + '\n')


def fenci(summary):
	nlp_features_list = ""
	nlp_features = pseg.cut(summary)
	for w in nlp_features:
		if w.flag == 'ns' or w.flag == 'n' or w.flag == 'ng' or w.flag == 'nl' or "a" in w.flag or "v" in w.flag or w.flag == 't':
			#print w.word + w.flag 
			nlp_features_list = nlp_features_list + " " + w.word 
	##jieba.analyse.extract_tags(summary,allowPOS=('ns', 'n','ng','nl','t','v','a'), topK=1000)
	#nlp_features_list = ' '.join(nlp_features)
	return nlp_features_list  

es = Elasticsearch()


types = ['剧情','喜剧','动作','爱情', '科幻', '音乐', '纪录片', '悬疑', '情色']
folder = ['juqing','xiju']
file_root_path = 'raw_data/'

foler_index=0
juqing_type_list = []
xiju_type_list = []
dongzuo_type_list = []
aiqing_type_list = []
kehuan_type_list = []
yinyue_type_list = []
jilu_type_list = []
xuanyi_type_list = []
qingse_type_list = []


for type in types:
	#file = open(file_root_path + folder[foler_index], 'w')
	res = es.search(index="douban", doc_type=type, size=2000, body={"query": {"match_all": {}}})
	print("Got %d Hits:" % res['hits']['total'])
	for hit in res['hits']['hits']:
		id = hit["_id"]
		##print(hit["_id"])
#		print(hit["_source"]["alt_title"])
#		print(hit["_source"]["summary"])
		##feature = fenci(hit["_source"]["summary"])
		#save_raw_data(file , hit["_source"]["alt_title"])
		##save_raw_data(file , hit["_source"]["summary"])
		if type == '剧情':
			juqing_type_list.append(filer_str(hit["_source"]["summary"]))
		if type == '喜剧':
			xiju_type_list.append(filer_str(hit["_source"]["summary"]))
		if type == '动作':
			dongzuo_type_list.append(filer_str(hit["_source"]["summary"]))
		if type == '爱情':
			aiqing_type_list.append(filer_str(hit["_source"]["summary"]))
		if type == '科幻':
			kehuan_type_list.append(filer_str(hit["_source"]["summary"]))
		if type == '音乐':
			yinyue_type_list.append(filer_str(hit["_source"]["summary"]))
		if type == '纪录片':
			jilu_type_list.append(filer_str(hit["_source"]["summary"]))
		if type == '悬疑':
			xuanyi_type_list.append(filer_str(hit["_source"]["summary"]))
		if type == '情色':
			qingse_type_list.append(filer_str(hit["_source"]["summary"]))																
		#print(hit["_source"]["summary"])
	foler_index = foler_index+1	
	#print raw_type_list
#	print("Got %d Hits:" % res['hits']['total'])

neg = pd.DataFrame(juqing_type_list) 
print(neg)

pos = pd.DataFrame(xiju_type_list) 
print(pos)

dongzuo = pd.DataFrame(dongzuo_type_list) 
print(dongzuo)

aiqing = pd.DataFrame(aiqing_type_list) 
print(aiqing)

kehuan = pd.DataFrame(kehuan_type_list) 
print(kehuan)

yinyue = pd.DataFrame(yinyue_type_list) 
print(yinyue)

jilu = pd.DataFrame(jilu_type_list) 
print(jilu)

xuanyi = pd.DataFrame(xuanyi_type_list) 
print(xuanyi)

qingse = pd.DataFrame(qingse_type_list) 
print(qingse)

#neg = pd.read_csv('raw_data/juqing', sep=",", header = None)
#print(neg)
#pos=pd.read_txt('raw_data/xiju',header=None,index=None) #读取训练语料完毕
pos['mark']=1
neg['mark']=0 #给训练语料贴上标签
dongzuo['mark']=2
aiqing['mark']=3
kehuan['mark']=4
yinyue['mark']=5
jilu['mark']=6
xuanyi['mark']=7
qingse['mark']=8



pn=pd.concat([pos,neg, dongzuo, aiqing, kehuan, yinyue, jilu, xuanyi, qingse  ],ignore_index=True) #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目

cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)
print (pn['words'])

#comment = pd.read_excel('sum.xls') #读入评论内容
#comment = pd.read_csv('a.csv', encoding='utf-8')
#comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
#comment['words'] = comment['rateContent'].apply(cw) #评论分词 

d2v_train = pd.concat([pn['words']], ignore_index = True) 


print (d2v_train)

w = [] #将所有词语整合在一起
for i in d2v_train:
  w.extend(i)

dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数
del w,d2v_train
dict['id']=list(range(1,len(dict)+1))

get_sent = lambda x: list(dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent) #速度太慢

maxlen = 10000
max_features = 10000

print("Pad sequences (samples x time)")
pn['sent'] = list(sequence.pad_sequences(pn['sent'], maxlen=maxlen))
print(pn['sent']) 

x = np.array(list(pn['sent']))[::2] #训练集
y = np.array(list(pn['mark']))[::2]
xt = np.array(list(pn['sent']))[1::2] #测试集
yt = np.array(list(pn['mark']))[1::2]
xa = np.array(list(pn['sent'])) 
print(xa)
ya = np.array(list(pn['mark']))

print('Build model...')
model = Sequential()
model.add(Dense(512, input_shape=(max_features,), activation='tanh'))
#model.add(LSTM(256)) # try using a GRU instead, for fun
#model.add(Dropout(0.5))
#model.add(Dense( 1));
	
	##, activation='softmax'))
#model.add(Activation('softmax'))

#model.add(Dense(512, input_shape=(max_features,), activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(9))
model.add(Activation('softmax'))

#model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', class_mode="categorical")


print('before tain...')
model.fit(xa, ya, batch_size=16, nb_epoch=10) #训练时间为若干个小时

#classes = model.predict_classes(xa)
#acc = np_utils.accuracy(classes, ya)
#print('Test accuracy:', acc)

# evaluate the model

# serialize model to JSON
model_json = model.to_json()
with open("model_test_10000.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_test_10000.h5")
print("Saved model to disk")

scores = model.evaluate(xa, ya, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
