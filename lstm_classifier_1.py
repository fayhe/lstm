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
from keras.models import model_from_json
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
#print(neg)

pos = pd.DataFrame(xiju_type_list) 
#print(pos)

dongzuo = pd.DataFrame(dongzuo_type_list) 
##print(dongzuo)

aiqing = pd.DataFrame(aiqing_type_list) 
##print(aiqing)

kehuan = pd.DataFrame(kehuan_type_list) 
##print(kehuan)

yinyue = pd.DataFrame(yinyue_type_list) 
##print(yinyue)

jilu = pd.DataFrame(jilu_type_list) 
##print(jilu)

xuanyi = pd.DataFrame(xuanyi_type_list) 
##print(xuanyi)

qingse = pd.DataFrame(qingse_type_list) 
##print(qingse)


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



pn=pd.concat([pos,neg, dongzuo, aiqing, kehuan, yinyue, jilu, xuanyi, qingse  ],ignore_index=True) # #合并语料
neglen=len(neg)
poslen=len(pos) #计算语料数目

cw = lambda x: list(jieba.cut(x)) #定义分词函数
pn['words'] = pn[0].apply(cw)
#print (pn['words'])

#comment = pd.read_excel('sum.xls') #读入评论内容
#comment = pd.read_csv('a.csv', encoding='utf-8')
#comment = comment[comment['rateContent'].notnull()] #仅读取非空评论
#comment['words'] = comment['rateContent'].apply(cw) #评论分词 

d2v_train = pd.concat([pn['words']], ignore_index = True) 


#print (d2v_train)

w = [] #将所有词语整合在一起
for i in d2v_train:
  w.extend(i)

print("length!!!")
print(len(w))
dict = pd.DataFrame(pd.Series(w).value_counts()) #统计词的出现次数

print(dict)

del w,d2v_train
dict['id']=list(range(1,len(dict)+1))

print(dict)

##return x - y if x>y else abs(x-y)  
get_sent = lambda x: list( dict['id'][x])
pn['sent'] = pn['words'].apply(get_sent)
print(pn['sent'])




xiju_type_list = ['结婚五周年纪念日的早上，尼克·邓恩（本·阿弗莱克 Ben Affleck 饰）来到妹妹玛戈（凯莉·库恩 Carrie Coon 饰）的酒吧，咒骂抱怨那个曾经彼此恩爱缠绵的妻子艾米（罗莎蒙德·派克 Rosamund Pike 饰）以及全然看不见希望的婚姻。当他返回家中时， 却发现客厅留下了暴行的痕迹，而妻子竟不见了踪影。女探员朗达·邦妮（金·迪肯斯 Kim Dickens 饰）接到报案后赶来调查，而现场留下的种种蛛丝马迹似乎昭示着这并非是一件寻常的失踪案，其背后或许隐藏着裂变于夫妻之情的谋杀罪行。艾米的失踪通过媒体大肆渲染和妄加揣测很快闻名全国，品行不端的尼克被推上风口浪尖，至今不见踪影的爱人对他进行无情审判，你侬我侬的甜言蜜语早已化作以血洗血的复仇与折磨…… ',
                  '掌管跨国经济体的路晋（金城武 饰）刻薄挑剔，仅有美食这一个爱好。创意厨师顾胜男（周冬雨 饰）迷糊邋遢得过且过。一次收购，一道女巫汤，路晋喜欢上了顾胜男的菜却讨厌极了她这个人。喜欢还是讨厌究竟何去何从？',
                  '想要在巨星姐姐面前证明自己的18线小演员上官娣娣，和多年期待真爱却在最后被狠狠出卖的空间站黑鸟面馆老板娘许春梅，当两个人的世界以想象不到的方式不期而遇，她们的命运会发生什么样的改变?',
                  '故事发生在《加勒比海盗3：世界的尽头》沉船湾之战20年后。男孩亨利（布兰顿·思怀兹 Brenton Thwaites 饰）随英国海军出航寻找被聚魂棺诅咒的父亲“深海阎王”威尔·特纳（奥兰多·布鲁姆 Orlando Bloom 饰），却在百慕大三角遭遇被解封的亡灵萨拉查船长（哈维尔·巴登 Javier Bardem 饰）。获取自由的萨拉查屠尽加勒比海盗，征服了整个海域。里海海盗王赫克托·巴博萨船长（杰弗里·拉什 Geoffrey Rush 饰）在女巫Haifaa Meni（格什菲·法拉哈尼 Golshifteh Farahani 饰）口中得知了萨拉查的真实目的：为寻找他的宿敌杰克船长（约翰尼·德普 Johnny Depp 饰）。海盗的命运皆压在落魄的老杰克被封印的黑珍珠号，以及天文学家卡琳娜·史密斯（卡雅·斯考达里奥 Kaya Scodelario 饰）口中的远古三叉戟上。',
                  '多米尼克（范·迪塞尔 Vin Diesel 饰）与莱蒂（米歇尔·罗德里格兹 Michelle Rodriguez 饰）共度蜜月，布莱恩与米娅退出了赛车界，这支曾环游世界的顶级飞车家族队伍的生活正渐趋平淡。然而，一位神秘女子Cipher（查理兹·塞隆 Charlize T heron 饰）的出现，令整个队伍卷入信任与背叛的危机，面临前所未有的考验']

neg1 = pd.DataFrame(xiju_type_list)
neg1['words'] = neg1[0].apply(cw)
print(neg1)
neg1['sent'] = neg1['words'].apply(get_sent)
print(neg1['sent'])

maxlen = 1000

print("Pad sequences (samples x time)")
print(neg1['sent']) 

new_sent= []
for index, row in neg1.iterrows():
	l = [ 0 if np.isnan(word_index) else word_index for word_index in row['sent']]
	print(l)
	new_sent.append(l)
	#for word_index in row['sent']:
	#	if(np.isnan(word_index)):			
	#		word_index = -1
neg1['new_sent'] = 	new_sent
print(neg1['new_sent']) 			
neg1['new_sent'] = list(sequence.pad_sequences(neg1['new_sent'], maxlen=maxlen))
print(neg1['new_sent']) 

###############



xa = np.array(list(neg1['new_sent'])) #全集


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
json_file = open('model_test_1000.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_test_1000.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
##loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', class_mode="categorical")

classes = loaded_model.predict_classes(xa)
print(classes)

scores = loaded_model.predict(xa)
print(scores)

