# coding=utf-8
import urllib2
import json
from elasticsearch import Elasticsearch
import jieba
import jieba.analyse
import jieba.posseg as pseg
import pandas as pd 

def save_raw_data(file,data):	
	file.write( data.encode('utf8').replace('\n','').replace('\'','').replace('"','').replace(',','')  + '\n')


def fenci(summary):
	print "hahah"
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
types = ['剧情','喜剧']
folder = ['juqing','xiju']
file_root_path = 'raw_data/'

foler_index=0
juqing_type_list = []
xiju_type_list = []
for type in types:
	file = open(file_root_path + folder[foler_index], 'w')
	res = es.search(index="douban", doc_type=type, size=2000, body={"query": {"match_all": {}}})
	print("Got %d Hits:" % res['hits']['total'])
	for hit in res['hits']['hits']:
		id = hit["_id"]
		##print(hit["_id"])
#		print(hit["_source"]["alt_title"])
#		print(hit["_source"]["summary"])
		##feature = fenci(hit["_source"]["summary"])
		#save_raw_data(file , hit["_source"]["alt_title"])
		save_raw_data(file , hit["_source"]["summary"])
		if type == '剧情':
			juqing_type_list.append(hit["_source"]["summary"])
		if type == '喜剧':
			xiju_type_list.append(hit["_source"]["summary"])			
		#print(hit["_source"]["summary"])
	foler_index = foler_index+1	
	#print raw_type_list
#	print("Got %d Hits:" % res['hits']['total'])

neg = pd.DataFrame(juqing_type_list) 
print neg

pos = pd.DataFrame(xiju_type_list) 
print pos
