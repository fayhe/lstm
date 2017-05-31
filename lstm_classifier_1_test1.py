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
types_e = ['juqing','xiju','dongzuo','aiqing', 'kehuan', 'yinyue', 'jilu', 'xuanyi', 'qingse']
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
                  '故事发生在2029年，彼时，X战警早已经解散，作为为数不多的仅存的变种人，金刚狼罗根（休·杰克曼 Hugh Jackman 饰）和卡利班（斯戴芬·莫昌特 Stephen Merchant 饰）照顾着年迈的X教授（帕特里克·斯图尔特 Patrick Stewart 饰），由于衰老，X教授已经丧失了对于自己超能力的控制，如果不依赖药物，他的超能力就会失控，在全球范围内制造无法挽回的灾难。不仅如此，金刚狼的自愈能力亦随着时间的流逝逐渐减弱，体能和力量都早已经大不如从前。 某日，一位陌生女子找到了金刚狼，将一个名为劳拉（达芙妮·基恩 Dafne Keen 饰）的女孩托付给他，嘱咐他将劳拉送往位于加拿大边境的“伊甸园”。让罗根没有想到的是，劳拉竟然是被植入了自己的基因而培养出的人造变种人，而在传说中的伊甸园里，有着一群和劳拉境遇相似的孩子。邪恶的唐纳德（波伊德·霍布鲁克 Boyd Holbrook 饰）紧紧的追踪着罗根一行人的踪迹，他的目标只有一个，就是将那群人造变种人彻底毁灭。',
'马哈维亚（阿米尔·汗 Aamir Khan 饰）曾经是一名前途无量的摔跤运动员，在放弃了职业生涯后，他最大的遗憾就是没有能够替国家赢得金牌。马哈维亚将这份希望寄托在了尚未出生的儿子身上，哪知道妻子接连给他生了两个女儿，取名吉塔（法缇玛·萨那·纱卡 Fatima Sana Shaikh 饰）和巴比塔（桑亚·玛荷塔 Sanya Malhotra 饰）。让马哈维亚没有想到的是，两个姑娘展现出了杰出的摔跤天赋，让他幡然醒悟，就算是女孩，也能够昂首挺胸的站在比赛场上，为了国家和她们自己赢得荣誉。就这样，在马哈维亚的指导下，吉塔和巴比塔开始了艰苦的训练，两人进步神速，很快就因为在比赛中连连获胜而成为了当地的名人。为了获得更多的机会，吉塔进入了国家体育学院学习，在那里，她将面对更大的诱惑和更多的选择。',
'一晃眼二十年过去。远赴他乡的雷登（伊万·麦克格雷格 Ewan McGregor 饰）再度踏上了爱丁堡的故土。哪知道刚一回来，就撞上了走投无路陷入绝望之中的屎霸（艾文·布莱纳 Ewen Bremner 饰）试图自杀，原来，屎霸不仅婚姻失败还再度染上了毒瘾。在雷登的鼓励下，屎霸决定用书写来治愈内心的伤痕。之后，雷登找到了曾经的挚友病孩（约翰·李·米勒 Jonny Lee Miller 饰），病孩因为曾经的龃龉痛揍了雷登，之后，一个复仇的计划在他心中慢慢成型。在病孩的邀请下，雷登决定和病孩合伙开一家妓院，之后，擅长设计的屎霸亦加入了进来。那边厢，锒铛入狱的贝格比（罗伯特·卡莱尔 Robert Carlyle 饰）越狱成功，在了解了当年事件的真相后，他亦决定找到雷顿向他寻仇。',
'亚马逊公主戴安娜·普林斯（盖尔·加朵 Gal Gadot 饰），经过在家乡天堂岛的训练，取得上帝赐予的武器 与装备，化身神奇女侠，与空军上尉史蒂夫·特雷弗（克里斯·派恩 Chris Pine 饰）一同来到人类世界，一起捍卫和平、拯救世界，在一战期间上演了震撼人心的史诗传奇',
'梁丹妮（热依扎 饰）是演艺圈里冉冉升起的一颗新星，有着持续升温的人气和无量的前途。然而，一场意外的发生让狗仔记者们拍到了她和一个名为李非凡（郑凯 饰）的男子在旅店的亲密照，虽然这一切只是一场误会，但一切澄清和洗白都为时已晚。实际上，梁丹妮已经有了一位交往已久的男友——地产富商石非凡（郭晓冬 饰），为了挽回女友的声誉，石非凡扎到了李非凡，给了后者巨额报酬，要求他成为梁丹妮的“临时男友”，经受不住巨大利益的诱惑，李非凡沦陷了。一边是养尊处优的女明星，一边是低眉顺眼的代驾小哥，这一对性格和身份迥异的男女凑到一起，闹出了无数的矛盾和笑料。',
'几位漂浮在空间站的宇航员们发现了一个来自火星的“神秘样本”，这个样本其实就是他们一直在找寻的高智能“智慧生命体”。它不仅集肌细胞、神经细胞于一体，更拥有超强大脑，能够进行超能进化。但却不知其体内竟然暗藏巨大的杀机，导致整个空间站都弥漫着死亡的气息，人类的命运也危在旦夕。 ',
'在经历过一系列风波后，约翰·威克（基努·里维斯饰）也重新养了一只健康的狗狗。一心想要退休的约翰·威克再度被自己之前的事业所困扰，他此前的雇主现在正在被追杀，受困于自己之前的承诺，约翰·威克只能飞去罗马帮助他脱围。在罗马这个古老的城市里，约翰·威克遇见了全世界最强的杀手，于是乎他只能拿起枪来保护自己。 ',
'一场意外让平凡的厨师顾胜男（周冬雨 饰）结识了霸道总裁路晋（金城武 饰）。之后，路晋入住了顾胜男所在的酒店，在并不知晓对方真实身份的情况下，顾胜男用一道“女巫汤”征服了路晋挑剔的味蕾，一位厨师和一位食客，两人在美食的联结下产生了奇妙的缘分。 路晋的失眠症日益严重，一次偶然中，他发现自己竟然在顾胜男家的沙发上酣然入睡，从此，这张沙发便成为了他的专属床位。随着时间的推移，在一餐一饭之中，顾胜男和路晋之间渐渐产生了懵懂的感情。然而路晋在父亲严苛而又独裁的教育下长大，顾胜男则向来大大咧咧毛手毛脚，个性和身份迥异的他们压抑着内心的感情，由此生出了诸多的误会。'
,
'新年已至，欢乐颂22楼每个人的新问题也接踵而来：安迪（刘涛饰）因包奕凡（杨烁饰）迎来情感的新可能，却也面临来自身世及包家内部带来的新困扰；樊胜美（蒋欣饰）尝试起步新生活，却仍难脱离家庭泥淖，对王柏川（张陆饰）处处依赖事事紧逼；曲筱绡（王子文饰）与赵医生（王凯饰）差距仍存，分和不断，曲家看似稳定的家庭关系实则危机四伏；邱莹莹（杨紫饰）对应勤（吴昊宸饰）一片痴情，情感经历却令应勤无法接受；关雎尔（乔欣饰）邂逅摇滚青年谢滨（邓伦饰）坠入爱河，却遭到父母的激烈反对。 ',
'李雨燃（桂纶镁 饰）是一位优秀的成功律师，过着高质量的白领单身生活，正在有条不紊地准备去海外留学充电，对美好的未来抱着无限的自信与憧憬。然而一次意外，让她发现自己置身于一个奇怪的场所——命运中转站，等待她的是一个意想不到的身份转变：全职太太。她的老公张涛（陈坤 饰 ）是一 位勤恳老实的设计师，女儿星星（欧阳娜娜 饰）是一个叛逆的青春期少女，儿子天天（王元也 饰）是一个天真温暖的男孩。从一位优秀的职业女性突然变成一个要照顾四口之家的家庭主妇的角色，雨燃从一开始强硬的抗拒、不适应到后来慢慢融入，开始用心温柔关爱家人。而当期限来临，转变后的李雨燃又将面临新的选择······ ',
'莲见雄一（市原隼人 饰）与母亲、继父、弟弟生活在一个幽静的小城里。他跟同班同学星野修介（忍成修吾 饰）同在剑道部，两人成为好朋友。星野本来是一个品学兼优的学生，却遭到了同学的嫉妒。当暑假时一群同学自冲绳回来以后，星野变了一个人似的。接着星野开始以欺负同学为乐，而且手段残忍，雄一也不能幸免。雄一性格孤僻，喜欢歌手莉莉周，在自己建立的莉莉周论坛上他认识了一个叫“青猫”的人。雄一与青猫都过着不容易的生活，两人互相鼓励，并约定在莉莉周的演唱会上见面，“青猫”的出现令雄一十分惊讶。',
'匆忙浮躁的都市中，四个与音乐相关的男男女女看似偶然般地邂逅了，他们分别是第一小提琴手卷真纪（松隆子 饰）、大提琴手世吹雀（满岛光 饰）、中提琴手家森谕高（高桥一生 饰）以及第二提琴手别府司（松田龙平 饰）。仿佛是对音乐的共同志向，他们组建了名为“甜甜圈洞”的四重奏乐队，暂时落脚于别府家位于轻井泽的别墅，过起了与世隔绝的人生。然而四个人终究无法超脱世俗存在，除了最基本的吃饭问题，每个人似乎都被各自的秘密所牵扯纠缠。其中雀与真纪“邂逅”的原因，恰恰正是因为真纪丈夫的不辞而别。雪花飘落，寒风萧瑟，四重奏的悠扬旋律荡涤且治愈着他们每一个人的心',
'安娜（达科塔·约翰逊 Dakota Johnson 饰）是一名校报记者，某日，她代替朋友凯特（艾洛斯·慕福特 Eloise Mumford 饰）去采访名为格雷（詹米·多南 Jamie Dornan 饰）的男子。格雷年轻而又英俊，是大财团的董事长，关于他的新闻几乎寥寥无几，这让这位多金的男人显得更加神秘。格雷很快就迷恋上了安娜的聪慧和青涩，安娜亦不自觉的被格雷强大的气场所吸引，随着时间的推移，两人之间产生了感情，然而，让安娜没有想到的是，她发现了格雷极力想要隐藏的惊天秘密，并且成为了格雷的“猎物”。就这样，安娜陷入了格雷一手编织的情网之中，在激情，性感和缠绵之中不可自持，越陷越深',
'服完兵役的高龄大学生恩植（任昌丁饰）就读于法律系，校园里满是对性充满好奇的年轻男女。可惜外表憨直的他没有什么女人缘，脑筋也是不大好使。一次偶然机会，恩植认识了学校健身俱乐部的头牌队员、校花级美女银孝（河智苑饰），他立马被对方的美貌性感吸引。恩植想尽办法接进银孝，博取好感，却都以失败告终，甚至引发许多误会，比如在公车上，他被银孝当成色狼一脚踢中要害。后来，银孝迷住了有钱的花花公子相旭（郑敏饰），甚至怀了孕，但最终被抛弃，伤心绝望。与此同时，恩植和银孝之间产关系有了微妙变化',
'如果，樱花掉落的速度是每秒5厘米，那么两颗心需要多久才能靠近？ 少年时，贵树（水橋研二配）和明理（近藤好美配）是形影不离的好朋友，可很快，一道巨大的鸿沟便横亘在两人中间：明理转学，贵树也随着父母工作的调动搬到遥远的鹿儿岛。在搬家前，贵树乘坐新干线千里迢迢和明理相会，在漫长的等待后，茫茫大雪中，两人在枯萎的樱花树下深情相拥，并献出彼此的first kiss，约定着下一次再一起来看樱花。 时光荏苒，两人竟再没见过，虽然在人海中一直搜寻彼此的身影，但似乎总是徒然。再后来，他们分别有了各自的生活，只是还偶尔会梦到13岁时的这段青涩而美好的感情，才明白当年怎么也说不出口的那个字就是爱。',
'日本神户某个飘雪的冬日，渡边博子（中山美穗）在前未婚夫藤井树的三周年祭日上又一次悲痛到不能自已。正因为无法抑制住对已逝恋人的思念，渡边博子在其中学同学录里发现“藤井树” 在小樽市读书时的地址时，依循着寄发了一封本以为是发往天国的情书。不想不久渡边博子竟然收到署名为“藤井树（中山美穗）”的回信，经过进一步了解，她知晓此藤井树是一个同她年纪相仿的女孩，且还是男友藤井树（柏原崇）少年时代的同班同学。为了多了解一些昔日恋人在中学时代的情况，渡边博子开始与女性藤井树书信往来。而藤井树在不断的回忆中，渐渐发现少年时代与她同名同姓的那个藤井树曾对自己藏了一腔柔情。',
'故事发生在2025年，因为和妻子张代晨（徐静蕾 饰）婚姻破裂，男主角江丰（黄渤 饰）走进记忆大师医疗中心接受手术，却不料手术失误记忆被错误重载，他莫名其妙变成了“杀人凶手”。警官沈汉强（段奕宏 饰）的穷追不舍让他逐渐发现，自己脑内的错误记忆不仅是破案的关键，更是救赎自己的唯一希望。与此同时，妻子身边出现的女人陈姗姗（杨子姗 饰）、记忆中浮现出的神秘女子（许玮甯 饰），似乎也和真相有着千丝万缕的联系，一场记忆烧脑战也随之开始。',
'本片根据印度畅销书作家奇坦·巴哈特（Chetan Bhagat）的处女作小说《五点人》（Five Point Someone）改编而成。法兰（马德哈万 R Madhavan 饰）、拉杜（沙曼·乔希 Sharman Joshi 饰）与兰乔（阿米尔·汗 Aamir Khan 饰）是皇家工程学院的学生，三人共居一室，结为好友。在以严格著称的学院里，兰乔是个非常与众不同的学生，他不死记硬背，甚至还公然顶撞校长“病毒”（波曼·伊拉尼 Boman Irani 饰），质疑他的教学方法。他不仅鼓动法兰与拉杜去勇敢追寻理想，还劝说校长的二女儿碧雅（卡琳娜·卡普 Kareena Kapoor 饰）离开满眼铜臭的未婚夫。兰乔的特立独行引起了模范学生“消音器”（奥米·维嘉 Omi Vaidya 饰）的不满，他约定十年后再与兰乔一决高下，看哪种生活方式更能取得成功',
'BBC曾经制作出《蓝色星球》的纪录片摄影团队，再次集结奉上了这部堪称难以超越的经典纪录片《地球脉动》。从南极到北极，从赤道到寒带，从非洲草原到热带雨林，再从荒凉峰顶到深邃大海，难以数计的生物以极其绝美的身姿呈现在世人面前。我们看到了Okavango洪水的涨落及其周边赖以生存的动物们的生存状态，看到了罕见的雪豹在漫天大雪中猎食的珍贵画面；看到了冰原上企鹅、北极熊、海豹等生物相互依存的严苛情景，也见识了生活在大洋深处火山口高温环境下的惊奇生物。当然还有地球各地的壮观美景与奇特地貌，无私地将其最为光艳的一面展现出来',
'美国青年杰西（伊桑·霍克 Ethan Hawke 饰）在火车上偶遇了法国女学生塞琳娜（朱莉·德尔佩 Julie Delpy 饰），两人在火车上交谈甚欢。当火车到达维也纳时，杰西盛情邀请塞琳娜一起在维也纳游览一番，即使杰西翌日便要坐飞机离开。与杰西一见钟情的塞琳娜接受了杰西的邀请。 他们一边游览城市，一边谈论着彼此的过去 ，彼此对生活的感想，两人了解越加深刻。他们非常珍惜这美妙的晚上，这对恋人一起经历了很多浪漫的经历因为他们约定在半年后再见，而此次约会将会在日出之间结束……'











                  ]
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
json_file = open('model_test_1000_lstm_50.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_test_1000_lstm_50.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
##loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

loaded_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', class_mode="categorical")

classes = loaded_model.predict_classes(xa)
print(classes)

className_cs = [ types_e[className] for className in classes]

test_data_index=0

for className_c in className_cs:
	print(className_c)
	print(xiju_type_list[test_data_index]) 
	test_data_index = test_data_index+1 

##print([ types_e[className] for className in classes])

scores = loaded_model.predict(xa)
print(scores)

