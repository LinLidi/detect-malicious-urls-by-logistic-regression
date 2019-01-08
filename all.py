# coding: utf-8

import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import urllib
import pickle
import html

class urlTest(object):
    def __init__(self):
        good_query_list = self.get_query_list('goodqueries.txt')
        bad_query_list = self.get_query_list('badqueries.txt')

        good_y = [0 for i in range(0,len(good_query_list))]
        bad_y = [1 for i in range(0,len(bad_query_list))]

        queries = bad_query_list+good_query_list
        y = bad_y + good_y

        #TF-IDF特征矩阵
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)
        X = self.vectorizer.fit_transform(queries)
        print('TF-IDF特征矩阵:{}'.format(X))

        #训练逻辑回归模型
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.lgs = LogisticRegression()
        self.lgs.fit(X_train, y_train)
        print('模型的精确度:{}'.format(self.lgs.score(X_test, y_test)))
    
    # 对新的url请求进行预测
    def predict(self,new_queries):
        new_queries = [urllib.parse.unquote(url) for url in new_queries]
        X_predict = self.vectorizer.transform(new_queries)
        res = self.lgs.predict(X_predict)
        for q,r in zip(new_queries,res):
            tmp = '正常请求'if r == 0 else '恶意请求'
            print('{}  {}'.format(q,tmp))
        return res
        

    # 得到文本中的请求列表
    def get_query_list(self,filename):
        directory = str(os.getcwd())
        filepath = directory + "/" + filename
        data = open(filepath,'r').readlines()
        query_list = []
        for d in data:
            d = str(urllib.parse.unquote(d))
            query_list.append(d)
        return list(set(query_list))


    #tokenizer function
    def get_ngrams(self,query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0,len(tempQuery)-3):
            ngrams.append(tempQuery[i:i+3])
        #print(ngrams)
        return ngrams

if __name__ == '__main__':
    if not os.isfile(os.path.join(os.getcwd(), 'lgs.pickle')):
        print('the model is not founded, starting trainning.')
        w = urlTest()
        with open('lgs.pickle','wb') as output:
            wpickle.dump(w,output)
    else:
        with open('lgs.pickle','rb') as input:
            w = pickle.load(input)

    w.predict(['qq.com','google/images','<script>alert(1)</script>',
    'wp-content/wp-pluginswp-content/wp-plugins','example/test/q=<script>alert(1)</script>','q=../etc/passwd'])
    #'www.foo.com/name=admin\' or 1=1', 'abc.com/admin.php',
    #'"><svg onload=confirm(1)>', 'test/q=<a href="javascript:confirm(1)>', 'q=../etc/passwd']

