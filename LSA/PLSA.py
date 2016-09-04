import math
import operator
import random
import gzip
import sys
import marshal

def cos_sim(p, q):
    sum0 = sum(map(lambda x:x*x, p))
    sum1 = sum(map(lambda x:x*x, q))
    sum2 = sum(map(lambda x:x[0]*x[1], zip(p, q)))
    return sum2/(sum0**0.5)/(sum1**0.5)

def _rand_mat(sizex, sizey):
    ret = []
    for i in xrange(sizex):
        ret.append([])
        for _ in xrange(sizey):
            ret[-1].append(random.random())
        norm = sum(ret[-1])
        for j in xrange(sizey):
            ret[-1][j] /= norm
    return ret


class Plsa:

    def __init__(self, corpus, topics=2):
        self.topics = topics #主题词个数
        self.corpus = corpus
        self.docs = len(corpus)
        self.each = map(sum, map(lambda x:x.values(), corpus))
        self.words = max(reduce(operator.add, map(lambda x:x.keys(), corpus)))+1
        self.likelihood = 0
        self.zw = _rand_mat(self.topics, self.words) #P(z|w)，这个是训练出来最后需要使用的，看哪个zw的概率最大，朴素贝叶斯思想
        self.dz = _rand_mat(self.docs, self.topics) #P(d|z)
        self.dw_z = None #隐含变量z在当前参数取值条件下的后验概率（P(z|d,w)），在E步骤中计算
        self.p_dw = [] #联合概率
        self.beta = 0.8

    #并不重要的两个
    def save(self, fname, iszip=True):
        d = {}
        for k, v in self.__dict__.items():
            if hasattr(v, '__dict__'):
                d[k] = v.__dict__
            else:
                d[k] = v
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            marshal.dump(d, open(fname, 'wb'))
        else:
            f = gzip.open(fname, 'wb')
            f.write(marshal.dumps(d))
            f.close()

    def load(self, fname, iszip=True):
        if sys.version_info[0] == 3:
            fname = fname + '.3'
        if not iszip:
            d = marshal.load(open(fname, 'rb'))
        else:
            try:
                f = gzip.open(fname, 'rb')
                d = marshal.loads(f.read())
            except IOError:
                f = open(fname, 'rb')
                d = marshal.loads(f.read())
            f.close()
        for k, v in d.items():
            if hasattr(self.__dict__[k], '__dict__'):
                self.__dict__[k].__dict__ = v
            else:
                self.__dict__[k] = v

    def _cal_p_dw(self):
        self.p_dw = [] #每次都要重新计算
        for d in xrange(self.docs):
            self.p_dw.append({}) #加一列
            for w in self.corpus[d]:
                tmp = 0
                for _ in range(self.corpus[d][w]):
                    for z in xrange(self.topics):
                        tmp += (self.zw[z][w]*self.dz[d][z])**self.beta #p_dw根据zw和dz更新
                self.p_dw[-1][w] = tmp

    def _e_step(self):
        self._cal_p_dw() #主要目的是计算dw_z，但之前要先更新目前参数下的p_dw，然后用带概率公式算
        self.dw_z = [] #更新dw_z，求后验
        for d in xrange(self.docs):
            self.dw_z.append({})
            for w in self.corpus[d]:
                self.dw_z[-1][w] = []
                for z in xrange(self.topics):
                    self.dw_z[-1][w].append(((self.zw[z][w]*self.dz[d][z])**self.beta)/self.p_dw[d][w])

    def _m_step(self):
        #zw是最后预测主要使用的
        for z in xrange(self.topics):
            self.zw[z] = [0]*self.words
            for d in xrange(self.docs):
                for w in self.corpus[d]:
                    self.zw[z][w] += self.corpus[d][w]*self.dw_z[d][w][z]
            norm = sum(self.zw[z])
            for w in xrange(self.words):
                self.zw[z][w] /= norm
        #dz由于E步骤需要计算值，也需要求
        for d in xrange(self.docs):
            self.dz[d] = [0]*self.topics
            for z in xrange(self.topics):
                for w in self.corpus[d]:
                    self.dz[d][z] += self.corpus[d][w]*self.dw_z[d][w][z]
            for z in xrange(self.topics):
                self.dz[d][z] /= self.each[d]

    def _cal_likelihood(self):
        self.likelihood = 0
        for d in xrange(self.docs):
            for w in self.corpus[d]:
                self.likelihood += self.corpus[d][w]*math.log(self.p_dw[d][w])

    def train(self, max_iter=100):
        cur = 0
        for i in xrange(max_iter):
            print('%d iter' % i)
            self._e_step()
            self._m_step()
            self._cal_likelihood()
            if cur != 0 and abs((self.likelihood-cur)/cur) < 1e-8: #满足条件
                break
            cur = self.likelihood

    def inference(self, doc, max_iter=100): #预测函数
        doc = dict(filter(lambda x:x[0]<self.words, doc.items()))
        words = sum(doc.values())
        ret = []
        for _ in xrange(self.topics):
            ret.append(random.random())
        norm = sum(ret)
        for i in xrange(self.topics):
            ret[i] /= norm
        tmp = 0
        for _ in xrange(max_iter):
            p_dw = {}
            for w in doc:
                p_dw[w] = 0
                for _ in range(doc[w]):
                    for z in xrange(self.topics):
                        p_dw[w] += (ret[z]*self.zw[z][w])**self.beta
            # e setp
            dw_z = {}
            for w in doc:
                dw_z[w] = []
                for z in xrange(self.topics):
                    dw_z[w].append(((self.zw[z][w]*ret[z])**self.beta)/p_dw[w])
            # m step
            ret = [0]*self.topics #数乘
            for z in xrange(self.topics):
                for w in doc:
                    ret[z] += doc[w]*dw_z[w][z]
            for z in xrange(self.topics):
                ret[z] /= words
            # cal likelihood
            likelihood = 0
            for w in doc:
                likelihood += doc[w]*math.log(p_dw[w])
            if tmp != 0 and abs((likelihood-tmp)/tmp) < 1e-8: #满足条件
                break
            tmp = likelihood
        return ret

#测试
import unittest
class TestPlsa(unittest.TestCase):
    def test_inference(self):
        corpus = [{0:2,3:5},{0:5,2:1},{1:2,4:5}]
        p = Plsa(corpus)
        p.train()
        z = p.inference({0:4, 6:7})
        self.assertTrue(abs(cos_sim(p.dz[0], p.dz[1])-cos_sim(p.dz[0], z))<1e-8)