class LSA(object):
    def __init__(self, stopwords, ignorechars):
        self.stopwords =stopwords #停用词表
        self.ignorechars =ignorechars
        self.wdict ={}
        self.dcount =0

    def parse(self, doc):
        words = segmentation(doc) #使用segmentation函数进行分词，标点会被过滤完毕
        for w in words:
            if w in self.stopwords: #无视停用词
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount) #把一个词出现的文档号填入到wdict中去
            else:
                self.wdict[w] = [self.dcount]
                self.dcount += 1

    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1] #所有出现的词被取出
        self.keys.sort() #排序
        self.A = zeros([len(self.keys), self.dcount]) #建立一个矩阵，其行数是词的个数，列数是文档个数
        #所有的词和文档对所对应的矩阵单元的值被统计出来
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1 #创建出的矩阵