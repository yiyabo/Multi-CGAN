# def tofasta(path,fasta_path,tag):
#     txt = open(path, 'r')
#     txts = txt.readlines()
#     num = 0
#     with open(fasta_path, 'w')as f:
#
#         for i in txts:
#             num += 1
#             if num > tag:
#                 if num - tag > 100 :
#                     f.write('>' + (str)(num) + '|0|testing' + '\n')
#                 else:
#                     f.write('>' + (str)(num) + '|0|training' + '\n')
#             else:
#                 if tag - num > 100:
#                     f.write('>' + (str)(num) + '|1|training'+'\n')
#                 else:
#                     f.write('>' + (str)(num) + '|1|testing' + '\n')
#
#             f.write(i)
import random
def gettxt(old,new):
    txt = open(old,'r')
    txts = txt.readlines()
    with open(new,'w') as f:
        for i in txts:
            f.write(i.split('\t')[-1])
def tofasta(path,fasta_path,tag,word):
    txt = open(path, 'r')
    txts = txt.readlines()
    num = 0
    num2 = 0

    with open(fasta_path, 'w')as f:

        for i in txts:
            flag = True
            num += 1
            for w in word:
                if w in i:
                    flag = False

            if flag:
                num2+=1
                if num > tag:
                    # if num - tag > 100 :
                    #     f.write('>' + (str)(num) + '|0|testing' + '\n')
                    # else:

                    f.write('>' + (str)(num2) + '|0' + '\n')
                else:
                    # if tag - num > 100:
                    f.write('>' + (str)(num2) + '|1' + '\n')
                    # else:
                    #     f.write('>' + (str)(num) + '|1|testing' + '\n')

                f.write(i)
def ttof(p,n,fa,word):
    ptxt = open(p,'r')
    ptxts = ptxt.readlines()
    ntxt = open(n,'r')
    ntxts = ntxt.readlines()
    num = 0
    with open(fa,'w')as f:
        for i in ptxts:

            flag = True
            # num += 1
            for w in word:
                if w in i:
                    flag = False
            if flag:
                num += 1
                f.write('>' + (str)(num) + '|1' + '\n')
                f.write(i)
        for i in ntxts:

            flag = True
            # num += 1
            for w in word:
                if w in i:
                    flag = False
            if flag:
                num += 1
                f.write('>' + (str)(num) + '|0' + '\n')
                f.write(i)

def inittrain(pos,neg,fasta_path):
    pos_txt = open(pos, 'r',encoding='GB2312')
    pos_txts = pos_txt.readlines()
    neg_txt = open(neg,'r')
    neg_txts = neg_txt.readlines()
    num = 0
    with open(fasta_path,'w') as f:
        for i in pos_txts:
            i = i.strip()
            num += 1
            f.write('>' + (str)(num) + '|1|training' + '\n')
            f.write(i.split('\t')[-1]+'\n')
        for i in neg_txts:
            i = i.strip()
            num += 1
            f.write('>' + (str)(num) + '|0|training' + '\n')
            f.write(i.split('\t')[-1]+'\n')
def inittest(pos,neg,fasta_path,page):
    pos_txt = open(pos, 'r')
    pos_txts = pos_txt.readlines()
    neg_txt = open(neg,'r')
    neg_txts = neg_txt.readlines()
    num = page
    with open(fasta_path,'a+') as f:
        for i in pos_txts:
            i = i.strip()
            num += 1
            f.write('>' + (str)(num) + '|1|testing' + '\n')
            # f.write(i.split('\t')[-1])
            f.write(i+'\n')
        for i in neg_txts:
            i = i.strip()
            num += 1
            f.write('>' + (str)(num) + '|0|testing' + '\n')
            f.write(i.split('\t')[-1]+'\n')

def addtrain(pos,neg,fasta,page):
    pos_txt = open(pos, 'r')
    pos_txts = pos_txt.readlines()
    neg_txt = open(neg, 'r')
    neg_txts = neg_txt.readlines()
    num = page
    with open(fasta,'a+')as f:
        for i in pos_txts:
            i = i.strip()
            num += 1
            f.write('>' + (str)(num) + '|1|training' + '\n')
            # f.write(i.split('\t')[-1]+'\n')
            f.write(i + '\n')
        for i in neg_txts:
            i = i.strip()
            num += 1
            f.write('>' + (str)(num) + '|0|training' + '\n')
            # f.write(i.split('\t')[-1]+'\n')
            f.write(i + '\n')

def addtrain2(pos,neg,fasta,page):
    pos_txt = open(pos, 'r')
    pos_txts = pos_txt.readlines()
    neg_txt = open(neg, 'r')
    neg_txts = neg_txt.readlines()
    num = page
    with open(fasta,'a+')as f:
        for i in pos_txts:
            i = i.strip()
            num += 1
            f.write('>' + (str)(num) + '|1' + '\n')
            # f.write(i.split('\t')[-1]+'\n')
            f.write(i + '\n')
        for i in neg_txts:
            i = i.strip()
            num += 1
            f.write('>' + (str)(num) + '|0' + '\n')
            # f.write(i.split('\t')[-1]+'\n')
            f.write(i + '\n')
def addx(pos,neg,x):
    n = open('newdata/exp4_neg.txt')
    ns = n.readlines()
    p = open('newdata/exp4_pos.txt')
    np = p.readlines()
    genn = random.sample(ns,x)
    genp = random.sample(np,x)
    with open(pos,'w')as f:
        for i in genp:
            f.write(i)

    with open(neg,'w')as f:
        for i in genn:
            f.write(i)

words=['B','J','O','U','X','Z']

# tofasta('toxicity.txt','res2.fasta',1805,words)
# inittrain('yf/t1.txt','yf/t0.txt','yf/data3.txt')
# inittest('yf/v1.txt','yf/v0.txt','yf/data3.txt',3356)
# addx('newdata/500p.txt','newdata/500n.txt',250)
# addx('newdata/1000p.txt','newdata/1000n.txt',500)
# addx('newdata/1500p.txt','newdata/1500n.txt',750)
# addx('newdata/2000p.txt','newdata/2000n.txt',1000)
# addtrain2('newdata/500p.txt','newdata/500n.txt','newdata/nbase_500.txt',3080)
# addtrain2('newdata/1000p.txt','newdata/1000n.txt','newdata/nbase_1000.txt',3080)
# addtrain2('newdata/1500p.txt','newdata/1500n.txt','newdata/nbase_1500.txt',3080)
# addtrain2('newdata/2000p.txt','newdata/2000n.txt','newdata/nbase_2000.txt',3080)
# addtrain('1000_pos.txt','1000_neg.txt','base_14800.fasta',3804)
gettxt('yf/neg.txt','yf/n.txt')
# ttof('newdata/input.txt','newdata/neg.txt','newbase.fasta',words)