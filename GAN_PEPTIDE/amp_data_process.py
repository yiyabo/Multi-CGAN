#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yhq
@contact:1318231697@qq.com
@version: 1.0.0
@file: amp_data_process.py
@time: 2021/9/14 16:36
"""
import csv
import pandas as pd
import numpy as np
from glob import glob
import torch.nn.functional as F
import torch

def create_dict(words):
    word_dict={}
    word_index=0
    for word in words:
        word_dict[word]=word_index
        word_index+=1
    return word_dict
def creat_dict_word(word_dict,words):
    word_num_dict={}
    for word in words:
        word_num_dict[word_dict[word]]=word
    return word_num_dict
def tsv_to_vector(tsv_path,word_dict):
    max=0
    vectors=[]
    # num=0
    all_data=pd.read_csv(tsv_path,sep="\t",header=0,index_col='index')
    # print(all_data['sequence'].size)
    for i in range((all_data['sequence'].size)):
        sequence_vector=''
        sequence = all_data['sequence'][i]
        # print(type(sequence))
        if len(sequence)>max:
            max=len(sequence)
        for char in sequence:
            sequence_vector+=str(word_dict[char])+' '
        vectors.append(sequence_vector)
    return vectors,max
def tsv_to_vector_max(tsv_path,word_dict,max=30):
    max2=0
    vectors=[]
    # num=0
    all_data=pd.read_csv(tsv_path,sep="\t",header=0,index_col='index')
    # print(all_data['sequence'].size)
    for i in range((all_data['sequence'].size)):
        sequence_vector=''
        sequence = all_data['sequence'][i]
        # print(type(sequence))
        if len(sequence)>max2:
            max2=len(sequence)
        for char in sequence:
            sequence_vector+=str(word_dict[char])+' '
        if len(sequence)<=max:
            vectors.append(sequence_vector)
    return vectors
def save_data(vector,max,save_path,padding):
    # print(len(vector))
    data_numpy=np.zeros((len(vector),max),dtype=int)
    for i in range(len(vector)):
        vector_list=vector[i].split(' ')
        num =0
        # print(vector_list[:-1])
        for j in vector_list[:-1]:
            # print(i)
            data_numpy[i][num]=(int)(j)
            num+=1
            if num==30:
                break
        while num<max:

            data_numpy[i][num]=(int)(padding)
            num+=1
    print(11111111111111111111)
    np.save(save_path,data_numpy)
def save_gen_vector(gen_path,new_path,max,padding):
    # word_num = creat_dict_word(word_dict, words)
    convert = np.load(gen_path)
    # print(convert)
    convert = convert.tolist()
    filter_gen=[]
    # //WGAN生成数据时有用
    # old1 = np.load('WGAN/seq.npy')
    # old2 = np.load('WGAN/laten_codes.npy')
    # new1=[]
    # new2=[]
    # convert=str(convert)
    # print(word_num[10])
    # print(convert)
    count=0
    num = 0
    for i in range(len(convert)):
        # isGen = True
        num += 1
        flag=-1
        pep=[]
        for j in range(max):
            # print(i,j)
            index = (convert[i][j])
            # print(index)
            # print(index)
            # convert[i][j] = word_num[index]
            if index==padding:
                continue
                # print(i,convert[i][j+1],isGen)
                flag=j
            else:
                pep.append(index)

        while len(pep) < max:
            pep.append(20)

            # if isGen:
            #     if index ==padding:
            #         count+=1
        # if isGen:
            # print(convert[i])
        filter_gen.append((pep))
            # new1.append(old1[i])
            # new2.append(old2[i])
        # else:
        #     print('no:'+ (str)(num))

    # print(filter_gen,count)
    # np.save('WGAN/seq.npy',new1)
    # np.save('WGAN/laten_codes.npy', new2)
    np.save(new_path,filter_gen)

# def save_gen_vector(gen_path,new_path,max,padding):
#     # word_num = creat_dict_word(word_dict, words)
#     convert = np.load(gen_path)
#     # print(convert)
#     convert = convert.tolist()
#     filter_gen=[]
#     # //WGAN生成数据时有用
#     # old1 = np.load('WGAN/seq.npy')
#     # old2 = np.load('WGAN/laten_codes.npy')
#     # new1=[]
#     # new2=[]
#     # convert=str(convert)
#     # print(word_num[10])
#     # print(convert)
#     count=0
#     num = 0
#     for i in range(len(convert)):
#         isGen = True
#         num += 1
#         flag=-1
#         for j in range(max):
#             # print(i,j)
#             index = (convert[i][j])
#             # print(index)
#             # convert[i][j] = word_num[index]
#             if index==padding and j<=max-2:
#                 if j < 4:
#                     isGen = False
#                 if convert[i][j+1]==padding:
#                     pass
#
#                 else:
#                     isGen = False
#                 # print(i,convert[i][j+1],isGen)
#                 flag=j
#
#
#
#             # if isGen:
#             #     if index ==padding:
#             #         count+=1
#         if isGen:
#             # print(convert[i])
#             filter_gen.append(convert[i])
#             # new1.append(old1[i])
#             # new2.append(old2[i])
#         else:
#             print('no:'+ (str)(num))
#
#     # print(filter_gen,count)
#     # np.save('WGAN/seq.npy',new1)
#     # np.save('WGAN/laten_codes.npy', new2)
#     np.save(new_path,filter_gen)
#
# def save_gen_vector(gen_path,new_path,max,padding):
#     # word_num = creat_dict_word(word_dict, words)
#     convert = np.load(gen_path)
#     # print(convert)
#     convert = convert.tolist()
#     filter_gen=[]
#     # //WGAN生成数据时有用
#     old1 = np.load('CWGAN/exp6/seq_a.npy')
#     old2 = np.load('CWGAN/exp6/laten_codes.npy')
#     new1=[]
#     new2=[]
#     # convert=str(convert)
#     # print(word_num[10])
#     # print(convert)
#     count=0
#     num = 0
#     for i in range(len(convert)):
#         isGen = True
#         num += 1
#         flag=-1
#         for j in range(max):
#             # print(convert[i][j])
#             index = (convert[i][j])
#             # print(index)
#             # convert[i][j] = word_num[index]
#             if index==padding and j<=max-2:
#                 if j < 4:
#                     isGen = False
#                 if convert[i][j+1]==padding:
#                     pass
#
#                 else:
#                     isGen = False
#                 # print(i,convert[i][j+1],isGen)
#                 flag=j
#
#
#
#             # if isGen:
#             #     if index ==padding:
#             #         count+=1
#         if isGen:
#             # print(convert[i])
#             filter_gen.append(convert[i])
#             new1.append(old1[i])
#             new2.append(old2[i])
#         else:
#             print('no:'+ (str)(num))
#
#     # print(filter_gen,count)
#     np.save('CWGAN/exp6/seq_a.npy',new1)
#     np.save('CWGAN/exp6/laten_codes.npy', new2)
#     np.save(new_path,filter_gen)
def filter_gen(word_dict,words,gen_path,max):
    word_num = creat_dict_word(word_dict, words)
    convert = np.load(gen_path,allow_pickle=True)
    convert = convert.tolist()
    # print(convert)
    filter_gen=[]

    # convert=str(convert)
    # print(word_num[10])
    # print(convert)
    # count=0
    for i in range(len(convert)):
        # isGen = True
        # flag=-1
        for j in range(max):
            # print(i,j)
            index = (convert[i][j])
            # print(index)
            convert[i][j] = word_num[index]
            # if index==26 and j!=29:
            #     if flag==-1:
            #         pass
            #     if flag!=-1 and flag+1==j:
            #         pass
            #     else:
            #         isGen = False
            #     flag=j
            #
            #
            #
            # if isGen:
            #     if index ==26:
            #         count+=1
        filter_gen.append(convert[i])
        # if (convert[i][0]) != 20 and convert[i][1] != 20:
        #     filter_gen.append(convert[i])

    # print(filter_gen)
    return filter_gen
def save_train(word_dict,save_path='AMP_dataset/training.tsv',path='data/train_AMPdata.npy',padding=20):
    vector, max = tsv_to_vector(save_path, word_dict)
    # print('starting save...')
    # print(len(vector))
    save_data(vector, 30, path,padding)
    # print('save successfully!')

def save_gen(word_dict,save_path='AMP_dataset/neg.tsv',path='data/gen_AMPdata.npy',padding=20,max=30):
    vector, _ = tsv_to_vector(save_path, word_dict)
    # print('starting save...')
    save_data(vector, max, path,padding)
    # print('save successfully!')
def save_in_txt(npy,txt_path,is_a=False):
    txt=[]
    for i in npy:
        # print(i)
        string=''
        for char in i:
            string+=char
        txt.append(string)
    if is_a:
        with open(txt_path, 'a+') as f:
            for line in txt:
                f.write(line + '\n')
            f.close()
    else:
        with open(txt_path, 'w') as f:
            for line in txt:
                f.write(line + '\n')
            f.close()
    return txt
def save_all_txt(npy_paths,txt_path,word_dict,words):
    # npy files (train procrss generates)-> txt file
    files=sorted(glob(npy_paths+'/*'))
    total=[]
    for file in files:
        save_gen_vector(file,'data/gen_fliter2.npy',30,20)
        filter=filter_gen(word_dict,words,'data/gen_fliter2.npy',30)
        # total.append(filter)
        save_in_txt(filter,txt_path,True)
def return_index(one_hot_coding):
    # one_hot = one_hot_coding.numpy()
    index = np.argwhere(one_hot_coding == 1)
    return index[:, -1].reshape(-1, 30)
#
words=['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y',' ']

word_dict=create_dict(words)
# print(word_dict)
# save_train(word_dict)
# save_gen(word_dict)
# a=np.load('data/train_AMPdata.npy')
# a=a.tolist()
# print(len(a))
#

# file = open('CWGAN/data/amp.txt','r')
# lines = file.readlines()
# datas = []
# for line in lines:
#     line = line.strip('\n')
#     if len(line) >30:
#         continue
#     data=[]
#     for c in range(0,30):
#         if c<len(line) and line[c] not in word_dict:
#             break
#         if c >= len(line):
#             data.append(20)
#         else:
#             data.append(word_dict[line[c]])
#     if len(data) == 30:
#
#         datas.append(data)
# np.save('CWGAN/data/amp_pos.npy',datas)
# a = np.load('CWGAN/data/amp_pos.npy', allow_pickle=True)
# print(a)

# print(1)
###########################################################################################
# wgan_index = np.load('CWGAN/data/struct_amp.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
# save_gen_vector('CWGAN/data/struct_amp.npy','CWGAN/data/amp_posf.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/data/amp_posf.npy',30)
# filter_txt=save_in_txt(fliter,'SS_score/bothpos.txt')
# print(filter_txt)
# wgan_index = np.load('CWGAN/data/amp_neg.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/gen_wgan_data.npy',return_index(wgan_index))
# save_gen_vector('CWGAN/data/amp_neg.npy','CWGAN/data/amp_negf.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/data/amp_negf.npy',30)
# filter_txt=save_in_txt(fliter,'SS_score/amp_neg.txt')
# print(filter_txt)
#sticity_neg jiushi neg t_a yeshi neg
# wgan_index = np.load('CWGAN/exp6/exp6_st.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/exp5/exp52b.npyy',return_index(wgan_index))
# save_gen_vector('CWGAN/exp6/exp6_st.npy','CWGAN/exp6/exp6_stf.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/exp6/exp6_stf.npy',30)
# filter_txt=save_in_txt(fliter, 'CWGAN/exp6/exp6_st.txt')
# print(filter_txt)
# wgan_index = np.load('CWGAN/exp6/gen_wgan_edit.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/exp5/exp52b.npyy',return_index(wgan_index))
# save_gen_vector('CWGAN/exp6/gen_wgan_edit.npy','CWGAN/exp6/gen_wgan_editf.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/exp6/gen_wgan_editf.npy',30)
# filter_txt=save_in_txt(fliter, 'CWGAN/exp6/gen_wgan_edit.txt')
# print(filter_txt)
# wgan_index = np.load('CWGAN/exp6/gen.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/exp5/exp52b.npyy',return_index(wgan_index))
# save_gen_vector('CWGAN/exp6/gen.npy','CWGAN/exp6/genf.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/exp6/genf.npy',30)
# filter_txt=save_in_txt(fliter, 'CWGAN/exp6/gen.txt')
# print(filter_txt)
# wgan_index = np.load('CWGAN/exp5/new1.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/gen_wgan_data.npy',return_index(wgan_index))
# save_gen_vector('CWGAN/exp5/new1.npy','CWGAN/exp5/new1f.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/exp5/new1f.npy',30)
# filter_txt=save_in_txt(fliter,'CWGAN/exp5/new1.txt')
# print(filter_txt)
# wgan_index = np.load('CWGAN/exp5/new2.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/gen_wgan_data.npy',return_index(wgan_index))
# save_gen_vector('CWGAN/exp5/new2.npy','CWGAN/exp5/new2f.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/exp5/new2f.npy',30)
# filter_txt=save_in_txt(fliter,'CWGAN/exp5/new2.txt')
# print(filter_txt)
# wgan_index = np.load('CWGAN/exp5/new3.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/gen_wgan_data.npy',return_index(wgan_index))
# save_gen_vector('CWGAN/exp5/new3.npy','CWGAN/exp5/new3f.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/exp5/new3f.npy',30)
# filter_txt=save_in_txt(fliter,'CWGAN/exp5/new3.txt')
# print(filter_txt)
wgan_index = np.load('CWGAN/new/at1.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
save_gen_vector('CWGAN/new/amp1.npy','CWGAN/new/at1f.npy',30,20)
fliter=filter_gen(word_dict,words,'CWGAN/new/at1f.npy',30)
filter_txt=save_in_txt(fliter,'CWGAN/new/at1.txt')
print(filter_txt)

def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            i = i.strip()
            num+=1
            f.write('>seq'+(str)(num)+'\n')
            f.write(i+'\n')

# save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
save_fasta('CWGAN/new/at1.txt','CWGAN/new/at1.fasta')
#


wgan_index = np.load('CWGAN/new/at2.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
save_gen_vector('CWGAN/new/amp2.npy','CWGAN/new/at2f.npy',30,20)
fliter=filter_gen(word_dict,words,'CWGAN/new/at2f.npy',30)
filter_txt=save_in_txt(fliter,'CWGAN/new/at2.txt')
print(filter_txt)

def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            i = i.strip()
            num+=1
            f.write('>seq'+(str)(num)+'\n')
            f.write(i+'\n')

# save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
save_fasta('CWGAN/new/at2.txt','CWGAN/new/at2.fasta')

wgan_index = np.load('CWGAN/new/at3.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
save_gen_vector('CWGAN/new/at3.npy','CWGAN/new/at3f.npy',30,20)
fliter=filter_gen(word_dict,words,'CWGAN/new/at3f.npy',30)
filter_txt=save_in_txt(fliter,'CWGAN/new/at3.txt')
print(filter_txt)

def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            i = i.strip()
            num+=1
            f.write('>seq'+(str)(num)+'\n')
            f.write(i+'\n')

# save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
save_fasta('CWGAN/new/at3.txt','CWGAN/new/at3.fasta')

wgan_index = np.load('CWGAN/new/st1.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
save_gen_vector('CWGAN/new/st1.npy','CWGAN/new/st1f.npy',30,20)
fliter=filter_gen(word_dict,words,'CWGAN/new/st1f.npy',30)
filter_txt=save_in_txt(fliter,'CWGAN/new/st1.txt')
print(filter_txt)

def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            i = i.strip()
            num+=1
            f.write('>seq'+(str)(num)+'\n')
            f.write(i+'\n')

# save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
save_fasta('CWGAN/new/st1.txt','CWGAN/new/st1.fasta')

wgan_index = np.load('CWGAN/new/st2.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
save_gen_vector('CWGAN/new/st2.npy','CWGAN/new/st2f.npy',30,20)
fliter=filter_gen(word_dict,words,'CWGAN/new/st2f.npy',30)
filter_txt=save_in_txt(fliter,'CWGAN/new/st2.txt')
print(filter_txt)

def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            i = i.strip()
            num+=1
            f.write('>seq'+(str)(num)+'\n')
            f.write(i+'\n')

# save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
save_fasta('CWGAN/new/st2.txt','CWGAN/new/st2.fasta')

wgan_index = np.load('CWGAN/new/st3.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
save_gen_vector('CWGAN/new/st3.npy','CWGAN/new/st3f.npy',30,20)
fliter=filter_gen(word_dict,words,'CWGAN/new/st3f.npy',30)
filter_txt=save_in_txt(fliter,'CWGAN/new/st3.txt')
print(filter_txt)

def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            i = i.strip()
            num+=1
            f.write('>seq'+(str)(num)+'\n')
            f.write(i+'\n')

# save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
save_fasta('CWGAN/new/st3.txt','CWGAN/new/st3.fasta')

wgan_index = np.load('CWGAN/new/as1.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
save_gen_vector('CWGAN/new/s1.npy','CWGAN/new/as1f.npy',30,20)
fliter=filter_gen(word_dict,words,'CWGAN/new/as1f.npy',30)
filter_txt=save_in_txt(fliter,'CWGAN/new/as1.txt')
print(filter_txt)

def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            i = i.strip()
            num+=1
            f.write('>seq'+(str)(num)+'\n')
            f.write(i+'\n')

# save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
save_fasta('CWGAN/new/as1.txt','CWGAN/new/as1.fasta')

wgan_index = np.load('CWGAN/new/as2.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
save_gen_vector('CWGAN/new/as2.npy','CWGAN/new/as2f.npy',30,20)
fliter=filter_gen(word_dict,words,'CWGAN/new/as2f.npy',30)
filter_txt=save_in_txt(fliter,'CWGAN/new/as2.txt')
print(filter_txt)

def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            i = i.strip()
            num+=1
            f.write('>seq'+(str)(num)+'\n')
            f.write(i+'\n')

# save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
save_fasta('CWGAN/new/as2.txt','CWGAN/new/as2.fasta')

wgan_index = np.load('CWGAN/new/as3.npy')
# temp = []
# for i  in range(wgan_index):
#     for j in wgan_index[i]:
#         temp.append(return_index(j))
# np.save('data/gen_wgan_data.npy',return_index(wgan_index))
save_gen_vector('CWGAN/new/as3.npy','CWGAN/new/as3f.npy',30,20)
fliter=filter_gen(word_dict,words,'CWGAN/new/as3f.npy',30)
filter_txt=save_in_txt(fliter,'CWGAN/new/as3.txt')
print(filter_txt)

def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            i = i.strip()
            num+=1
            f.write('>seq'+(str)(num)+'\n')
            f.write(i+'\n')

# save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
save_fasta('CWGAN/new/as3.txt','CWGAN/new/as3.fasta')

# wgan_index = np.load('CWGAN/new/all1.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/gen_wgan_data.npy',return_index(wgan_index))
# save_gen_vector('CWGAN/new/all1.npy','CWGAN/new/all1f.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/new/all1f.npy',30)
# filter_txt=save_in_txt(fliter,'CWGAN/new/all1.txt')
# print(filter_txt)
#
# def save_fasta(txt_path,fasta_path):
#     txt=open(txt_path,'r')
#     txts=txt.readlines()
#     num=0
#     with open(fasta_path,'w')as f:
#         for i in txts:
#             i = i.strip()
#             num+=1
#             f.write('>seq'+(str)(num)+'\n')
#             f.write(i+'\n')
#
# # save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# # save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# # save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# # save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# # save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
# save_fasta('CWGAN/new/all1.txt','CWGAN/new/all1.fasta')
#
# wgan_index = np.load('CWGAN/new/all3.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/gen_wgan_data.npy',return_index(wgan_index))
# save_gen_vector('CWGAN/new/all3.npy','CWGAN/new/all3f.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/new/all3f.npy',30)
# filter_txt=save_in_txt(fliter,'CWGAN/new/all3.txt')
# print(filter_txt)
#
# def save_fasta(txt_path,fasta_path):
#     txt=open(txt_path,'r')
#     txts=txt.readlines()
#     num=0
#     with open(fasta_path,'w')as f:
#         for i in txts:
#             i = i.strip()
#             num+=1
#             f.write('>seq'+(str)(num)+'\n')
#             f.write(i+'\n')
#
# # save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# # save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# # save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# # save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# # save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
# save_fasta('CWGAN/new/all3.txt','CWGAN/new/all3.fasta')
#
# wgan_index = np.load('CWGAN/new/all2.npy')
# # temp = []
# # for i  in range(wgan_index):
# #     for j in wgan_index[i]:
# #         temp.append(return_index(j))
# # np.save('data/gen_wgan_data.npy',return_index(wgan_index))
# save_gen_vector('CWGAN/new/all2.npy','CWGAN/new/all2f.npy',30,20)
# fliter=filter_gen(word_dict,words,'CWGAN/new/all2f.npy',30)
# filter_txt=save_in_txt(fliter,'CWGAN/new/all2.txt')
# print(filter_txt)
#
# def save_fasta(txt_path,fasta_path):
#     txt=open(txt_path,'r')
#     txts=txt.readlines()
#     num=0
#     with open(fasta_path,'w')as f:
#         for i in txts:
#             i = i.strip()
#             num+=1
#             f.write('>seq'+(str)(num)+'\n')
#             f.write(i+'\n')
#
# # save_fasta('CWGAN/exp6/gen.txt','CWGAN/exp6/gen.fasta')
# # save_fasta('CWGAN/exp6/gen_wgan_edit.txt','CWGAN/exp6/gen_wgan_edit.fasta')
# # save_fasta('CWGAN/exp5/new1.txt','CWGAN/exp5/new1.fasta')
# # save_fasta('CWGAN/exp5/new2.txt','CWGAN/exp5/new2.fasta')
# # save_fasta('CWGAN/exp5/new3.txt','CWGAN/exp5/new3.fasta')
# save_fasta('CWGAN/new/all2.txt','CWGAN/new/all2.fasta')
# save_all_txt('data/generate','data/gen_total.txt',word_dict,words)
# a,b=tsv_to_vector('./AMP_dataset/test.tsv',word_dict)
# print(b)


# pep=np.load('data/gen_fliter.npy')
# # print(pep)
# pep=torch.LongTensor(pep)
# # print(pep[0])
# one_hot_pep=F.one_hot(pep,21)
# # print(one_hot_pep)
# one_hot=one_hot_pep.numpy()
# index=np.argwhere(one_hot==1)
# index=index[:,-1].reshape(-1,30)
# print(index.shape)