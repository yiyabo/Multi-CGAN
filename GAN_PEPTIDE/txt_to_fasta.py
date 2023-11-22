#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: yhq
@contact:1318231697@qq.com
@version: 1.0.0
@file: txt_to_fasta.py
@time: 2021/10/6 15:16
"""


def save_fasta(txt_path,fasta_path):
    txt=open(txt_path,'r')
    txts=txt.readlines()
    num=0
    with open(fasta_path,'w')as f:
        for i in txts:
            num+=1
            f.write('>sequence'+(str)(num)+'\n')
            f.write(i)

save_fasta('data/gen_fliter3.txt','data/gen_cwgan_data3.fasta')