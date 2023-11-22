def save_txt(fasta_path,txt_path):
    txt=open(fasta_path,'r')
    txts=txt.readlines()
    num=0
    with open(txt_path,'w')as f:
        for i in txts:
            # flag = True
            num+=1
            if num % 2 == 1:
                i = i.strip()
                flag = i.split('|')[1] == '0' and i.split('|')[-1] == 'training'
                # if not(i.split('|')[1] == '0' and i.split('|')[-1] == 'training'):
                #     # print(i.split('|')[1] ,i.split('|')[-1])
                #     # print(i.split('|')[1] == '0',i.split('|')[-1] == 'training')
                #     flag = False
            if num % 2 == 0 and flag:
                print(num)
                f.write(i)
def save_txt2(fasta_path,txt_path):
    txt = open(fasta_path, 'r')
    txts = txt.readlines()
    num = 0
    with open(txt_path, 'w')as f:
        for i in txts:
            # flag = True
            num += 1
            # if num % 2 == 1:
            #     i = i.strip()
            #     flag = i.split('|')[1] == '0' and i.split('|')[-1] == 'training'
            #     # if not(i.split('|')[1] == '0' and i.split('|')[-1] == 'training'):
            #     #     # print(i.split('|')[1] ,i.split('|')[-1])
            #     #     # print(i.split('|')[1] == '0',i.split('|')[-1] == 'training')
                #     flag = False
            if num % 2 == 0 and len(i) <= 30:
                print(num)
                f.write(i)
def returnmax(fasta_path):
    txt = open(fasta_path, 'r')
    txts = txt.readlines()
    num = 0
    max = 0
    for i in txts:
        # flag = True
        num += 1
        if num % 2 == 0:
            print(len(i))
            if max < len(i):
                max = len(i)
    return max
def countmax(fasta_path,max):
    txt = open(fasta_path, 'r')
    txts = txt.readlines()
    num = 0
    # max = 0
    ans=0
    total=0
    for i in txts:
        # # flag = True
        # num += 1
        # if num % 2 == 1:
        #     i = i.strip()
        #     flag = i.split('|')[-1] == 'training'
        # if num % 2 == 0 and flag:
        #     # print(len(i))
        #     total += 1
        #     if max < len(i):
        #         ans +=1
        total += 1
        if max < len(i):
            ans += 1
    print(ans,total)
#
save_txt('newdata/base.fasta','newdata/neg.txt')
# print(returnmax('newbase.txt'))
# countmax('aneg.txt',50)
# save_txt2('newdata/NAMP_tr.fa','newdata/neg2.txt')
# countmax('newdata/pos2.txt',returnmax('newdata/pos2.txt'))