

def getres(data_path):
    f = open(data_path,'r')
    datas = f.readlines()
    res = []
    for i in range(len(datas)):
        if datas[i].split(" ")[0] == 'Query=':
            i += 6
            d = datas[i].strip('\n')
            d = ' '.join(d.split())
            e =(d.split(" "))
            if len(e) == 6:
                key = (float)(d.split(" ")[-2])
                value = (float)(d.split(" ")[-1])
                res.append((key,value))
        else:
            pass
    return res
def cal(res,max,min):
    total = 0
    # count = 0

    for i in res:
        # if min == -1:
        #     if i[1] <= max:
        #         total
        if i[1] > min and i[1] <=max:
            total+=1
            # count+=i1
    return total / len(res)
def cal2(res,max):
    total = 0
    # count = 0
    for i in res:
        if i[1] >= max :
            total+=1
            # count+=i1
    return total / len(res)
def cal_score(res):
    total = 0
    # count = 0
    tot = 0
    for i in res:
        # if min == -1:
        #     if i[1] <= max:
        #         total
        if i[0] > 20:
            total += 1
            # count+=i1
            tot += i[1]
    return tot / total
res = getres('result.txt')
# print(cal(res,0.001,0))
# print(res)
print(cal_score(res))