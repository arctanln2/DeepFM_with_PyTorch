test = open("./test.txt",'w') #新建一个存放测试结果的文件并打开
train = open("./train.txt",'w') # 新建一个存放训练结果的文件并打开
with open("./sample.txt") as f:  #打开需要处理的文件
    lines = f.readlines()           #将总文件的行全部读到lines里面
    for i in range(lines.__len__()):  #遍历lines
        if i >= lines.__len__()*0.9:                  #把双数行存到test里面，单数行存到train里面
            test.write(lines[i])
        else:                         
            train.write(lines[i])
test.close()     #因为test，和train没用with open，用完后要记得close
train.close()
