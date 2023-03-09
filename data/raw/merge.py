result = "train_test.txt"
# 打开当前目录下的result.txt文件，如果没有则创建
file = open(result, 'w+', encoding="utf-8")
# 向文件中写入字符
 
# 先遍历文件名
for i in range(10):
    filename="test.txt"
    # 遍历单个文件，读取行数
    for line in open(filename, encoding="utf-8"):
        file.writelines(line)
# 关闭文件
file.close()
