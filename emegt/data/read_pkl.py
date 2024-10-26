import pickle

# 读取pkl文件
with open('emegt/data/data/dataexamples/train_block.pkl', 'rb') as file:
    data = pickle.load(file)

# 打印读取的内容
print(data)
