import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
from matplotlib import font_manager
from itertools import accumulate

import pickle
import numpy as np
from keras.utils import np_utils, plot_model
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df_c=pd.read_csv(r'D:\data\Circ_RNA.csv')
df_c['label'] = 'yes'
df_m=pd.read_csv(r'D:\data\mRNA.csv')
df_m['label'] = 'no'
data = pd.concat([df_c,df_m],axis=0,ignore_index=True)

len_df = data.groupby('length').count()
print(len_df)
sent_length = len_df.index.tolist()
sent_freq = len_df['SequenceID'].tolist()

# 绘制句子长度及出现频数统计图
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
plt.bar(sent_length, sent_freq)
plt.title("sample序列长度及出现频数统计图")
plt.xlabel("RNA序列长度")
plt.ylabel("RNA序列长度出现的频数")
plt.savefig("./sample序列长度及出现频数统计图.png")
plt.close()

sent_pentage_list = [(count/sum(sent_freq)) for count in accumulate(sent_freq)]

# 绘制CDF
plt.plot(sent_length, sent_pentage_list)

# 寻找分位点为quantile的句子长度
quantile = 0.9
#print(list(sent_pentage_list))
for length, per in zip(sent_length, sent_pentage_list):
    if round(per, 2) == quantile:
        index = length
        break
print("\n分位点为%s的句子长度:%d." % (quantile, index))
plt.plot(sent_length, sent_pentage_list)
plt.hlines(quantile, 0, index, colors="c", linestyles="dashed")
plt.vlines(index, 0, quantile, colors="c", linestyles="dashed")
plt.text(0, quantile, str(quantile))
plt.text(index, 0, str(index))
plt.rcParams['font.sans-serif']=['Microsoft YaHei']
plt.rcParams['axes.unicode_minus']=False
plt.title("sample序列长度累积分布函数图")
plt.xlabel("RNA序列长度")
plt.ylabel("RNA序列长度累积频率")
plt.savefig("sample序列长度累积分布函数图.png")
plt.close()

# 标签及词汇表
labels, vocabulary = list(data['label'].unique()), list(data['SequenceID'].unique())

# 构造字符级别的特征
string = ''
for word in vocabulary:
    string += word
vocabulary = set(string)

word_dictionary = {word: i+1 for i, word in enumerate(vocabulary)}
print(word_dictionary)
inverse_word_dictionary = {i+1: word for i, word in enumerate(vocabulary)}
print(inverse_word_dictionary)
label_dictionary = {label: i for i, label in enumerate(labels)}
print(label_dictionary)
output_dictionary = {i: labels for i, labels in enumerate(labels)}

vocab_size = len(word_dictionary.keys()) # 词汇表大小
label_size = len(label_dictionary.keys()) # 标签类别数量
y = [[label_dictionary[sent]] for sent in data['label']]
y = [np_utils.to_categorical(label, num_classes=label_size) for label in y]
y = np.array(list(i[0] for i in y))
x = [[word_dictionary[word] for word in sent] for sent in data['SequenceID']]
x = pad_sequences(maxlen=2000, sequences=x, padding='post', value=0)

model = Sequential()
model.add(Embedding(input_dim=vocab_size + 1, output_dim=20,
                        input_length=2000, mask_zero=True))
model.add(LSTM(125, input_shape=(x.shape[0], x.shape[1])))
model.add(Dropout(0.2))
model.add(Dense(label_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
plot_model(model, to_file='D:\data\model_lstm.png', show_shapes=True)
model.summary()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1, random_state = 42)
model.fit(train_x, train_y, epochs=10, batch_size=32, verbose=1)
N = test_x.shape[0]  # 测试的条数
predict = []
label = []
for start, end in zip(range(0, N, 1), range(1, N+1, 1)):
    sentence = [inverse_word_dictionary[i] for i in test_x[start] if i != 0]
    y_predict = model.predict(test_x[start:end])
    label_predict = output_dictionary[np.argmax(y_predict[0])]
    label_true = output_dictionary[np.argmax(test_y[start:end])]
    print(''.join(sentence), label_true, label_predict) # 输出预测结果
    predict.append(label_predict)
    label.append(label_true)

acc = accuracy_score(predict, label) # 预测准确率
print('模型在测试集上的准确率为: %s.' % acc)
