# -*- 本脚本用于训练中文自动文本摘要模型 Created by Songyujian 2023.8.7-*-
import numpy as np
import pandas as pd
import jieba_fast
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from tensorflow.python import keras
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional
from keras.models import Model
from keras.callbacks import EarlyStopping
from attention import AttentionLayer
import warnings
from sklearn.model_selection import train_test_split
import visdom
import copy
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
# 读取数据集
data = pd.read_csv("./testdata/cndata.csv", nrows=20000)
data.drop_duplicates(subset=['Text'], inplace=True)  # 去除重复值
data.dropna(axis=0, inplace=True)  # 去除NA值


# 判断是否是中文
def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False

# 保留中文字符
def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
        else:
            pass
    return content_str

# 分词
def divide(datas):
    jieba_fast.setLogLevel(jieba_fast.logging.INFO)
    cut_words = list(jieba_fast.lcut(str(datas)[2:len(str(datas)) - 2]))
    return cut_words

# 去除停用词
def drop_stopwords(contents, stopwords):
    contents_clean = []
    for line in contents:
        line_clean = []
        for word in line:
            if word not in stopwords:
                line_clean.append(word)
        if line_clean == []:
            pass
        else:
            contents_clean.append(''.join(line_clean))
    return contents_clean

# 文本预处理
def text_cleaner(text):
    newString = list(map(lambda s: s.replace(' ', ''), text))
    chinese_list = []
    chinese_list.append(format_str(newString))
    tokens = copy.deepcopy(divide(chinese_list))
    path = "./testdata/stopwords.txt"
    stopwords = [line.strip() for line in open(path, encoding="utf-8").readlines()]
    tokens = copy.deepcopy(drop_stopwords(tokens, stopwords))
    long_words = []
    for i in tokens:
        if len(i) >= 1:  # 去除短文本
            long_words.append(i)
    return (" ".join(long_words)).strip("[' ']")

# 预处理后的文本
cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t))


# print(cleaned_text[0])
# 摘要字符保留
def is_summary(uchar):
    if (u'\u4e00' <= uchar <= u'\u9fa5') or (u'\u0030' <= uchar <= u'\u0039') or uchar == '.':
        return True
    else:
        return False

# 保留符合摘要要求的字符串
def format_sum(content):
    content_str = ''
    for i in content:
        if is_summary(i):
            content_str = content_str + i
        else:
            pass
    return content_str

# 摘要预处理
def summary_cleaner(text):
    newString = list(map(lambda s: s.replace(' ', ''), text))
    chinese_list = []
    chinese_list.append(format_sum(newString))
    tokens = copy.deepcopy(divide(chinese_list))
    long_words = ''
    for i in tokens:
        if len(i) >= 1:  # 去除短文本
            long_words = long_words + i
    return long_words

# 预处理后的摘要
cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(summary_cleaner(t))

data['cleaned_text'] = cleaned_text
data['cleaned_summary'] = cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0, inplace=True)
# 摘要的开头和结尾添加START和END特殊标记
data['cleaned_summary'] = data['cleaned_summary'].apply(lambda x: '_START_ ' + x + ' _END_')

# for i in range(5):
#      print("Review:",data['cleaned_text'][i])
#      print("Summary:",data['cleaned_summary'][i])
#      print("\n")

# 绘制统计图
viz = visdom.Visdom()
text_word_count = []
summary_word_count = []
# populate the lists with sentence lengths
for i in data['cleaned_text']:
    text_word_count.append(len(i.split()))

for i in data['cleaned_summary']:
    summary_word_count.append(len(i.split()[1]))

viz.histogram(
    X = text_word_count,
    opts=dict(
        title = "评论长度",
        numbins = 30
    )
)

viz.histogram(
    X = summary_word_count,
    opts=dict(
        title = "摘要长度",
        numbins = 30
    )
)

# 确立最大评论长度和最大摘要长度
max_len_text = 1000
max_len_summary = 100
# 数据集拆分
x_tr, x_val, y_tr, y_val = train_test_split(data['cleaned_text'], data['cleaned_summary'], test_size=0.2,
                                            random_state=0, shuffle=True)
# prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

# convert text sequences into integer sequences
x_tr = x_tokenizer.texts_to_sequences(x_tr)
x_val = x_tokenizer.texts_to_sequences(x_val)

# padding zero upto maximum length
x_tr = pad_sequences(x_tr, maxlen=max_len_text, padding='post')
x_val = pad_sequences(x_val, maxlen=max_len_text, padding='post')

x_voc_size = len(x_tokenizer.word_index) + 1

# preparing a tokenizer for summary on training data
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))
# convert summary sequences into integer sequences
y_tr = y_tokenizer.texts_to_sequences(y_tr)
y_val = y_tokenizer.texts_to_sequences(y_val)
# padding zero upto maximum length

y_tr = pad_sequences(y_tr, maxlen=max_len_summary, padding='post')
y_val = pad_sequences(y_val, maxlen=max_len_summary, padding='post')
y_voc_size = len(y_tokenizer.word_index) + 1

K.clear_session()
latent_dim = 500

# Encoder
encoder_inputs = Input(shape=(max_len_text,))
enc_emb = Embedding(x_voc_size, latent_dim, trainable=True)(encoder_inputs)

# LSTM 1
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
# 当return sequences参数设置为True时，LSTM为每个时间步生成隐藏状态和单元状态
# 当return state = True时，LSTM仅生成最后一个时间步的隐藏状态和单元状态
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# LSTM 2
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# LSTM 3
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

# Set up the decoder.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(y_voc_size, latent_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

# LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])
# initial_state 用于在第一个时间步初始化LSTM的内部状态

# Attention Layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

# Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=20,
                    callbacks=[es], batch_size=64,
                    validation_data=([x_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))
print(history.history['loss'])
print(history.history['val_loss'])

reverse_target_word_index = y_tokenizer.index_word
reverse_source_word_index = x_tokenizer.index_word
target_word_index = y_tokenizer.word_index

# encoder inference
encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_outputs, state_h, state_c])

# decoder inference
# Below tensors will hold the states of the previous time step
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_hidden_state_input = Input(shape=(max_len_text, latent_dim))

# Get the embeddings of the decoder sequence
dec_emb2 = dec_emb_layer(decoder_inputs)

# To predict the next word in the sequence, set the initial states to the states from the previous time step
decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2,
                                                    initial_state=[decoder_state_input_h, decoder_state_input_c])

# attention inference
attn_out_inf, attn_states_inf = attn_layer([decoder_hidden_state_input, decoder_outputs2])
decoder_inf_concat = Concatenate(axis=-1, name='concat')([decoder_outputs2, attn_out_inf])

# A dense softmax layer to generate prob dist. over the target vocabulary
decoder_outputs2 = decoder_dense(decoder_inf_concat)

# Final decoder model
decoder_model = Model(
    [decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = target_word_index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        try:
            sampled_token = reverse_target_word_index[sampled_token_index]
        except:
            stop_condition = True

        if (sampled_token != 'end'):
            decoded_sentence += ' ' + sampled_token

            # Exit condition: either hit max length or find stop word.
            if (sampled_token == 'end' or len(decoded_sentence.split()) >= (max_len_summary - 1)):
                stop_condition = True

                # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update internal states
        e_h, e_c = h, c

    return decoded_sentence


def seq2summary(input_seq):
    newString = ''
    for i in input_seq:
        if ((i != 0 and i != target_word_index['start']) and i != target_word_index['end']):
            newString = newString + reverse_target_word_index[i] + ' '
    return newString


def seq2text(input_seq):
    newString = ''
    for i in input_seq:
        if (i != 0):
            newString = newString + reverse_source_word_index[i] + ' '
    return newString


for i in range(len(x_val)):
    print("Review:", seq2text(x_val[i]))
    print("Original summary:", seq2summary(y_val[i]))
    print("Predicted summary:", decode_sequence(x_val[i].reshape(1, max_len_text)))
    print("\n")
