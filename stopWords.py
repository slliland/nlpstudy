import pandas as pd
import jieba_fast
import warnings
import ssl
import copy
import numpy as np
import visdom

ssl._create_default_https_context = ssl._create_unverified_context


pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore")
data=pd.read_csv("./testdata/cndata.csv",nrows=10)
data.drop_duplicates(subset=['Text'],inplace=True)  # 去除重复值
data.dropna(axis=0,inplace=True)   # 去除NA值

def is_chinese(uchar):
    if u'\u4e00' <= uchar <= u'\u9fa5':
        return True
    else:
        return False

def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
        else:
            pass
    return content_str

def divide(datas):
    jieba_fast.setLogLevel(jieba_fast.logging.INFO)
    cut_words = list(jieba_fast.lcut(str(datas)[2:len(str(datas))-2]))
    return cut_words

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
        if len(i) >= 1:  # removing short word
            long_words.append(i)
    return (" ".join(long_words)).strip("[' ']")

cleaned_text = []
for t in data['Text']:
    cleaned_text.append(text_cleaner(t))
for i in cleaned_text:
    print(i)

def is_summary(uchar):
    if (u'\u4e00' <= uchar <= u'\u9fa5') or (u'\u0030' <= uchar <= u'\u0039') or uchar == '.':
        return True
    else:
        return False

def format_sum(content):
    content_str = ''
    for i in content:
        if is_summary(i):
            content_str = content_str + i
        else:
            pass
    return content_str

def summary_cleaner(text):
    newString = list(map(lambda s: s.replace(' ', ''), text))
    chinese_list = []
    chinese_list.append(format_sum(newString))
    tokens = copy.deepcopy(divide(chinese_list))
    long_words = ''
    for i in tokens:
        if len(i) >= 1:  # removing short word
            long_words = long_words + i
    return long_words

cleaned_summary = []
for t in data['Summary']:
    cleaned_summary.append(summary_cleaner(t))

data['cleaned_text']=cleaned_text
data['cleaned_summary']=cleaned_summary
data['cleaned_summary'].replace('', np.nan, inplace=True)
data.dropna(axis=0,inplace=True)
data['cleaned_summary'] = data['cleaned_summary'].apply(lambda x : '_START_ '+ x + ' _END_')

# for i in range(5):
#      print("Review:",data['cleaned_text'][i])
#      print("Summary:",data['cleaned_summary'][i])
#      print("\n")

# viz = visdom.Visdom()
text_word_count = []
summary_word_count = []
# populate the lists with sentence lengths
for i in data['cleaned_text']:
    text_word_count.append(len(i.split()))

for i in data['cleaned_summary']:
    summary_word_count.append(len(i.split()[1]))

# viz.histogram(
#     X = text_word_count,
#     opts=dict(
#         title = "评论长度",
#         numbins = 30
#     )
# )
#
# viz.histogram(
#     X = summary_word_count,
#     opts=dict(
#         title = "摘要长度",
#         numbins = 30
#     )
# )

