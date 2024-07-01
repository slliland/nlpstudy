# README
### 项目文件
1. testdata：包含实验所用的数据,其中包括：\
cndata.csv：用于中文自动文本摘要的数据集，包含24500条数据，大小为149.3mb。\
stopwords.txt：中文停用词表。来源为：<https://gitee.com/UsingStuding/stopwords/blob/master/哈工大停用词表.txt#>。\
Revies.csv：用于英文自动文本摘要的数据集，大小为300.9mb。\
train.txt：用于情感分析的训练集。\
word_freq.txt：出现频率大于25的词存入该文件。\
wiki_word2vec_50.bin：word2vec。
2. abstract.py：英文自动文本摘要。
3. cnAbstract.py：中文自动文本摘要。
4. creatDataset.py：创建中文数据集脚本。
5. drawPic.py：绘制自动文本摘要的损失值变化图脚本。
6. sentiment1.py：情感分析。
