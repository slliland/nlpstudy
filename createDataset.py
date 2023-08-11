# -*- 本脚本用于规范化中文数据集 Created by Songyujian 2023.8.9-*-
import pandas as pd
import re

def extract_text(line):
    s_pattern = re.compile('summary{{(.*?)}}')
    t_pattern = re.compile('text{{(.*?)}}')
    summary = s_pattern.findall(line)
    text = t_pattern.findall(line)
    return summary, text

summ = []
txt = []
Id = []
for i in range(1,24500):
    try:
        path = '/Users/songyujian/Downloads/chinese_abstractive_corpus-master/copus/'+str(i)+'.txt'
        data = pd.read_csv(path, header=None)
        summary = data.loc[0, 0]
        text = data.loc[1, 0]
        line = summary + text
        Id.append(i)
        summ.append((' '.join(str(i) for i in extract_text(line)[0])))
        txt.append((' '.join(str(i) for i in extract_text(line)[1])))
    except:
        pass
df = pd.DataFrame({'Id': Id, 'Summary': summ, 'Text': txt})
file_path = "./testdata/cndata.csv"
df.to_csv(file_path, index=False)
# print(df)





