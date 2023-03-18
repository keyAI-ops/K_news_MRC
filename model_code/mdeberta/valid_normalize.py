#%%
import pickle
from pprint import pprint
import pandas as pd
import csv
import os

with open('data_pickle.pkl','rb') as f:
    mydata = pickle.load(f)
    
#%%
pre_answers = mydata[0]
valid_data_answer = mydata[1]
#%%
import re

def normalize_answer(s):    
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub('[-=#@&*,/?:$}""▲△▽▼]', '', text) 
        if text.find('(') != -1:
            text = text[:text.find('(')]
    
        return text

    def white_space_fix(text):
        return ' '.join(text.split())
    
    return white_space_fix(remove_(s))
#%%
import copy
answers_sub = copy.deepcopy(pre_answers)

#%%
for i in range(len(pre_answers)):
    pre_answers[i] = normalize_answer(pre_answers[i])

#%%
from levenshtein_distance_1 import levenshtein_distance

distance_1, avg_distance_1 = levenshtein_distance(answers_sub, valid_data_answer)
distance_2, avg_distance_2 = levenshtein_distance(pre_answers, valid_data_answer)

print(avg_distance_1)
print(avg_distance_2)
# %%
# Predicted는 정규화 하기 전이고 normalize_Predicted 정규화 하고 난 후 값이다.
output_df = pd.DataFrame({
    'distance_1' : distance_1,
    'distance_2' : distance_2,
    'answer' : valid_data_answer,
    'Predicted' : answers_sub,
    'normalize_Predicted' : pre_answers
    })
output_df.to_csv("C:\\Users\\ChangKI\\Groom_Project\\project_2\\out\\comparison_mdeberta.csv", mode='w', index=False)

os.makedirs('out', exist_ok=True)
with open('out/comparison_mdeberta.csv', 'w', encoding='UTF-8', newline='') as fd:
    writer = csv.writer(fd)
    writer.writerow(['distance_1', 'distance_2', 'answer','Predicted', 'normalize_Predicted'])
    
    rows = []
    for i in range(len(pre_answers)):
        rows.append([distance_1[i], distance_2[i], valid_data_answer[i], answers_sub[i], pre_answers[i]])
        
    writer.writerows(rows)# %%

# %%
'''
# 여기부터는 train.json 파일을 question, answer 형태로만 보고싶어서 만듦

import json
train_json = open('C:\\Users\\ChangKI\\Groom_Project\\project_2\\data\\train.json', encoding = 'utf-8')
data_list = [json.load(train_json)]

# data_list 돌면서 data, paragraph, qna 인덱스 받아오기
indices = []
for d_id, document in enumerate(data_list[0]['data']):
    for p_id, paragraph in enumerate(document['paragraphs']):
        for q_id, qas in enumerate(paragraph['qas']):
            indices.append((d_id, p_id, q_id))

data_list[0]['data'][0]['paragraphs'][0]['context']
question = data_list[0]['data'][0]['paragraphs'][0]['qas'][0]['question']
answer = data_list[0]['data'][0]['paragraphs'][0]['qas'][0]['answers'][0]['text']

from tqdm import tqdm
q = []
a = []
for d_id ,p_id, q_id in tqdm(indices):
    question = data_list[0]['data'][d_id]['paragraphs'][p_id]['qas'][q_id]['question']
    answer = data_list[0]['data'][d_id]['paragraphs'][p_id]['qas'][q_id]['answers'][0]['text']
    q.append(question)
    a.append(answer)
    
output_df = pd.DataFrame({
    'question' : q,
    'answer' : a
    })
output_df.to_csv("C:\\Users\\ChangKI\\Groom_Project\\project_2\\out\\train_data.csv", mode='w', index=False)

os.makedirs('out', exist_ok=True)
with open('out/comparison_train_data.csv', 'w', encoding='UTF-8', newline='') as fd:
    writer = csv.writer(fd)
    writer.writerow(['question', 'answer'])
    
    rows = []
    for i in range(len(q)):
        rows.append([q, a])
        
    writer.writerows(rows)
'''