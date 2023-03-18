#%%
from transformers import pipeline
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

#%%
# Data(json) 파일 불러오기
import json
test_json = open('C:\\Users\\ChangKI\\Groom_Project\\project_2\\data\\test.json', encoding = 'utf-8')
data_list = [json.load(test_json)]
#%%
# data_list 돌면서 data, paragraph, qna 인덱스 받아오기
indices = []
for d_id, document in enumerate(data_list[0]['data']):
    for p_id, paragraph in enumerate(document['paragraphs']):
        for q_id, _ in enumerate(paragraph['qas']):
            indices.append((d_id, p_id, q_id))
#%%
# 모델, 토크나이저 불러오기
qna_tokenizer = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2")
qna_model = AutoModelForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2")

qNa = pipeline('question-answering', 
                model=qna_model,
                tokenizer=qna_tokenizer) 
#%%
answers = []
guids = []
for d_id ,p_id, q_id in indices[:10]:
    context = data_list[0]['data'][d_id ]['paragraphs'][p_id]['context']
    question = data_list[0]['data'][d_id ]['paragraphs'][p_id]['qas'][q_id]['question']
    guid = data_list[0]['data'][d_id ]['paragraphs'][p_id]['qas'][q_id]['guid']
    answer = qNa({'question': f'{question}', 'context': f'{context}'})
    answers.append(answer)
    guids.append(guid)
    
# %%
import csv
import os
import pandas as pd

output_df = pd.DataFrame({
    'Id' : guids,
    'Predicted' : answers
    })
output_df.to_csv("out/baseline.csv", mode='w')

#%%
os.makedirs('out', exist_ok=True)
with open('out/baseline.csv', 'w', encoding='UTF-8', newline='') as fd:
    writer = csv.writer(fd)
    writer.writerow(['Id', 'Predicted'])
    
    rows = []
    for i in range(len(answers)):
        rows.append([guids[i], answers[i]['answer']])
        
    writer.writerows(rows)

# %%
import pickle

data = {
    'answers': answers,
    'guids' : guids
}

# save
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

# load
# with open('data.pickle', 'rb') as f:
#     data = pickle.load(f)
# %%
