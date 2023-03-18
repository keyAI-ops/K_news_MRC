#%%
# import sys
# sys.path.insert(0,'/usr/lib/chromium-browser/chromedriver')
# from selenium import webdriver

import pandas as pd
import os
import torch
from tqdm import tqdm, trange
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import re
import csv

#%%
import pandas as pd
valid_data = pd.read_csv('C:\\Users\\ChangKI\\Groom_Project\\project_2\\split_valid_data\\valid_10percent.csv')

valid_data_context = valid_data['pharagraph'].values.tolist()
valid_data_question = valid_data['question'].values.tolist()
valid_data_answer = valid_data['answer'].values.tolist()

#%%
# device 설정 : GPU 사용가능 하면 cuda 사용 아니면 cpu 써라 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

# load pre-trained Model
# model_QA = AutoModelForQuestionAnswering.from_pretrained("timpal0l/mdeberta-v3-base-squad2").to(device)
# model_QA.load_state_dict(torch.load('C:\\Users\\ChangKI\\Groom_Project\\project_2\\model_bin\\mdeberta_model9.bin'))
model_QA = torch.load('C:\\Users\\ChangKI\\Groom_Project\\project_2\\model_bin\\mdeberta_train_4.pt')
tokenizer_QA = AutoTokenizer.from_pretrained("timpal0l/mdeberta-v3-base-squad2", do_lower_case=True) # do_lower_case=True : 모두 소문자로 변환 / translation task도 같이 있어서 붙은거 같음..
#%%

# Q&A TASK 진행
from transformers import pipeline
qNa = pipeline('question-answering', 
                model=model_QA,
                tokenizer=tokenizer_QA,
                device = 0) # GPU 사용 가능하게함

#%%
pre_answers = []

for num in tqdm(range(len(valid_data_context))):
          
  context = valid_data_context[num]
  question = valid_data_question[num]
    
  pre_answer = qNa({'question': f'{question}', 'context': f'{context}'})
  pre_answers.append(pre_answer['answer'])
  
#%%
# import pickle
# with open('data_pickle_2.pkl','rb') as f:
#     mydata = pickle.load(f)
    
# #%%
# import pickle
# pre_answers = mydata[0]
# valid_data_answer = mydata[1]
#%%
import re

def normalize_answer(s):    
    def remove_(text):
        ''' 불필요한 기호 제거 '''
        text = re.sub('[-=#/?:$}]', '', text) 
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
from levenshtein_distance import levenshtein_distance

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
output_df.to_csv("C:\\Users\\ChangKI\\Groom_Project\\project_2\\out\\mdeberta(2).csv", mode='w', index=False)

# %%
