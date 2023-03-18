#%%

import os
import torch
from tqdm import tqdm, trange
from tqdm.notebook import tqdm
from kobert_tokenizer import KoBERTTokenizer
import re
import csv
import pickle

#%%
import pandas as pd
valid_data = pd.read_csv('C:\\Users\\ChangKI\\Groom_Project\\project_2\\split_valid_data\\valid_10percent.csv')

valid_data_context = valid_data['pharagraph'].values.tolist()
valid_data_question = valid_data['question'].values.tolist()
valid_data_answer = valid_data['answer'].values.tolist()
#%%
# input값 설정

def input_QA(question, context, tokenizer, doc_stride=128):  # 
  input_ids = []    # id token
  input_masks = []  # attention mask랑 같은 의미? 불필요한 패딩은 0 실제 단어는 1로 표기
  segment_ids = []  # segment id : 문장 구분하기 위한 id
  question_len = len(tokenizer.encode(question)) -2   # 2를 빼는 이유는 문장이 시작할 때 2가 붙고 문장이 끝날때 3이 붙음
  context_len = len(tokenizer.encode(context)) -2

  if context_len >= 509 - question_len:    # 예시 context_len: 520 / question_len 10
    cont_len = 509 - question_len          # cont_len : 499
    num = int(context_len / cont_len)      # num = 1.xxx
    for i in range(num+1):
      cont_split = tokenizer.encode(context)[1:-1]
      if i == 0 :  
         stride = 0
      else:
         stride = doc_stride
      cont_split = cont_split[i*cont_len - stride : (i + 1)*cont_len - stride] # -> - stride에 따라 나눠짐
      sequence = tokenizer.encode(question) + cont_split + tokenizer.encode(['[SEP]'])[1:2]
      input_mask = [1] * len(sequence)
      segment_id = [0] * (question_len+2)+ [1] * (len(cont_split)+1)
      input_ids.append(torch.tensor([sequence]))
      input_masks.append(torch.tensor([input_mask]))
      segment_ids.append(torch.tensor([segment_id]))
  else:
    sequence = question + '[SEP]' + context
    sequence = tokenizer.encode(sequence)
    input_mask = [1] * len(sequence)
    segment_id = [0] * (question_len+2)+ [1] * (len(sequence) - (question_len+2)) # 아까 줄여놓은 길이 되돌리고 0과 1로 구분
    input_ids.append(torch.tensor([sequence]))
    input_masks.append(torch.tensor([input_mask]))
    segment_ids.append(torch.tensor([segment_id]))
  return zip(input_ids, input_masks, segment_ids)

#%%
# device 설정 : GPU 사용가능 하면 cuda 사용 아니면 cpu 써라 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 

# load pre-trained Model
# model_QA.load_state_dict(torch.load('C:\\Users\\ChangKI\\Groom_Project\\project_2\\model_bin\\mdeberta_0206.bin'))
model_QA = torch.load('C:\\Users\\ChangKI\\Groom_Project\\project_2\\model_bin\\kobert_aihub_epoch3.pt')
tokenizer_QA = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1") # do_lower_case=True : 모두 소문자로 변환 / translation task도 같이 있어서 붙은거 같음..
#%%

# Q&A TASK 진행
with torch.no_grad():
  predicted_answers = []
  proba = []
  for num in tqdm(range(len(valid_data_context))):
    answers = []
    start_logits = []
    end_logits = []

    data = input_QA(valid_data_question[num], valid_data_context[num], tokenizer_QA, doc_stride=128)    
    for input_id, input_mask, segment_id in data:
      input_id = input_id.to(device)
      input_mask = input_mask.to(device)
      segment_id = segment_id.to(device)

      inputs = {'input_ids' : input_id,
                'token_type_ids' : segment_id,
                'attention_mask' : input_mask}
    
      outputs_QA = model_QA(**inputs)
      
      answer_start_index = outputs_QA.start_logits.argmax(dim=-1)  # argmax(dim=-1) : 행을 기준으로 최대값 인덱스 뽑아옴
      answer_start_logits = torch.nn.Softmax(dim=1)(outputs_QA.start_logits)[0][answer_start_index]
      answer_end_index = outputs_QA.end_logits.argmax(dim=-1)
      answer_end_logits = torch.nn.Softmax(dim=1)(outputs_QA.end_logits)[0][answer_end_index]
  
      if answer_start_index.item() != 0 and answer_end_index != 0 and tokenizer_QA.decode(input_id[0][answer_start_index:answer_end_index+1]).strip() != '':
        end_logits.append(answer_end_logits)
        start_logits.append(answer_start_logits)

        answer = tokenizer_QA.decode(input_id[0][answer_start_index:answer_end_index+1]).strip()
        answers.append(answer)

    if len(answers) != 0: 
      start_logits = torch.tensor(start_logits).unsqueeze(0)
      end_logits = torch.tensor(end_logits).unsqueeze(0)
      prob_matrix = start_logits * end_logits
      answer_index = (start_logits * end_logits).argmax(dim=1).item()
      pred_ans = answers[answer_index]
      prob = prob_matrix[0][answer_index].item()

    else:
      pred_ans = ''
      predicted_answers.append(pred_ans)  # 원래 이 부분이 없었는데 없는 상태로 진행하게 되면 blank 부분을 빼고 진행함. 그래서 output의 수도 줄어든다.

    if pred_ans != '':
      predicted_answers.append(pred_ans)  # 답이 있다면 추가해라
      proba.append(prob)                  # prob는 확률
    
#%%
# normalizing
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
    
    def trunc(text):
        if len(text) > 12:
              text = ''
        return text
      
    return trunc(white_space_fix(remove_(s)))
#%%
# 혹시 몰라서 복사
import copy
answers_sub = copy.deepcopy(predicted_answers)
#%%
# 정규화 진행
for i in range(len(predicted_answers)):
    predicted_answers[i] = normalize_answer(predicted_answers[i])

#%%
from levenshtein_distance import levenshtein_distance

distance_1, avg_distance_1 = levenshtein_distance(answers_sub, valid_data_answer)
distance_2, avg_distance_2 = levenshtein_distance(predicted_answers, valid_data_answer)

print('예측값 편집거리 : ', avg_distance_1)
print('정규화 후 편집거리 : ', avg_distance_2)

# %%
# Predicted는 정규화 하기 전이고 normalize_Predicted 정규화 하고 난 후 값이다.
output_df = pd.DataFrame({
    'distance_1' : distance_1,
    'distance_2' : distance_2,
    'answer' : valid_data_answer,
    'Predicted' : answers_sub,
    'normalize_Predicted' : predicted_answers
    })
output_df.to_csv("C:\\Users\\ChangKI\\Groom_Project\\project_2\\model_code\\kobert\\out\\kobert(5).csv", mode='w', index=False)

# %%
