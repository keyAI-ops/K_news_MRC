#%%

import pandas as pd
import os
import json
import torch
from tqdm import tqdm, trange
from tqdm.notebook import tqdm
from torch import nn
from transformers import BertForQuestionAnswering, AutoTokenizer, AutoModelForQuestionAnswering
from kobert_tokenizer import KoBERTTokenizer
import string
import re
import pickle

#%%
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
# model_QA = AutoModelForQuestionAnswering.from_pretrained("skt/kobert-base-v1").to(device)
model_QA = torch.load('C:\\Users\\ChangKI\\Groom_Project\\project_2\\model_bin\\kobert_aihub_epoch3.pt')
# model_QA.load_state_dict(torch.load('C:\\Users\\ChangKI\\Groom_Project\\project_2\\model_bin\\korquad_multilingual_model3.bin'))
# tokenizer_QA = AutoTokenizer.from_pretrained("bert-base-multilingual-uncased", do_lower_case=True)
tokenizer_QA = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1", do_lower_case=True) # do_lower_case=True : 모두 소문자로 변환 / translation task도 같이 있어서 붙은거 같음..
#%%

# Q&A TASK 진행

with torch.no_grad():
  predicted_answers = []
  proba = []
  for d_id ,p_id, q_id in tqdm(indices):  # context, question 뽑아오기
    context = data_list[0]['data'][d_id ]['paragraphs'][p_id]['context']
    question = data_list[0]['data'][d_id ]['paragraphs'][p_id]['qas'][q_id]['question']

    answers = []
    start_logits = []
    end_logits = []

    data = input_QA(question, context, tokenizer_QA, doc_stride=128)    
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

# %%

# guid는 순서대로 진행이 되기 때문에 바로 list로 변환함
guids = []
for d_id ,p_id, q_id in indices:
    guid = data_list[0]['data'][d_id ]['paragraphs'][p_id]['qas'][q_id]['guid']
    guids.append(guid)

# df -> csv 파일로 변환

output_df = pd.DataFrame({
    'Id' : guids,
    'Predicted' : predicted_answers
    })
output_df.to_csv("C:\\Users\\ChangKI\\Groom_Project\\project_2\\model_code\\kobert\\out\\baseline_k.csv", mode='w')
# %%
