#%%
# -*- coding: utf-8 -*-

import pandas as pd
import os
import json
import collections
import torch
from tqdm import tqdm, trange
from tqdm.notebook import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoTokenizer
import pickle
#%%
"""# data"""

# 정답 추출
train_data = open('C:\\Users\\ChangKI\\Groom_Project\\project_2\\data\\news_train.json', encoding = 'utf-8')
data_list = [json.load(train_data)]

df = pd.DataFrame()
context = []
questions = []
answers = []
is_impossible = []
answer_start = []

for i in data_list:
  for j in i['data']:
    cont = j['paragraphs'][0]['context']
    qas = j['paragraphs'][0]['qas']
    for k in qas:
      context.append(cont)
      questions.append(k['question'])
      if k['is_impossible'] == True:
        is_impossible.append(1)
      else:
        is_impossible.append(0)
      answers.append(k['answers']['text'])
      answer_start.append(k['answers']['answer_start'])

#%%

df['pharagraph'] = context
df['question'] = questions
df['answer'] = answers
df['answer_start'] = answer_start
df['is_impossible'] = is_impossible

#%%
# data load
import pandas as pd
data = df
#%%

from tqdm import trange

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 cls_index,
                 p_mask,
                 paragraph_len,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.cls_index = cls_index
        self.p_mask = p_mask
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
#%%

class InputFeaturesForEval(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 paragraph_len,
                 start_position=None,
                 end_position=None,):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.paragraph_len = paragraph_len
        self.start_position = start_position
        self.end_position = end_position
#%%

def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    do_print = False
    features = []
    for (example_index, example) in enumerate(trange(len(examples))):

        example = examples[example_index]
        query_tokens = tokenizer.tokenize(example.question)

        # 쿼리 길이가 길 경우 max_query_length로 자르기
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.context):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start]
            if example.end < len(example.context) - 1:
                tok_end_position = orig_to_tok_index[example.end + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.answer)

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple( 
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        if len(doc_spans) == 2:
            do_print = True

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # Query
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            # start_position의 위치가 doc_span의 길이 전체를 넘어버릴 경우 is_impossible=True로 수정하기
            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and span_is_impossible:
                start_position = cls_index
                end_position = cls_index

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    cls_index=cls_index,
                    p_mask=p_mask,
                    paragraph_len=paragraph_len,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=span_is_impossible))
            unique_id += 1

    return features
#%%

import os
import json
import torch
from torch.utils.data import Dataset, TensorDataset

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False
#%%

class SquadExample():
    def __init__(self, context, question, answer, start, end, is_impossible):
        self.context = context
        self.question = question
        self.answer = answer
        self.start = start
        self.end = end
        self.is_impossible = is_impossible
        
    def __repr__(self):
        return 'id:{}  question:{}...  answer:{}...  is_impossible:{}'.format(
            self.question[:10],
            self.answer[:10],
            self.is_impossible)
#%%

class SquadDataset(Dataset):
    def __init__(self, data, tokenizer, is_train=True, is_inference=False):

        self.examples = []
        for i in range(len(data)):
            entry = data.iloc[[i]]
            context = entry["pharagraph"].values[0]
            
            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in context:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)                    
            if entry['is_impossible'].values[0] == 0:
                is_impossible = False # 0-True, 1-False
            else:
                is_impossible = True
            if not is_impossible:
                original_answer = entry['answer'].values[0]
                answer_start = int(entry['answer_start'].values[0])
                
                answer_length = len(original_answer)
                start_pos = char_to_word_offset[answer_start]
                end_pos = char_to_word_offset[answer_start + answer_length - 1]

                answer_end = answer_start + len(original_answer)
            else:
                original_answer = ''
                start_pos = 1
                end_pos = -1

            example = SquadExample(
                context=doc_tokens,
                question=entry['question'].values[0],
                answer=original_answer,
                start=start_pos,
                end=end_pos,
                is_impossible=is_impossible)
            self.examples.append(example)
        print('examples: {}'.format(len(self.examples)))

        self.features = convert_examples_to_features(
            examples=self.examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=True )
        print('is_training: {}'.format(True))


    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]
#%%

class SquadDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, is_inference=False, shuffle=True):
        self.is_inference = is_inference
        super().__init__(dataset, collate_fn=self.squad_collate_fn, batch_size=batch_size, shuffle=shuffle)
        
    def squad_collate_fn(self, features):
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        # return 6 tensors
        if self.is_inference:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            return all_input_ids, all_input_mask, all_segment_ids, all_cls_index, all_p_mask, all_example_index
        # return 7 tensors
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            return all_input_ids, all_input_mask, all_segment_ids, all_cls_index, all_p_mask, all_start_positions, all_end_positions
#%%

"""### mdeberta"""

# tokenizer load
tokenizer = AutoTokenizer.from_pretrained('timpal0l/mdeberta-v3-base-squad2')

# Dataset Loading
train_dataset = SquadDataset(data, tokenizer, is_train=True)

# Data Loader
# train_dataloader_32 = SquadDataLoader(train_dataset, batch_size=32, is_inference=False, shuffle=True)
train_dataloader_16 = SquadDataLoader(train_dataset, batch_size=16, is_inference=False, shuffle=True)
#%%

"""# model train"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#%%

# mdeberta
from transformers import AutoModelForQuestionAnswering
# model = AutoModelForQuestionAnswering.from_pretrained('timpal0l/mdeberta-v3-base-squad2').to(device)
model = torch.load('/content/drive/MyDrive/Colab Notebooks/프로젝트2_pt_파일/mdeberta_train_10.pt')
model.train

optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
loss = nn.CrossEntropyLoss()
n_epoch = 5
#%%

# train fucntion
def train(model, dataloader, optimizer):
    tbar = tqdm(dataloader, desc='Training', leave=True)
    
    total_loss = 0.0
    for i, batch in enumerate(tbar):
        optimizer.zero_grad()
        
        # cls_index와 p_mask는 XLNet 모델에 사용되므로 BERT에서는 사용하지 않음
        input_ids, input_mask, segment_ids, cls_index, p_mask, start_positions, end_positions = batch
        
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        start_positions = start_positions.to(device)
        end_positions = end_positions.to(device)
        
        inputs = {'input_ids' : input_ids,
                 'token_type_ids' : segment_ids,
                 'attention_mask' : input_mask}
        
        out = model(**inputs, start_positions=start_positions, end_positions=end_positions)
        loss = out.loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data.item()
        tbar.set_description("Average Loss = {:.4f}".format(total_loss/(i+1)))
    
for i in range(n_epoch):
    train(model, train_dataloader_16, optimizer)
torch.save(model, '/content/drive/MyDrive/Colab Notebooks/프로젝트2_pt_파일/mdeberta_aihub_10.pt')
