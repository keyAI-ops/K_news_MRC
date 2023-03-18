#%%
import json
from pprint import pprint

news_json1 = open('C:\\Users\\ChangKI\\Groom_Project\\project_2\\news_data\\1.Training\\라벨링데이터\\TL_span_extraction\\TL_span_extraction.json', encoding = 'utf-8')
news_dict1 = json.load(news_json1)

#%%
indices = []
for d_id, document in enumerate(news_dict1['data']):
    for p_id, paragraph in enumerate(document['paragraphs']):
        for q_id, _ in enumerate(paragraph['qas']):
            indices.append((d_id, p_id, q_id))


#%%

for d_id ,p_id, q_id in indices:
    news_dict1['data'][d_id]['paragraphs'][p_id]['qas'][q_id]['question'] = news_dict1['data'][d_id]['paragraphs'][p_id]['qas'][q_id]['question']+'?'
    news_dict1['data'][d_id]['paragraphs'][p_id]['context'] = news_dict1['data'][d_id]['paragraphs'][p_id]['context'].replace("\n", "")
print(news_dict1['data'][0]['paragraphs'][0]['qas'][1]['question'])
print(news_dict1['data'][0]['paragraphs'][0]['context'])
#%%

# with open('output_train.json', 'w',encoding='utf-8') as outfile:
#     json.dump(result, outfile, indent="\t", ensure_ascii=False)

with open('C:\\Users\\ChangKI\\Groom_Project\\project_2\\model_code\\mdeberta\\out\\news_train.json', 'w',encoding='utf-8') as outfile:
    json.dump(news_dict1, outfile, indent="\t", ensure_ascii=False)
# %%
