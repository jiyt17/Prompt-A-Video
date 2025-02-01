import json
from tqdm import tqdm


prompt_scores = json.load(open('/path/to/prompts_candidates_scores.json', 'r'))
res = []
for item in tqdm(prompt_scores):
    overall_scores = []
    for score in item['refined_scores']:
        overall_scores.append(score['VQ'] + score['TC'] + score['FC'] + score['DD'] * 1.1 + score['TVA'] * 1.3 + score['AES'] * 4 / 5 + score['MPS'] / 3)
    
    score_indexes = sorted(range(5), key=lambda x:overall_scores[x], reverse=True)

    # 0
    index = score_indexes[0]
    score = item['refined_scores'][index]
    if score['VQ'] > 2.6 and score['TC'] > 2.6 and score['FC'] > 2.55 and score['AES'] > 5.5 and score['TVA'] > 2.9 and score['MPS'] > 8.5 and score['DD'] > 2.65:
        for i in score_indexes[1:]:
            if overall_scores[index] - overall_scores[i] > 1.4:
                res.append({'prompt':item['user_prompts'], 'chosen': item['refined_prompts'][index], 'rejected': item['refined_prompts'][i]})

    # 1
    index = score_indexes[1]
    score = item['refined_scores'][index]
    if score['VQ'] > 2.6 and score['TC'] > 2.6 and score['FC'] > 2.55 and score['AES'] > 5.5 and score['TVA'] > 2.9 and score['MPS'] > 8.5 and score['DD'] > 2.65:
        for i in score_indexes[2:]:
            if overall_scores[index] - overall_scores[i] > 1.4:
                res.append({'prompt':item['user_prompts'], 'chosen': item['refined_prompts'][index], 'rejected': item['refined_prompts'][i]})

    # max min
    # max_ind = score_indexes[0]
    # min_ind = score_indexes[-1]
    # res.append({'prompt':item['user_prompts'], 'chosen': item['refined_prompts'][max_ind], 'rejected': item['refined_prompts'][min_ind]})

chosen_len = []
for item in res:
    chosen_len.append(len(item['chosen'].split()))
print(chosen_len)
res = [item for item in res if len(item['chosen'].split()) <= 130]
with open('dpo_dataset.json', 'w') as f:
    json.dump(res, f, indent=4)
