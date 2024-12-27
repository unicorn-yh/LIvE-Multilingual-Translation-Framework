import sacrebleu
from nltk.translate.meteor_score import meteor_score
import jieba
import nltk
nltk.download('wordnet')

# Define the reference and hypothesis sentences
# refs = [['The quick brown fox jumps over the lazy dog.']]  # This should be a list of lists, each sublist being one set of references for the translation
# hypo = ['The quick brown fox jumped over the lazy dogs.']  # This should be a list of hypothesis strings

# Calculate chrF score
def chrf(translate, ground_truth):
    hypo = [translate]
    refs = [[ground_truth]]
    chrf_score = sacrebleu.corpus_chrf(hypo, refs)  # Correct order and format
    return chrf_score.score
    # print(f"chrF score: {chrf_score.score:.3f}")
    
def meteor(hypothesis, reference):
    hypothesis = list(jieba.cut(hypothesis))
    reference = list(jieba.cut(reference))
    score = meteor_score([reference], hypothesis)
    return score
    
    
import pandas as pd
from bleu import compute_bleu
from comet_utils import init_comet, comet_score
import numpy as np
model="gemma"
cf = pd.read_excel("/Users/olivia/Project/Gemma-Multilingual-Model/evaluation/all_model_eval.xlsx",
                   sheet_name=model, skiprows=1)

bleus = 0
comets = 0
chrfs = 0
meteors = 1
score = []
for i in range(len(cf)):
        input = cf.iloc[i,1]
        translated_chinese = cf.iloc[i,2]
        ground_truth_chinese = cf.iloc[i,3]
        if bleus:
            bleu_score = compute_bleu(translated_chinese, ground_truth_chinese)
            print(bleu_score)
            score.append(bleu_score)
        elif comets:
            data = [
                {
                    "src": input,
                    "mt": translated_chinese,
                    "ref": ground_truth_chinese,
                }
            ]
            model = init_comet()
            cs = comet_score(model, data)
            print(cs)
            score.append(cs)
            
        elif chrfs:
            chrf_score = chrf(translated_chinese, ground_truth_chinese)
            print(chrf_score)
            score.append(chrf_score)
        elif meteors:
            mscore = meteor(translated_chinese, ground_truth_chinese)
            print(mscore)
            score.append(mscore)
            
    
print("\nAverage:")
avg = np.mean(np.array(score))
print(avg)
print(model)

