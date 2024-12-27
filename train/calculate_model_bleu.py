import json
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import sacrebleu
import os
import nltk
# import jieba
import pkuseg

seg = pkuseg.pkuseg()
# nltk.download('stopwords')

def init():
    dirr = "../evaluation/"
    files = os.listdir(dirr)
    # custom_order = [
    #     "妇产科", "儿科", "普通外科", "风湿免疫科", "精神心理科", "皮肤科", 
    #     "神经内外科", "胃肠科", "心脏科", "血液科", "肿瘤科", "肾脏内科", 
    #     "骨科", "耳鼻喉科", "口腔科", "泌尿外科", "内分泌科", "感染科", 
    #     "急诊", "眼科", "呼吸科"
    # ]
    # files = sorted(files, key=lambda x: custom_order.index(x.split("_")[0]))
    print(files)
    print("BLEU Scores:")

def compute_bleu(reference, candidate, language = 'zh'):
    if language == 'zh':  # Chinese
        ref = [' '.join(seg.cut(reference))]
        cand = ' '.join(seg.cut(candidate))
    elif language == 'my':  # Malay
        stop_words = set(stopwords.words('malay'))  # Malay stopwords
        ref = [[word.lower() for word in reference.split() if word.isalpha() and word.lower() not in stop_words]]
        cand = [word.lower() for word in candidate.split() if word.isalpha() and word.lower() not in stop_words]
        ref = [' '.join(sentence) for sentence in ref]
        cand = ' '.join(cand)
    else:  # Default to English
        stop_words = set(stopwords.words('english'))  # English stopwords
        ref = [[word.lower() for word in reference.split() if word.isalpha() and word.lower() not in stop_words]]
        cand = [word.lower() for word in candidate.split() if word.isalpha() and word.lower() not in stop_words]
        ref = [' '.join(sentence) for sentence in ref]
        cand = ' '.join(cand)
    score = sacrebleu.sentence_bleu(cand, ref).score
    return score

    
def calculate_model_bleu(file_path, write_path, ind_ls):
    bleu = []
    with open(file_path, "r", encoding="utf-8") as ff:
        for line in ff:
            data = json.loads(line)
            translated_chinese = data["translated_chinese"]
            ground_truth_chinese = data["ground_truth_chinese"]
            score = compute_bleu(translated_chinese, ground_truth_chinese)
            bleu.append(score)
    avg_bleu = np.mean(np.array(bleu))
    print(f"{file_path}: {avg_bleu:.4f}")
    with open (write_path, "w", encoding="utf-8") as wf:
        wf.write("MODEL: ")
        wf.write(file_path)
        wf.write("\nBLEU Score: ")
        wf.write(str(avg_bleu))
        wf.write("\nIndex List: ")
        wf.write(str(ind_ls))
        wf.write("\nBLEU Score List: ")
        wf.write(str(bleu))
    # print(f"{avg_bleu/100:.4f}")
    return avg_bleu
    
def main():
    init()
    index=0
    for file in files:
        bleu = []
        index+=1
        calculate_model_bleu(dirr + file)
