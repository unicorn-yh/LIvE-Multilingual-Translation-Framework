import pandas as pd
from chatgpt_api import chatgpt_api
import json


def instructions(field, rate_fluency, rate_semantic, rate_grammar, rate_terminology):
    st = ""
    if rate_fluency:
        st = "Fluency (1-7): Do the translated sentences sound natural and are they free of awkward phrasing or unnatural constructions?"
    elif rate_semantic:
        st = "Semantic Consistency (1-7): Are the translated sentences logically coherent, and do they align with the intended meaning of the context?"
    elif rate_terminology:
        st = "Terminology (1-7): Are field-specific terms used correctly and appropriately within the given translated context?"
    elif rate_grammar:
        st = "Grammatical Accuracy (1-7): Are the translated text free of grammatical errors and syntactical issues?"
    instruction = f"""You are a knowledgeable expert in the field of {field}. Please rate the provided corpus pair context which contains original Malay context and translated Chinese sentences out of 7. 
    
    [Evaluation Criterion]
    The evaluation criteria is the {st}
    
    Print the score on its own line corresponding to the evaluation. At the end, repeat just the selected score again on a new line.
    """
    return instruction


def eval_data(corpus_pair, field,  rate_fluency, rate_semantic, rate_grammar, rate_terminology):
    corpus_pair = json.dumps(corpus_pair)
    instruction = instructions(field, rate_fluency, rate_semantic, rate_grammar, rate_terminology)
    response = chatgpt_api(instruction=instruction, data=corpus_pair)
    return response


if __name__ == "__main__":
    field = "Neurobiology"
    rate_fluency = 0
    rate_semantic = 1
    rate_terminology = 0
    rate_grammar = 1
    
    
    gemma = 0
    gpt4 = 1
    glm4 = 0
    google = 0
    
    title = ""
    if rate_fluency:
        title = "Fluency"
    elif rate_grammar:
        title = "Grammar"
    elif rate_terminology:
        title = "Terminology"
    elif rate_semantic:
        title = "Semantic"
        
    model = ""
    if gemma:
        model = "gemma"
    elif gpt4:
        model = "gpt4"
    elif glm4:
        model = "glm4"
    elif google:
        model = "google"
        
            
    cf = pd.read_excel("/Users/olivia/Project/Gemma-Multilingual-Model/evaluation/all_model_eval.xlsx",
                    sheet_name=model, skiprows=1)

    scores = []
    for i in range(len(cf)):
        malay = cf.iloc[i,1]
        translated_chinese = cf.iloc[i,2]
        chinese = cf.iloc[i,3]
        english = cf.iloc[i,4]
        corpus_pair = {
            "English": english,
            "Chinese": chinese,
            "Malay": malay
        }
        corpus_pair2 = {
            "Original Malay Context": malay,
            "Translated Chinese Context": translated_chinese
        }
        score_string = eval_data(corpus_pair2, field,  rate_fluency, rate_semantic, rate_grammar, rate_terminology)
        print(score_string)
        score = score_string.split("\n")[-1]
        score = int(score)
        scores.append(score)
    
    print("--------------------------------------------")
    print(model)
    print(title,":")
    for s in scores:
        print(s)
    
    
    
    
    