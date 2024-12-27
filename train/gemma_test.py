from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, BitsAndBytesConfig)
import torch
import os
import json
import pandas as pd
from calculate_model_bleu import compute_bleu
from rag import preprocess_retrieval_data, initialize_retriever, integrate_rag_with_finetuned_model, translate_with_rag


def load_model(path_to_model):
    print(path_to_model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )      
    tokenizer = AutoTokenizer.from_pretrained(
        path_to_model
    )
    model = AutoModelForCausalLM.from_pretrained(
        path_to_model,
        device_map="auto",
        quantization_config=bnb_config
    )
    return model, tokenizer


def load_test_dataset(data_file_path, DATA_SIZE):
    data = []
    with open(data_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    df = pd.DataFrame(data)
    df = df.head(DATA_SIZE)
    return df

def en2zh_text():
    en_text = "Similarly, the two DN2 are part of a clade. Exceptions to this are the clades containing DN1p and DN3. For DN1p, this can be explained by our findings, which show that this group comprises five morphologically distinct subtypes."
    en_text = "English: " + en_text + "\n中文: "
    return en_text

def my2zh_text():
    my_text = "Pertumbuhan akar saraf yang berlaku dalam saluran vertebra menyebabkan akar lumbar, sakral, dan koksigeal memanjang ke paras vertebra yang sesuai. Semua saraf tunjang, kecuali yang pertama, keluar di bawah vertebra yang bersesuaian. Dalam segmen serviks, terdapat 7 vertebra serviks dan 8 saraf serviks."
    my_text = "Melayu: " + my_text + "\n中文: "
    return my_text


def zh2my_text():
    zh_text = "大脑的发育与后来改变神经细胞之间连接的机制之间存在相似性，这一过程被称为神经可塑性。可塑性被认为是学习和记忆的基础。我们的神经系统能够记住电话号码以及你去年圣诞节所做的事情。"
    zh_text = "中文：" + zh_text + "\nMelayu: "
    return zh_text
    

def test_input(model, tokenizer, input_text, source, target, RAG):
    if source == "zh":
        prompt1 = "中文："
    elif source == "en":
        prompt1 = "English: "
    if source == "my":
        prompt1 = "Melayu: "
        
    if target == "zh":
        prompt2 = "\n中文："
    elif target == "en":
        prompt2 = "\nEnglish: "
    if target == "my":
        prompt2 = "\nMelayu: "
    
    input_text = prompt1 + input_text + prompt2
    # print("INPUT:")
    # print(input_text)
    
    if RAG:
        output_text = translate_with_rag(model, tokenizer, input_text, prompt1, prompt2)
    else:
        input_tokenizer = tokenizer(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_text = model.generate(**input_tokenizer, max_new_tokens=128)
        output_text = tokenizer.decode(output_text[0], skip_special_tokens=True)
        if prompt2 in output_text:
            output_text = output_text.split(prompt2)[1]
        if "\n" in output_text:
            output_text = output_text.split("\n")[0]
    return output_text


   
def run_test(path_to_model, data_file_path, write_test_path, 
             TEST_DATA_SIZE, TEST_ONE_SAMPLE, REFLECTION, 
             RAG, rag_model_path, retrieved_dataset_path, index_path, 
             context_encoder_path, rag_dataset, source, target):
    model, tokenizer = load_model(path_to_model) 
    
    if RAG:
        retrieval_data = rag_dataset["data"]
        retrieval_titles = rag_dataset["titles"]
        preprocess_retrieval_data(retrieval_data, retrieval_titles, retrieved_dataset_path, 
                                  index_path, context_encoder_path)
        retriever = initialize_retriever(retrieved_dataset_path, index_path, rag_model_path)
        model, tokenizer = integrate_rag_with_finetuned_model(model, retriever, rag_model_path)
        
    model.eval()
    # print(model.dtype)  # torch.float32
    # test1 = 0
    ind_ls = []
    if TEST_ONE_SAMPLE:
        # my_text = my2zh_text()
        text = zh2my_text()
        output_text = test_input(model, tokenizer, text, source, target, RAG)
        print("Translated Output:")
        print(output_text)
    else:
        dataset = load_test_dataset(data_file_path, TEST_DATA_SIZE)
        with open(write_test_path,"w",encoding="utf-8") as write_file:
            for index, row in dataset.iterrows():
                ind = int(row["Index"])
                my_text = str(row["Malay"])
                en_text= str(row["English"])
                zh_text= str(row["Chinese"])
                ind_ls.append(ind)
                output_zh = test_input(model, tokenizer, my_text, source, target, RAG)
                if REFLECTION:
                    score = compute_bleu(output_zh, zh_text)
                    if score < 0.5:
                        print(output_zh)
                        print(score)
                        output_zh = test_input(model, tokenizer, my_text, source, target, RAG)
                        score = compute_bleu(output_zh, zh_text)
                        print(output_zh)
                        print(score)
                tmp_dict = {
                    "id": ind,
                    "input": my_text,
                    "translated_chinese": output_zh,
                    "ground_truth_chinese": zh_text,
                    # "translated_english": output_en,
                    "ground_truth_english": en_text
                }
                print(tmp_dict)
                json.dump(tmp_dict, write_file, ensure_ascii=False)
                write_file.write("\n")
    return ind_ls
            
            
            


