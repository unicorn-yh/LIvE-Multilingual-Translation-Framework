import os
import torch
from accelerate import Accelerator
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, AutoConfig, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback, BitsAndBytesConfig)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
import warnings
import logging
from gemma_test import run_test
from calculate_model_bleu import calculate_model_bleu
from sentence_transformers import SentenceTransformer, util
import faiss

TRAIN = False
TEST = True


# Hyperparameters
EPOCH = 1
DATA_SIZE = 5000
LORA_RANK = 32
LORA_ALPHA = 32
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 0.00001  #1e-5
TEST_SIZE = 0.2
TEST_DATA_SIZE = 1
TEST_ONE_SAMPLE  = True
REFLECTION = 0
RAG = 0
BACK_TRANSLATION = 0
RAG_DATA_SIZE = 500 if RAG else 0
SOURCE = "my"
TARGET = "zh"

# Environment setup
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()


# Paths
path_to_model = "../Gemma-9B"  
cache_dir = "../cache"
model_name = f"R{LORA_RANK}_A{LORA_ALPHA}_LR{LEARNING_RATE}_{DATA_SIZE}_E{EPOCH}_BT{BACK_TRANSLATION}"
save_model_path = f"../fine_tuned_model/{model_name}"
data_file_path = "../data/biology_train.jsonl"
data_file_path_test = "../data/biology_test.jsonl"
os.makedirs(cache_dir, exist_ok=True)
log_file_path = f"../log/{model_name}.log"
write_test_path = f"../evaluation/{model_name}_{TEST_DATA_SIZE}_RAG{RAG}.jsonl"
write_bleu_path = f"../bleu/{model_name}_{TEST_DATA_SIZE}_RAG{RAG}.txt"
rag_model_path = f"../rag_model"
retrieved_dataset_path =  f"../retrieval_dataset"
index_path = f"../retrieval_index.faiss"
context_encoder_path = f"../context_encoder"
# os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Logging setup
def setup_logging(log_file_path):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("transformers")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    return logger

def load_model(path_to_model):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )      
    model_config = AutoConfig.from_pretrained(path_to_model)
    tokenizer = AutoTokenizer.from_pretrained(
        path_to_model
    )
    model = AutoModelForCausalLM.from_pretrained(
        path_to_model,
        device_map="auto",
        quantization_config=bnb_config,
    )
    torch.cuda.empty_cache()
    return model, tokenizer
    

def formatting_func_my2en(example):
    text = f"Melayu: {example['Malay']}\nEnglish: "
    label = f"{example['English']}"
    return {"text": text, "labels": label}

def formatting_func_en2zh(example):
    text = f"English: {example['English']}\n中文: "
    label = f"{example['Chinese']}"
    return {"text": text, "labels": label} 

def formatting_func_en2my(example):
    text = f"English: {example['English']}\nMalay: "
    label = f"{example['Malay']}"
    return {"text": text, "labels": label}

def formatting_func_zh2en(example):
    text = f"中文: {example['Chinese']}\nEnglish: "
    label = f"{example['English']}"
    return {"text": text, "labels": label}


def tokenize_function(example, tokenizer):
    inputs = tokenizer(example['text'], truncation=True, padding="max_length", max_length=128)
    labels = tokenizer(example['labels'], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = labels["input_ids"]
    return inputs


def load_train_dataset(data_file_path, DATA_SIZE, tokenizer, BACK_TRANSLATION):
    data_my2en = load_dataset("json", data_files=data_file_path)
    data_en2zh = load_dataset("json", data_files=data_file_path)
    data_en2my = load_dataset("json", data_files=data_file_path)
    data_zh2en = load_dataset("json", data_files=data_file_path)
    data_my2en['train'] = data_my2en['train'].select(range(DATA_SIZE))
    data_en2zh['train'] = data_en2zh['train'].select(range(DATA_SIZE))
    data_en2my['train'] = data_en2my['train'].select(range(DATA_SIZE))
    data_zh2en['train'] = data_zh2en['train'].select(range(DATA_SIZE))
    print(f"my2en train size: {len(data_my2en['train'])}")
    print(f"en2zh train size: {len(data_en2zh['train'])}")
    if BACK_TRANSLATION:
        data_en2my['train'] = data_my2en['train'].map(formatting_func_en2my)
        data_zh2en['train'] = data_en2zh['train'].map(formatting_func_zh2en)
    data_my2en['train'] = data_my2en['train'].map(formatting_func_my2en)
    data_en2zh['train'] = data_en2zh['train'].map(formatting_func_en2zh)
    if BACK_TRANSLATION:
        data_my2en['train'] = concatenate_datasets([data_my2en['train'], data_en2my['train']])
        data_en2zh['train'] = concatenate_datasets([data_en2zh['train'], data_zh2en['train']])
    print(data_en2zh)
    print(len(data_en2zh['train']))
    print(data_en2zh['train'][0])
    data_my2en['train'] = data_my2en['train'].map(lambda example: tokenize_function(example, tokenizer), batched=True)
    data_en2zh['train'] = data_en2zh['train'].map(lambda example: tokenize_function(example, tokenizer), batched=True)
    data_my2en['train'] = data_my2en['train'].remove_columns(['Index','Chinese','Malay','English','text'])
    data_en2zh['train'] = data_en2zh['train'].remove_columns(['Index','Chinese','Malay','English','text'])
    print(data_en2zh)
    print(len(data_en2zh['train']))
    print(data_en2zh['train'][0])   
    return data_my2en, data_en2zh

def load_rag_data(RAG_DATA_SIZE, data_file_path, source, target):
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
        
    data = load_dataset("json", data_files=data_file_path)
    rag_data = data["train"].select(range(len(data["train"]) - RAG_DATA_SIZE, len(data["train"])))
    titles = rag_data["Malay"] 
    data = rag_data["Chinese"]
    titles = [f"{prompt1}{title}" for title in titles]
    data = [f"{prompt2}{content}" for content in data]
    print(f"(RAG) Extracted {len(titles)} titles and {len(data)} data entries.")
    rag_dataset = {"titles": titles, "data": data}
    return rag_dataset
    
    
def calculate_average_tokens(dataset, field_names=["input_ids","labels"]):
    for field_name in field_names:
        lengths = [len(example[field_name]) for example in dataset]
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        print(f"Average tokens ({field_name}): {avg_length}")
        print(f"Maximum tokens ({field_name}): {max_length}")
        # return avg_length, max_length

    
    
def sft(model, tokenizer, dataset_my2en, dataset_en2zh, test_size = 0.2):
    dataset_my2en = dataset_my2en['train'].train_test_split(test_size=test_size, seed=42)
    tokenized_train_my2en = dataset_my2en['train']
    tokenized_eval_my2en = dataset_my2en['test']

    dataset_en2zh = dataset_en2zh['train'].train_test_split(test_size=test_size, seed=42)
    tokenized_train_en2zh = dataset_en2zh['train']
    tokenized_eval_en2zh = dataset_en2zh['test']
    
    # Set parameters for LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,  # Rank for the LoRA layers
        lora_alpha=LORA_ALPHA,  # Scaling factor
        lora_dropout=0.1,  # Dropout rate for LoRA layers
        target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj"],  # Layers to apply LoRA (q,v)
        # target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",  # Apply bias control
        task_type="CAUSAL_LM"  # Task type for causal language modeling
    )

    # Apply LoRA to the model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False
        
    # model = accelerator.prepare(model)
    torch.cuda.empty_cache()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        num_train_epochs=EPOCH,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,  # Accumulate gradients over 8 steps (simulate larger batch size)
        per_device_train_batch_size=BATCH_SIZE,  # Keep the batch size small
        per_device_eval_batch_size=BATCH_SIZE,
        logging_dir="../log",
        logging_steps=8,
        save_steps=1000,
        evaluation_strategy="steps",
        fp16=True,                      # Mixed precision
        learning_rate=LEARNING_RATE,           
        # weight_decay=0.01, 
        load_best_model_at_end=True,
        # log_level="debug",  # Ensure debug-level logs are recorded
        # log_level_replica="warning",  # Optional: Reduce logging for replicas
        # metric_for_best_model="eval_loss",  # Metric for selecting the best model
        # greater_is_better=False,  # Set to False for loss-based metrics
        # save_total_limit=3,  # Limit number of saved checkpoints
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=2,       # Stop if no improvement for 2 evaluation steps
        early_stopping_threshold=0.0     # Minimum change to qualify as an improvement
    )

    # Trainer setup for Malay → English (Stage 1)
    trainer_stage_1 = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_my2en,
        eval_dataset=tokenized_eval_my2en,
        tokenizer=tokenizer,
        # callbacks=[early_stopping_callback]
    )
    trainer_stage_1.train()
    print(f"Loss after training step: {trainer_stage_1.state.log_history[-1]}")
    torch.cuda.empty_cache()
    
    # Trainer setup for English → Chinese (Stage 2)
    trainer_stage_2 = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_en2zh,
        eval_dataset=tokenized_eval_en2zh,
        tokenizer=tokenizer,
        # callbacks=[early_stopping_callback]
    )
    trainer_stage_2.train()
    print(f"Loss after training stage 2: {trainer_stage_2.state.log_history[-1]}")
    torch.cuda.empty_cache()
    
    return model, tokenizer
    
    
def save_pretrained_model(model, tokenizer, save_model_path):
    # Save the model after training
    model.save_pretrained(save_model_path)
    tokenizer.save_pretrained(save_model_path)
    print(f"Model saved to {save_model_path}")
    torch.cuda.empty_cache()

def done():
    with open(f"{save_model_path}/done.txt","w",encoding="utf-8") as wf:
        wf.write("done.")
        
def setup_retriever(corpus):
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Use a pre-trained embedding model
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(corpus_embeddings.cpu().numpy())
    return model, index, corpus

def retrieve_context(query, model, index, corpus, top_k=1):
    query_embedding = model.encode(query, convert_to_tensor=True)
    distances, indices = index.search(query_embedding.cpu().numpy(), top_k)
    return [corpus[i] for i in indices[0]]

def main():
    if TRAIN:
        logging = setup_logging(log_file_path)
        model, tokenizer = load_model(path_to_model)
        data_my2en, data_en2zh = load_train_dataset(data_file_path, DATA_SIZE, tokenizer, BACK_TRANSLATION)
        # calculate_average_tokens(data_en2zh["train"])
        model_trained, tokenizer = sft(model, tokenizer, data_my2en, data_en2zh, TEST_SIZE)
        save_pretrained_model(model_trained, tokenizer, save_model_path)
        done()
    if TEST:
        print(save_model_path)
        print(write_test_path)
        print(write_bleu_path)
        if RAG:
            rag_dataset = load_rag_data(RAG_DATA_SIZE, data_file_path, SOURCE, TARGET)
        else:
            rag_dataset = None
        ind_ls = run_test(save_model_path, data_file_path_test, write_test_path, 
                 TEST_DATA_SIZE, TEST_ONE_SAMPLE, REFLECTION,
                 RAG, rag_model_path, retrieved_dataset_path, index_path, 
                 context_encoder_path, rag_dataset, SOURCE, TARGET)
        bleu_score = calculate_model_bleu(write_test_path, write_bleu_path, ind_ls)
        
    

if __name__ == '__main__':
    main()
    

    
