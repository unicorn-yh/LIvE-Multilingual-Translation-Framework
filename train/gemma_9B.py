# # pip install accelerate
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# tokenizer = AutoTokenizer.from_pretrained("Gemma-9B")
# model = AutoModelForCausalLM.from_pretrained("Gemma-9B", 
#                                              offload_folder="model", 
#                                             #  torch_dtype=torch.float16,
#                                              device_map="auto")

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("mps")

# outputs = model.generate(**input_ids, max_new_tokens=50)
# print(tokenizer.decode(outputs[0]))


from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Load the tokenizer for Gemma-9B (you can replace with mBART or MarianMT if needed)
tokenizer = AutoTokenizer.from_pretrained("./Gemma-9B")  # Replace with your model path or correct model

# Example data
data = {
    "train": {
        "source": [
            "Apa khabar?", 
            "Saya sedang belajar terjemahan."
        ],
        "target_A": [
            "How are you?", 
            "I am learning translation."
        ],
        "target_B": [
            "你好吗？", 
            "我在学习翻译。"
        ]
    },
    "test": {
        "source": [
            "Selamat pagi!", 
            "Apa yang kamu lakukan?"
        ],
        "target_A": [
            "Good morning!", 
            "What are you doing?"
        ],
        "target_B": [
            "早上好！", 
            "你在做什么？"
        ]
    }
}

# Convert to Hugging Face dataset format
dataset = Dataset.from_dict(data)
dataset = DatasetDict({
    'train': Dataset.from_dict(data["train"]),
    'test': Dataset.from_dict(data["test"])
})
print(dataset["train"].column_names)

# Tokenize the dataset
def tokenize_function(examples):
    source = tokenizer(examples["source"], truncation=True, padding="max_length", max_length=512)
    target_A = tokenizer(examples["target_A"], truncation=True, padding="max_length", max_length=512)
    target_B = tokenizer(examples["target_B"], truncation=True, padding="max_length", max_length=512)
    
    return {
        "input_ids": source["input_ids"],
        "attention_mask": source["attention_mask"],
        "labels_A": target_A["input_ids"],
        "attention_mask_A": target_A["attention_mask"],
        "labels_B": target_B["input_ids"],
        "attention_mask_B": target_B["attention_mask"]
    }

# Apply tokenization
tokenized_train = dataset["train"].map(tokenize_function, batched=True)
tokenized_test = dataset["test"].map(tokenize_function, batched=True)

# print(tokenized_train)

# breakpoint()


from peft import get_peft_model, LoraConfig

# Define the LoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank for the LoRA layers
    lora_alpha=16,  # Scaling factor
    lora_dropout=0.1,  # Dropout rate for LoRA layers
    target_modules=["q_proj", "v_proj"],  # Layers to apply LoRA
    bias="none",  # Apply bias control
    task_type="CAUSAL_LM"  # Task type for causal language modeling
)

# Load your Gemma-9B model (pretrained)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./Gemma-9B")

# Apply LoRA to the model
model = get_peft_model(model, lora_config)


from transformers import Trainer, TrainingArguments

# Define training arguments
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=1000,
    evaluation_strategy="steps"
)

# Trainer setup for Malay → English (Stage 1)
trainer_stage_1 = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
)

# Train the model on Malay → English
trainer_stage_1.train()

# Now fine-tune on English → Chinese (Stage 2)
trainer_stage_2 = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_train,
    tokenizer=tokenizer,
)

# Train the model on English → Chinese
trainer_stage_2.train()

# Save the model after training
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Example translation
inputs = tokenizer("Apa khabar?", return_tensors="pt", truncation=True, padding="max_length", max_length=512)
outputs = model.generate(inputs["input_ids"])

# Decode the output
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translated_text)  # Output will be in English

