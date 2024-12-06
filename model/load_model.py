from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login

# Replace with your actual Hugging Face token
login(token="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")

# Load the pre-trained model from Hugging Face
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")

# download the model files locally
model.save_pretrained('./Gemma-9B')
tokenizer.save_pretrained('./Gemma-9B')
