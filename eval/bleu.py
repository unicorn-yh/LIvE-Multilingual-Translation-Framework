
from bleu import calculate_model_bleu

# model = "R16_A16_LR1e-05_7000_E1_RAG0"
model = "new"
test_path = "/Users/olivia/Project/Gemma-Multilingual-Model/evaluation/"+model+".jsonl"
write_path_data = "/Users/olivia/Project/Gemma-Multilingual-Model/score/"+model+".jsonl"
write_path_score="/Users/olivia/Project/Gemma-Multilingual-Model/bleu/"+model+".txt"

avg_bleu = calculate_model_bleu(test_path, write_path_data, write_path_score)
print(avg_bleu)