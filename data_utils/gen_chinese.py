# -*- coding: utf-8 -*-

from chatgpt_api import chatgpt_api
import csv
import ast
import pandas as pd
import os
from tqdm import tqdm
import logging

# Set up logging to log to a file
def setup_logging(log_file):
    with open(log_file, 'w'): 
        pass
    logging.basicConfig(filename=log_file, 
                    level=logging.INFO,  # Set the log level to INFO
                    format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

def log_print(message):
    """Helper function to log print statements."""
    print(message)  # This will print to the console
    logging.info(message)  # This will save to the log file
    
    
def instructions(field):
    instruction = f"""您是一名知识渊博的{field}学专家。请将所提供的英文段落翻译成中文，确保所有科学术语得到精准翻译。请准确传达其含义，并使用{field}学领域的特定术语。保留科学严谨性和清晰度，而不是严格按照原文的词序翻译。"""
    return instruction

sample = [
    "Of these ten neurons, eight are predicted correctly to be sufficient to activate MN9. We previously found that five of the ten are required for sugar feeding initiation. Of these five, three are predicted by our computational model to cause a greater than 20% decrease in MN9 firing, and one of the others is predicted to cause a statistically significant decrease in MN9 firing, but less than 20%, when silenced.",
    "Phantom strongly synapses onto Scapula—a neuron that is also predicted to be inhibitory; Scapula, in turn, synapses onto Roundup, the pre-MN with the strongest predicted silencing phenotype. We speculate that activation of Phantom inhibits Scapula, potentially permitting Roundup and MN9 firing. Because the basal firing rate of all neurons in the model is 0, activation of inhibitory neurons in the model, in the absence of other input, cannot alter the firing of downstream neurons.",
    "A further explanation for incorrect predictions could be neuromodulation, which is not accounted for in our model. Particular neurons may be subject to neuromodulation, causing their activity to differ from predictions based on connectivity. Alternatively, neurons that express neuromodulators may be poorly modeled."
]

answer = [
        "在这十个神经元中，八个被预测为足以激活MN9。我们之前发现，其中五个神经元对于启动糖喂食反应是必需的。在这五个神经元中，三个位点被我们的计算模型预测为会导致MN9放电大于20%的下降，另一个则预测在静默时会导致MN9放电下降，尽管小于20%，但具有统计学显著性。",
        "Phantom 强烈地与 Scapula 发生突触连接，而 Scapula 也是一个被预测为抑制性的神经元；Scapula 又与 Roundup 发生突触连接，后者是具有最强预测沉默表型的前MN。我们推测，Phantom 的激活可能抑制 Scapula，从而可能允许 Roundup 和 MN9 放电。由于模型中所有神经元的基础放电率为0，在没有其他输入的情况下，激活抑制性神经元不能改变下游神经元的放电。",
        "对于错误预测的进一步解释可能是神经调节，这是我们的模型未考虑的因素。某些神经元可能受到神经调节的影响，导致其活动与基于连接性的预测不同。另一个可能性是表达神经调节因子的神经元模型不够准确。"
    ]

def gen_chinese(english_sentence, field):
    instruction = instructions(field)
    response = chatgpt_api(instruction=instruction, data=english_sentence)
    return response

          
if __name__ == '__main__':
    field = "生物"
    english_sentence_ls = []
    history_process_ls = []
    index_ls = []
    input_file = '../data/biology_brain.csv'
    write_file = '../data/biology_brain_cn.csv'
    history_file = '../data/gen_history_cn.csv'
    setup_logging("../log/gen_cn.log")
    
    with open(input_file, mode='r', newline='', encoding='utf-8') as f1, open(write_file, mode='a', newline='', encoding='utf-8') as f2, open(history_file, mode='a', newline='', encoding='utf-8') as f3:
        reader = csv.reader(f1)
        writer = csv.writer(f2)
        writer2 = csv.writer(f3)
        all_data = sum(1 for row in reader)
        f1.seek(0)
        next(reader)
        # Calculate the row number before appending
        f2.seek(0, 2)  # Move the file pointer to the end
        rows_before = sum(1 for row in open(write_file, mode='r', encoding='utf-8'))
        row_num = rows_before
        f3.seek(0, 2)  
        process_before = sum(1 for row in open(history_file, mode='r', encoding='utf-8'))
        process_num = process_before

        if row_num == 0:
            writer.writerow(['Index', 'English', 'Chinese'])
        if process_num == 0:
            writer2.writerow(['Processed Sentence'])
            
        if os.path.exists(history_file):
            try:
                df = pd.read_csv(history_file)
                history_process_ls = df["Processed Sentence"].tolist()
            except:
                pass
        
        with tqdm(total=all_data-process_num, desc="Processing Sentences", unit="sentence") as pbar:    
            for dat in reader:
                # print(dat)
                index = dat[0]
                english_sentence = dat[1]
                if english_sentence in history_process_ls:
                    continue
                while True:
                    try:
                        chinese_sentence = gen_chinese(english_sentence, field)
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        break
                # print(gen_chinese)
                writer.writerow([index, english_sentence, chinese_sentence])
                row_num += 1
                writer2.writerow([english_sentence])
                process_num += 1
                pbar.update(1)
                log_print(f"Process num: {process_num-1} / {all_data-1}")
                # log_print(f"Progress: {pbar.n}/{pbar.total} sentences processed ({(pbar.n/pbar.total)*100:.2f}%)")
                # break
                
                
        rows_written = row_num - rows_before
        log_print(f"Rows written before: {rows_before-1}")
        log_print(f"Rows written in this process: {rows_written}")
        log_print(f"Total data rows: {row_num-1}")        
        log_print(f"Total process data: {process_num-1} / {all_data-1}")  
        log_print(f"Data written to {write_file}")   
    
    




