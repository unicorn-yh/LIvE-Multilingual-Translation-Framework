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
        
def instructions():
    instruction = f"""Please correct the spelling and grammar in the following sentences. Remove any references to sources. Do not add other information which is not given."""
    return instruction

sample = [
    "A strong increase in brain volume occurred only with the appearance of Homo habilis about 2 mya having 700 cm 3. This means that brain size of our ancestors remained constant for 1. 5 mya despite strong environmental changes.",
    "Intelligence de fi ned in such a manner has developed several times independently during evolution, e. g., in cephalopods, social insects, some teleost fi shes, some birds, and mammals.",
    "However, brain size does not increase proportionally with body size, but “lags behind”, i. e., with an exponent of 0.",
    "6–0. 8, which is due to the fact that with an increase in body size brains become absolutely larger, but relatively smaller – this is called negative brain allometry  [ 44 ]. As a consequence, in small mice or insectivores brain volume may constitute 10 % or more of body volume, while in the blue whale, the largest living animal, the brain makes up only 0.",
    "01 % or even less of body mass [ 49 ] ;. Primates, in general, have higher RBS than all other groups of mammals. The human brain has a weight of 1,250–1,450 g on average and represents about 2 % of body mass."
]

answer = [
    "A strong increase in brain volume occurred only with the appearance of Homo habilis about 2 million years ago, having 700 cm³. This means that the brain size of our ancestors remained constant for 1.5 million years despite strong environmental changes.",
    "Intelligence defined in such a manner has developed several times independently during evolution, e.g., in cephalopods, social insects, some teleost fishes, some birds, and mammals.",
    'However, brain size does not increase proportionally with body size, but "lags behind," i.e., with an exponent of 0.6–0.8, which is due to the fact that with an increase in body size, brains become absolutely larger, but relatively smaller—this is called negative brain allometry.',
    'As a consequence, in small mice or insectivores, brain volume may constitute 10% or more of body volume, while in the blue whale, the largest living animal, the brain makes up only 0.01% or even less of body mass. Primates, in general, have higher RBS than all other groups of mammals. The human brain has a weight of 1,250–1,450 g on average and represents about 2% of body mass.'
]

def preprocess_pivot(english_sentence_ls):
    content = english_sentence_ls
    data = [
        (str(sample), str(answer)),
        (str(content), None),
    ]
    instruction = instructions()
    response = chatgpt_api(instruction=instruction, data=data)
    return response

if __name__ == '__main__':
    field = "Biology"
    english_sentence_ls = []
    history_process_ls = []
    input_file = './biology_v1.csv'
    write_file = '../data/biology_brain.csv'
    history_file = '../data/process_history.csv'
    setup_logging("../log/preprocess.log")
    
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
            writer.writerow(['Index', 'English'])
        if process_num == 0:
            writer2.writerow(['Processed Sentence'])
            
        if os.path.exists(history_file):
            try:
                df = pd.read_csv(history_file)
                history_process_ls = df["Processed Sentence"].tolist()
            except:
                pass
        with tqdm(total=all_data-process_num, desc="Processing Sentences", unit="sentence") as pbar:    
            for english_sentence in reader:
                if english_sentence[0] in history_process_ls:
                    continue
                english_sentence_ls.append(english_sentence[0])
                if len(english_sentence_ls) == 10:
                    # print(english_sentence_ls)
                    while True:
                        try:
                            postprocess_ls = preprocess_pivot(english_sentence_ls)
                            postprocess_ls = ast.literal_eval(postprocess_ls)
                            break
                        except Exception as e:
                            print(f"Error: {e}")
                    for sentence in postprocess_ls:
                        writer.writerow([row_num, sentence])
                        row_num += 1
                    for sentence in english_sentence_ls:
                        writer2.writerow([sentence])
                        process_num += 1
                    english_sentence_ls = []
                    pbar.update(len(postprocess_ls))
                    log_print(f"Process num: {process_num-1} / {all_data-1}")
                    # log_print(f"Progress: {pbar.n}/{pbar.total} sentences processed ({(pbar.n/pbar.total)*100:.2f}%)")
                    # break
                        
            # Handle remaining sentences if any
            if english_sentence_ls:
                while True:
                    try:
                        postprocess_ls = preprocess_pivot(english_sentence_ls)
                        postprocess_ls = ast.literal_eval(postprocess_ls)
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                for sentence in postprocess_ls:
                    writer.writerow([row_num, sentence])
                    row_num += 1
                for sentence in english_sentence_ls:
                    writer2.writerow([sentence])
                    process_num += 1
                english_sentence_ls = []
                pbar.update(len(postprocess_ls))
                log_print(f"Process num: {process_num-1} / {all_data-1}")
                # log_print(f"Progress: {pbar.n}/{pbar.total} sentences processed ({(pbar.n/pbar.total)*100:.2f}%)")
                
        rows_written = row_num - rows_before
        log_print(f"Rows written before: {rows_before-1}")
        log_print(f"Rows written in this process: {rows_written}")
        log_print(f"Total data rows: {row_num-1}")        
        log_print(f"Total process data: {process_num-1} / {all_data-1}")        
                      
    
    




