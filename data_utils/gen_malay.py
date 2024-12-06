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
    instruction = f"""Anda adalah seorang pakar dalam bidang {field} yang sangat berpengetahuan. Sila terjemahkan perenggan dalam bahasa Inggeris yang diberikan ke dalam bahasa Melayu, pastikan semua istilah saintifik diterjemahkan dengan tepat. Sampaikan maksudnya dengan tepat dan gunakan istilah khusus dalam bidang {field}. Kekalkan ketelitian dan kejelasan saintifik, bukannya terjemahan yang terlalu mengikut susunan perkataan asal."""
    return instruction



def gen_malay(english_sentence, field):
    instruction = instructions(field)
    response = chatgpt_api(instruction=instruction, data=english_sentence)
    return response

if __name__ == '__main__':
    field = "Biologi"
    english_sentence_ls = []
    history_process_ls = []
    index_ls = []
    input_file = '../data/biology_brain.csv'
    write_file = '../data/biology_brain_my.csv'
    history_file = '../data/gen_history_my.csv'
    setup_logging("../log/gen_my.log")
    
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
            writer.writerow(['Index', 'English', 'Malay'])
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
                        malay_sentence = gen_malay(english_sentence, field)
                        break
                    except Exception as e:
                        print(f"Error: {e}")
                        break
                writer.writerow([index, english_sentence, malay_sentence])
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
    
    




