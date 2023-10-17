import os
import glob
import re
import argparse 
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, LlamaTokenizer, AutoModelForSeq2SeqLM

def run_eval(model_path, model_id, prompt, num_answers, temperature, top_p, answer_path):
    ans = get_model_answers(model_path, model_id, prompt, num_answers, temperature, top_p)
    save_model_answer(ans, model_id, answer_path)

@torch.inference_mode()
def get_model_answers(model_path, model_id, prompt, num_answers, temperature=0.7, top_p=0.9, top_k=0):
    if model_id == "chatglm-6b":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).half().cuda()
    elif "t5" in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).cuda()
    elif "llama" in model_id:
        tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).cuda()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16
        ).cuda()

    ans = []
    input_ids = tokenizer([prompt]).input_ids
    for i in trange(num_answers):
        output_ids = model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=True,
            temperature=temperature,
            max_new_tokens=512,
            top_k=top_k,
            top_p=top_p,
        )
        output_ids = output_ids[0][len(input_ids[0]):]
        outputs = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        ans.append(outputs)
    return ans

def save_model_answer(ans, model_id, answer_path):
    data = []
    for response in ans:
        words = clean(response)
        data_all = [str(args.temperature)]
        data_all.extend(words)
        data.append(data_all)
    fn = f'{answer_path}{model_id}/data.txt'
    if os.path.exists(fn):
        with open(fn) as f:
            text = f.read()
            f.close()
        text_add = '\n\n'.join(["\n".join(i) for i in data])
        text_all = "\n\n".join([text, text_add])
    else:
        text_all = '\n\n'.join(["\n".join(i) for i in data])
    os.makedirs(f'{answer_path}{model_id}', exist_ok=True)
    with open(fn,'w') as f:
        f.write(text_all)
        f.close()

def clean(ans):
    if "\n" in ans and len(ans.split("\n")) >= 10:
        words = ans.split("\n")
    elif "," in ans:
        words = ans.split(',')
    elif "*" in ans:
        words = ans.split('*')
    else:
        return None
    words = [validate(word) for word in words]
    if len(words)>10:
        words = words[:10]
    if None not in words and len(list(set(words))) == 10:
        return words
    else:
        return None

def validate(word):
    """Clean up word and find best candidate to use"""
    # Strip unwanted characters
    clean = re.sub(r"[^a-zA-Z ]+", "", word).strip().lower()
    if len(clean) <= 1:
        return None # Word too short
    if " " in clean:
        clean = clean.split(" ")
        for i in clean:
            if i != "a":
                return i
    else:
        return clean

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="vicuna-13b")
    parser.add_argument("--model-id", type=str, default="vicuna-13b")
    parser.add_argument("--prompt", type=str, default="Please write 10 nouns in English that are as irrelevant from each other as possible, \
    in all meanings and uses of the words. Please note that the words you write should have only single word, \
    only nouns (e.g., things, objects, concepts), and no proper nouns (e.g., no specific people or places). Your answer:" )
    parser.add_argument("--answer-path", type=str, default="Top_p/")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num_answers", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    run_eval(
        args.model_path,
        args.model_id,
        args.prompt,
        args.num_answers,
        args.temperature,
        args.top_p,
        args.answer_path
    )