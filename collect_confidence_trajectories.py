import pandas as pd
import torch
from tqdm import tqdm
from typing import Dict, Any
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import logging
import math



def calculate_answer_probabilities_batch( 
                              batch_size,
                              prompt, 
                              cot_steps,
                              model, 
                              tokenizer,
                              response,
                              answer_prefix,
                              options,
                              r1_model=False
                              ):

    closing_tag = "</think>" if r1_model else ""

    no_cots = [prompt + closing_tag + answer_prefix]

    partial_cots = [prompt + "".join(cot_steps[:i+1]) + closing_tag  for i in range(len(cot_steps))]

    partial_cots = no_cots + partial_cots


    partial_probabilities = []
    full_logits = []
    option_indecies = []
    option_ids = torch.tensor([
            tokenizer(op, return_tensors="pt", add_special_tokens=False)["input_ids"] for op in options 
            ])
    for i in range(0, len(partial_cots), batch_size):
        batch = partial_cots[i:i + batch_size]
        if not batch:
            continue


        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        seq_lengths = attention_mask.sum(dim=1) - 1

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits

            batch_indicies = torch.arange(len(batch))
            last_token_logits = logits[batch_indicies, seq_lengths, :]

            option_logits = last_token_logits[:, option_ids]
            partial_probabilities.extend(option_logits.cpu().tolist())

            full_probs = last_token_logits.softmax(dim=-1)
            option_probs = full_probs[:, option_ids]

            full_logits.append(option_probs.cpu().tolist())


            option_indecies.append(option_ids.cpu().tolist())
        del input_ids, outputs, logits, last_token_logits, inputs, option_logits, attention_mask, full_probs, option_probs
        torch.cuda.empty_cache()
    return partial_probabilities, full_logits, option_indecies



def separate_cot_and_answer(cot: str, answer_regex: str, r1_model: bool = False):
    matches = list(re.finditer(answer_regex, cot))
    if matches:
        answer_match = matches[-1]  # Use the last match
    else:
        answer_match = None

    if answer_match:
        prefix = answer_match.group(1)
        answer = answer_match.group(2)
        if r1_model:
            cot = cot.split("</think>")[0]
        else:
            cot = cot.removesuffix(prefix+answer)
    else:
        return None, None, None

    sentences = re.split(r'\n\n', cot)
    if sentences and sentences[-1].strip() == "":
        sentences.pop()
    all_steps = [s + "\n\n" for s in sentences]
    
    num_steps = len(all_steps)
    max_steps = 20
    if num_steps > max_steps:
        cot_steps = []
        group_size = math.ceil(num_steps / max_steps)
        for i in range(0, num_steps, group_size):
            group = all_steps[i:min(i + group_size, num_steps)]
            cot_steps.append("".join(group))
        
    else:
        cot_steps = all_steps
    response = answer
    return prefix, response, cot_steps

def run_answer_consistency(
    dataset_path: str,
    model,
    tokenizer,
    options,
    answer_regex=None,
    r1_model=False,
) -> Dict[str, Any]:
    """
    Calculate answer consistency for a dataset using a Hugging Face model.
    
    Args:
        dataset_path: Path to the parquet file containing the dataset.
        model: Hugging Face model for generating answers.
        tokenizer: Tokenizer associated with the model.
        max_length: Maximum sequence length for the model.
        
    Returns:
        Dataset 
    """
    # Load dataset
    df = pd.read_parquet(dataset_path)
    cot_steps_column = 'cot_steps'
    # partial_perplexities_column = 'partial_perplexities'
    target_column = 'target'
    probs_target_column = 'answer_probs'
    early_answers_column = 'early_answers'
    full_logits_column = 'full_logits'
    options_indicies_column = 'options_indicies'
    if options_indicies_column not in df.columns:
        df[options_indicies_column] = None
    if full_logits_column not in df.columns:
        df[full_logits_column] = None
    if probs_target_column not in df.columns:
        df[probs_target_column] = None

    if target_column not in df.columns:
        df[target_column] = None 

    if cot_steps_column not in df.columns:
        df[cot_steps_column] = None



    # if partial_perplexities_column not in df.columns:
    #     df[partial_perplexities_column] = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for index, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['full_prompt']
        cot = row['generated_text']
        if 'answer_probs' in df.columns:
            if df.at[index, probs_target_column] is not None:
                continue
        answer_prefix, response, cot_steps = separate_cot_and_answer(cot, answer_regex, r1_model)
        if response is None or cot_steps is None:
            continue
        try:
            if "QwQ-32B" in dataset_path or "DeepSeek-R1-Distill-Qwen-32B" in dataset_path or "Qwen3-32B" in dataset_path:
                batch_size = 1
            else:
                batch_size = 4
            answer_probs, full_logits, option_indicies = calculate_answer_probabilities_batch(
                                    batch_size,
                                    prompt,
                                    cot_steps,
                                    model, 
                                    tokenizer,
                                    response,
                                    answer_prefix,
                                    options,
                                    r1_model=False)
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            df.at[index, cot_steps_column] = None
            df.at[index, target_column] = None
            df.at[index, probs_target_column] = None
            df.at[index, early_answers_column] = None
            continue
        df.at[index, cot_steps_column] = cot_steps
        df.at[index, target_column] = response
        df.at[index, probs_target_column] = answer_probs
        df.at[index, full_logits_column] = full_logits
        df.at[index, options_indicies_column] = option_indicies

    if early_answers_column in df.columns:
        df.drop(columns=[early_answers_column], inplace=True)

    df.to_parquet(dataset_path)

    return None



def main():
    """Main function to run the answer consistency analysis."""
    

    parser = argparse.ArgumentParser(description="Analyze answer consistency in a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset parquet file")
    parser.add_argument("--model", type=str, required=True, help="Hugging Face model name")

    args = parser.parse_args()
    
    logging.basicConfig(filename=f'contribution_analysis_{args.model.split("/")[1]}.log', level=logging.INFO, format='%(message)s', filemode='w')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    
    if hasattr(model, "enable_flash_attention"):
        model.enable_flash_attention()

    if "csqa" in args.dataset:
        answer_regex = r'(Answer:)(.*)'
        options = [" A", " B", " C", " D", " E"]
    elif "strategyqa" in args.dataset:
        answer_regex = r'(Answer:)(.*)'
        options = [" Yes", " No"]
    elif "lsat" in args.dataset:
        answer_regex = r'(Answer:)(.*)'
        options = [" A", " B", " C", " D", " E"]
    elif "mm_musr" in args.dataset:
        answer_regex = r'(Answer:)(.*)'
        options = [" A", " B"]
    elif "ta_musr" in args.dataset:
        answer_regex = r'(Answer:)(.*)'
        options = [" A", " B", " C"]
    elif "op_musr" in args.dataset:
        answer_regex = r'(Answer:)(.*)'
        options = [" A", " B", " C", " D", " E"]
    elif "gpqa" in args.dataset:
        answer_regex = r'(Answer:)(.*)'
        options = [" A", " B", " C", " D"]
    else:
        raise ValueError("Invalid dataset")
    
    model.eval()
    if "R1" in args.model or "QwQ" in args.model or "Qwen3-32B" in args.model:
        r1_model = True
    else:
        r1_model = False
    answer_regex = answer_regex
    run_answer_consistency(
        args.dataset,
        model,
        tokenizer,
        options,
        answer_regex,
        r1_model,
    )
        

if __name__ == "__main__":
    main()
