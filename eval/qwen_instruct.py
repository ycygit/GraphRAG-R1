import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 设置为 ERROR 级别,只显示错误信息，不显示警告和信息级别的
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from Evaluator import RolloutEvaluator
from datasets import Dataset
from trl import GRPOConfig
import re
from peft import PeftModel, PeftConfig
import random

import numpy as np

def process_text(example, type=None, mode = 0):
    base_prompts = {
    "v0c":"""Answer the given question. \
Reasoning step by step. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <|begin_of_query|> query <|end_of_query|> and it will return the top searched results between <|begin_of_documents|> and <|end_of_documents|>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations, keep concise. For example, <answer> Yes </answer> or <answer> Miller Brewing </answer> .
User:{question}Assistant: <think>""",
       
    }

    question = example.get("question", "")
    if not question:
        raise ValueError("Missing 'question' field in example")

    if type not in base_prompts:
        raise ValueError(f"Unknown prompt type: {type}")

    example["prompt"] = base_prompts[type].format(question=question)
    return {
        "question": question,
        "prompt": base_prompts[type].format(question=question),
        "mode": mode,
    }

def run(
    input_json_path,
    output_jsonl_path,
    model_ckpt,
    prompt_type="v0c",
    mode= 0,
    search_url="http://127.0.0.1:8081/query",
    plus=False
):
    print("模型路径:")
    print(model_ckpt)
    # 加载adapter配置
    peft_config = PeftConfig.from_pretrained(model_ckpt)
    base_model_path = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 第二步：加载 adapter（SFT）权重
    model = PeftModel.from_pretrained(base_model, model_ckpt)
    model.eval()
    
    grpo_config = GRPOConfig(
        cache_implementation="dynamic",    
        temperature=0.001,
        top_p=1.0,
        top_k=50,
        min_p=0.0,
        max_completion_length=512,
        use_vllm=False,
        report_to="wandb"
    )
    
    trainer =  RolloutEvaluator(
        model=model,
        reward_funcs=[],  # 推理不需要reward
        args=grpo_config,
        processing_class=tokenizer,
        search_url=search_url
    )



    # 读取后200条
    with open(input_json_path, 'r', encoding='utf-8') as f:
        raw_lines = [line for line in f if line.strip()]  # 去掉空行
        total_lines = len(raw_lines)
        
        # 计算倒数第500条数据的起始索引
        start_index = max(0, total_lines - 200)  # 确保不会出现负索引
        raw_data = [json.loads(line) for line in raw_lines[start_index:]]
        
        print("评估数据条数:")
        print(len(raw_data))

    # 打开输出文件，准备写入，一边生成，一边写入
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for idx, example in tqdm(enumerate(raw_data), desc="Generating"):
            example = process_text(example, type=prompt_type, mode = mode)
            ground_truth_answer = raw_data[idx].get("answer", "")
            inputs = [{"prompt": example["prompt"],"mode":example["mode"],"plus":plus}]
            inputs_prepared = trainer._generate(inputs)  
            print("准备写入*******************************************")
            # 取出completion
            completion_ids = inputs_prepared["completion_ids"]
            completion_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)[0]
            retrieve_num = inputs_prepared["retrieve_num"]
            response_times = inputs_prepared['response_times']
            generate_times = inputs_prepared['generate_times']
            retrieve_tokens = inputs_prepared['retrieve_tokens']
            reasoning_tokens = inputs_prepared['reasoning_tokens']
            # 提取标签之间的内容作为答案
            match = re.search(r'<answer>(.*?)</answer>', completion_text, re.DOTALL)
            if match:
                answer = match.group(1).strip()
                print("预测答案*******************************************")
                print(answer)
            else:
                answer = ""
            result = {
                "question": example["question"],
                "generated_answer": completion_text.strip(),
                "pred_ans": answer,
                "answer": ground_truth_answer,
                "retrieve_num": retrieve_num,
                "response_times":response_times,
                "generate_times":generate_times,
                "retrieve_tokens":retrieve_tokens,
                "reasoning_tokens":reasoning_tokens
            }
            # 写入文件
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"Finished. Saved to {output_jsonl_path}")


if __name__ == "__main__":

    result_path='./result/qwen_instruct'
    checkpoint_path='../checkpoints/qwen_instruct_v2'
    os.makedirs(result_path, exist_ok=True)
    search_url='http://127.0.0.1:8090/query'

    run(
        input_json_path="../datasets/hotpotqa/Question.json",         
        output_jsonl_path=result_path+"/hotpotqa.jsonl",
        model_ckpt=checkpoint_path,
        prompt_type="v0c", 
        mode=0,      
        search_url=search_url                  
    )
    run(
        input_json_path="../datasets/2wikimultihop/Question.json",         
        output_jsonl_path=result_path+"/2wiki.jsonl",
        model_ckpt=checkpoint_path,
        prompt_type="v0c",
        mode=0,      
        search_url=search_url                          
    )
    run(
        input_json_path="../datasets/musique/Question.json",         
        output_jsonl_path=result_path+"/musique.jsonl",
        model_ckpt=checkpoint_path,
        prompt_type="v0c", 
        mode=0,      
        search_url=search_url                           
    )




