
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from typing import Any, Callable, Optional, Sized, Union
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from torch import nn
from trl.import_utils import is_deepspeed_available, is_rich_available, is_vllm_available
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from typing import List
import warnings
import copy
import time
import requests
from dataclasses import dataclass

if is_wandb_available():
    import wandb



from datasets import Dataset, IterableDataset
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
from transformers.utils import is_peft_available
if is_peft_available():
    from peft import PeftConfig, get_peft_model

@dataclass
class Samples:
    sequences: torch.Tensor
    retrieve_num: torch.Tensor
    response_times: torch.Tensor
    generate_times: torch.Tensor
    retrieve_tokens: torch.Tensor
    reasoning_tokens: torch.Tensor

class RolloutEvaluator(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        response_times=None,
        generate_times=None,
        retrieve_tokens=None,
        reasoning_tokens=None,
        search_url="http://127.0.0.1:8081/query"
    ):
        super().__init__(model, reward_funcs, args, train_dataset, eval_dataset, processing_class, reward_processing_classes, callbacks, optimizers, peft_config)
        self.processing_class=processing_class

        self.search_url=search_url
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            pad_token_id=processing_class.pad_token_id,
            bos_token_id=processing_class.bos_token_id,
            eos_token_id=processing_class.eos_token_id,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            cache_implementation=args.cache_implementation,
            stop_strings=["<|end_of_query|>","</answer>"]

        )


    @profiling_decorator
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        print("!!"*20)
        print(inputs)
        print(type(inputs))
        mode = "eval" if self.control.should_evaluate else "train"
        mode = "eval"
        if mode == "train":
            if self.state.global_step % self.num_iterations == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[self._step % self.args.gradient_accumulation_steps] = inputs
            else:
                inputs = self._buffered_inputs[self._step % self.args.gradient_accumulation_steps]
            self._step += 1
        else:
            # In evaluation, we don't reuse completions across multiple updates, so we don't need to buffer inputs.
            inputs = self._generate_and_score_completions(inputs)
        return inputs




    def _generate_with_retieve(self,unwrapped_model,prompt_ids,prompt_mask,mode,plus=False):
        print("*"*40)

        # import copy
        # 加载分词器
        response_times = []  # 用于存储每次检索的响应时间
        generate_times = [] # 用于存储每次生成的响应时间 
        retrieve_tokens = [] # 用于存储每次检索结果的token数量
        reasoning_tokens = [] # 用于存储每次推理的token数量
        qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
       

        # all_prompts = prompt_ids
        prompts_w_idx_dict = []
        # print("prompt_ids:",prompt_ids,type(prompt_ids))
        for i,(idx_w_prompt,idx_w_prompt_mask) in enumerate(zip(prompt_ids,prompt_mask)):
            idx=i
            prompt = idx_w_prompt
            prompts_w_idx_dict.append({"idx": idx, "prompt_ids": prompt,'prompt_mask':idx_w_prompt_mask})
        # 从这里开始加检索功能
        stop_tokens = ["<|end_of_query|>"]
        all_outputs = []


        if True:
            idx_w_prompt_part = prompts_w_idx_dict
            data_keys = ["prompt_ids" , "idx","prompt_mask"]
            ds = Dataset.from_dict({key: [d[key] for d in idx_w_prompt_part] for key in data_keys})

            finished_all_list=[]
            continued_answer = copy.deepcopy(idx_w_prompt_part)

            for t in range(11):
                # print(prompt_ids.tolist()[0],t)
                finished_texts = []
                continued_texts = []
                start_time = time.time()
                                
                if t>0:
                    prompt_inputs = self.processing_class(
                                    text=ds['prompt'], return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
                                )
                    prompt_inputs = Trainer._prepare_inputs(self,inputs=prompt_inputs)
                    temp_prompt_ids, temp_prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
                    prompt_completion_ids = unwrapped_model.generate(
                        temp_prompt_ids, attention_mask=temp_prompt_mask, generation_config=self.generation_config,tokenizer=self.processing_class
                    )
                    prompt_length = temp_prompt_ids.size(1)
                else:
                    prompt_completion_ids = unwrapped_model.generate(
                        torch.tensor(ds['prompt_ids'], device="cuda"), attention_mask=torch.tensor(ds['prompt_mask'], device="cuda"), generation_config=self.generation_config,tokenizer=self.processing_class
                    )
                    prompt_length = prompt_ids.size(1)
                end_time = time.time()
                generate_time = end_time-start_time
                generate_times.append(generate_time)


                query_list=[]
                for q, output in enumerate(torch.split(prompt_completion_ids,split_size_or_sections=1,  dim=0)):

                    prompt = self.processing_class.batch_decode(output[:,:prompt_length], skip_special_tokens=True)[0]
                    
                    idx = continued_answer[q]["idx"]
                    # stop_reason = output.outputs[0].stop_reason
                    generated_text = self.processing_class.batch_decode(output[:,prompt_length:], skip_special_tokens=True)[0]
                    retrieve_num_count = continued_answer[q].get("retrieve_num_count",0)
                    all_token_ids = (output) .tolist()[0]
                    output_token_ids = (output[:,prompt_length:]).tolist()[0]

                    if t == 10: #检索次数太多了，直接停掉，就是未完成

                        original_data = {
                            "idx":idx,
                            "prompt_ids":all_token_ids,
                            "response_ids":output_token_ids,
                            "retrieve_num_count" : retrieve_num_count
                                }

                        if original_data["response_ids"] == None:
                            print(f"Here 1,Why this response_ids is None???,{original_data}")
                            time.sleep(10)
                        finished_texts.append(original_data)
                        continue
                    tokens = qwen_tokenizer.tokenize(generated_text)
                    reasoning_tokens.append(len(tokens))

                    if "<|begin_of_query|>" in generated_text and (
                        generated_text.strip().endswith("<|end_of_query|>") or
                        generated_text.strip().endswith("<|end_of_query|><") or
                        generated_text.strip().endswith('<|end_of_query|>"') or 
                        generated_text.strip().endswith('<|end_of_query|>".') or
                        generated_text.strip().endswith('<|end_of_query|>",') or
                        generated_text.strip().endswith('<|end_of_query|>.') or
                        generated_text.strip().endswith('<|end_of_query|>;') or 
                        generated_text.strip().endswith('<|end_of_query|></') or
                        generated_text.strip().endswith('<|end_of_query|>>"') 
                    ):
                        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
                        print("进入检索")
                        # print('generated_text:',generated_text)

                        query = generated_text.split("<|begin_of_query|>")[-1].split("<|end_of_query|>")[0]
                        query = query.replace('"',"").strip()
                        query = " ".join(query.split())
                        if query: #开始检索
                            query_list.append(query)
                            retrieve_num_count += 1

                            original_data = {
                                "idx":idx,
                                "prompt":self.processing_class.batch_decode(output[:,:], skip_special_tokens=True)[0],
                                "prompt_ids":prompt_ids.tolist()[0],
                                "response_ids":output_token_ids,
                                "retrieve_num_count" : retrieve_num_count
                                }
                            
                            if original_data["response_ids"] == None:
                                print(f"Here 2,Why this response_ids is None???,{original_data}")
                                time.sleep(10)
                            continued_texts.append(original_data)

                        else: 
                            original_data = {
                            "idx":idx,
                            "prompt":prompt + generated_text,
                            "prompt_ids":prompt_ids.tolist()[0],
                            "response_ids":output_token_ids,
                            "retrieve_num_count" : retrieve_num_count
                                }
                            if original_data["response_ids"] == None:
                                print(f"Here 3,Why this response_ids is None???,{original_data}")
                                time.sleep(10)
                            finished_texts.append(original_data)

                    else: #生成结束
                        print("**********************************************************")
                        print("生成结束！！！")

                        original_data = {
                        "idx":idx,
                        "prompt_ids":prompt_ids.tolist()[0],
                        "response_ids":output[:,prompt_ids.size(1):].tolist()[0],
                        # "response_ids":output_token_ids,
                        "retrieve_num_count" : retrieve_num_count
                        }

                        finished_texts.append(original_data)
                        if original_data["response_ids"] == None:
                            print(f"Here 4,Why this response_ids is None???,{original_data}")
                            time.sleep(10)


                assert len(query_list) == len(continued_texts), "Error in len of query_list and continued_texts"
                if len(query_list)!=0:
                    payload = {
                        "query": query_list
                    }
                     # 增加连不上服务器的时候，再尝试两次的逻辑处理
                    try_count = 0
                    max_try = 5
                    start_time = time.time()  # 记录发送请求的时间
                    while try_count <= max_try:
                        

                        response = requests.post(self.search_url,json=payload,timeout=60)
                        print(response.status_code)
                        if response.status_code == 200:
                            end_time = time.time()  # 记录收到响应的时间
                            response_time = end_time - start_time  # 计算响应时间

                        
                            response_times.append(response_time)  # 存储响应时间

                            
                            # 每一轮检索得到的文本
                            result = response.json()
                            answers = result["result"]
                            for k in range(len(answers)): 
                                continued_text_now = copy.deepcopy(continued_texts[k]) 
                                doc_content = ''
                                
                                if mode == 0:
                                    facts = answers[k].get('facts', [])
                                    docs = answers[k].get('docs', [])
                                    first_three_docs = docs[:3]
                                    doc_content = "\n".join(first_three_docs)
                                   # 如果实体数量大于5，只取前5个；如果少于或等于5个，取全部
                                    facts = facts[:5] if len(facts) > 5 else facts
                                    facts_strings = [" ".join(fact) for fact in facts]
                                    facts_with_braces = "\n".join([f"({i+1}) {{ {fact} }}" for i, fact in enumerate(facts_strings)])

                                    doc_content = facts_with_braces + "\n" + doc_content
                                elif mode == 1:
                                    docs = answers[k].get('docs', [])
                                    docs = docs[:5]
                                    doc_content = "\n".join(docs)
                                elif mode == 2:
                                    facts = answers[k]['facts']
                                    facts_strings = [" ".join(fact) for fact in facts]
                                    facts_with_braces = "\n".join([f"({i+1}) {{ {fact} }}" for i, fact in enumerate(facts_strings)])
                                    doc_content = facts_with_braces + "\n" 
                                else :
                                    input("没有得到正确的检索模式数据,请检查！！！！！！！！！！！")

                                
                                # 统计每一轮检索的token数量
                                print("doc_content:",doc_content)
                                tokens = qwen_tokenizer.tokenize(str(doc_content))
                                retrieve_tokens.append(len(tokens))


                                continued_text_now["prompt"] = continued_text_now["prompt"] + "<|begin_of_documents|>\n" +  doc_content[:1024] + "<|end_of_documents|>\n\n"
                                continued_texts[k] = continued_text_now  
                            break
                        else:
                            try_count += 1
                            if try_count > max_try:
                                print("链接不上服务器")   
                                # input("点击回车我要开始编造啦)


                finished_all_list.extend(finished_texts)

                if len(continued_texts)==0:
                    data_keys_again = finished_texts[0].keys()
                    if len(finished_all_list) != len(idx_w_prompt_part):
                        time.sleep(20)
                        print("finished_all_list:",finished_all_list)
                        print("idx_w_prompt_part:",idx_w_prompt_part)
                        time.sleep(20)
                    assert len(finished_all_list) == len(idx_w_prompt_part) , "Error in len of finished_all_list and idx_w_prompt_part"
                    all_outputs.append(finished_all_list)
                    break
                else:
                    data_keys_again = continued_texts[0].keys()
                    ds = Dataset.from_dict({key: [d[key] for d in continued_texts] for key in data_keys_again})
                    continued_answer = copy.deepcopy(continued_texts)

        
        all_outputs_cat = [item for sublist in all_outputs for item in sublist]
        if len(all_outputs_cat) != len(prompts_w_idx_dict):
            time.sleep(20)
            print("[all_outputs_cat,prompts_w_idx_dict]:",[all_outputs_cat,prompts_w_idx_dict])
            time.sleep(20)
        assert len(all_outputs_cat) == len(prompts_w_idx_dict), "Error in len of all_outputs and prompts_w_idx_dict"
        all_outputs_sorted = sorted(all_outputs_cat, key=lambda x: x['idx'])

        samples_list = []
        micro_rollout_batch_size=1
        for i in range(0, len(all_outputs_sorted), micro_rollout_batch_size):
            
            outputs = all_outputs_sorted[i : i + micro_rollout_batch_size]
            if True:

                sequences = []
                retrieve_num = []
                
                for i, output in enumerate(outputs):
                        input_len = prompt_length
                        
                        
                        output_len = len(output["response_ids"])
                        sequences.extend(prompt_ids.tolist()[0] + output["response_ids"])
                        retrieve_num.append(output["retrieve_num_count"])
                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                retrieve_nums = torch.tensor(retrieve_num, device="cuda", dtype=torch.float)
                samples_list.append( 
                    Samples(
                        response_times = response_times,
                        generate_times = generate_times,
                        retrieve_tokens = retrieve_tokens,
                        reasoning_tokens=reasoning_tokens,
                        sequences=sequences,
                        retrieve_num=retrieve_nums,
                    )
                )
        torch.cuda.empty_cache()
        return samples_list 


    def _generate(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            text=prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self,inputs=prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        mode_value = inputs[0].get('mode', None)
        plus = inputs[0].get('plus', None)
        if mode_value is not None:
            print(f"The value of 'mode' is: {mode_value}")
        else:
            print("No 'mode' key found in the first dictionary.")

        print(mode_value)
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]



        with unwrap_model_for_generation(
            self.model_wrapped, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:


            samples_list=self._generate_with_retieve(unwrapped_model=unwrapped_model,prompt_ids=prompt_ids,prompt_mask=prompt_mask,mode= mode_value,plus=plus)
            prompt_completion_ids=[]
            retrieve_masks=[]
            retrieve_num = 0
            response_time = []
            generate_time = []
            retrieve_tokens = []
            reasoning_tokens = []
            for sample in samples_list:
                prompt_completion_ids.append(sample.sequences)
                response_time=sample.response_times
                generate_time=sample.generate_times
                retrieve_tokens=sample.retrieve_tokens
                reasoning_tokens=sample.reasoning_tokens
                retrieve_num = int(sample.retrieve_num.item())
            prompt_completion_ids=torch.cat(tuple(prompt_completion_ids), dim=0)


            
        # Compute prompt length and extract completion ids
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        
        
        return {
            "completion_ids": completion_ids,
            "retrieve_num": retrieve_num,
            "response_times": response_time,
            "generate_times": generate_time,
            "retrieve_tokens": retrieve_tokens,
            "reasoning_tokens": reasoning_tokens
        }
    

