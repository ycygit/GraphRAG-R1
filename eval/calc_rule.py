import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import re
import string
import json
import jsonlines
from collections import Counter, defaultdict
import statistics  # 导入统计模块用于计算方差和中位数

# 预定义
sbert_model = None

# 对字符串进行归一化，清除无用的符号、空格、大小写等干扰项
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)
    def lower(text):
        return text.lower()
    def replace_underscore(text):
        return text.replace("_", " ")
    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))

# 把true映射成yes,false映射成no
def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s

# 完全一致得1分，否则0分
def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(bool_mapping(ground_truth))

def cover_exact_match_score_1(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()
    return all(ground in pre_list for ground in ground_list)

def cover_exact_match_score_2(prediction, ground_truth):
    pre_list = normalize_answer(bool_mapping(prediction)).split()
    ground_list = normalize_answer(bool_mapping(ground_truth)).split()
    for i in range(len(pre_list) - len(ground_list) + 1):
        if pre_list[i:i+len(ground_list)] == ground_list:
            return True
    return " ".join(ground_list) in " ".join(pre_list)

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    if (
        normalized_prediction in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0, 0, 0
    if (
        normalized_ground_truth in ["yes", "no", "noanswer"]
        and normalized_prediction != normalized_ground_truth
    ):
        return 0, 0, 0
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def sbert_cosine_similarity(prediction, ground_truth):
    global sbert_model
    if sbert_model is None:
        print("正在加载 SBERT 模型，请稍等...")
        from sentence_transformers import SentenceTransformer, util
        sbert_model = SentenceTransformer('shibing624/text2vec-base-chinese')
        sbert_model.util = util
        print("SBERT 模型加载完成！")
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))
    pred_embedding = sbert_model.encode(normalized_prediction, convert_to_tensor=True)
    gt_embedding = sbert_model.encode(normalized_ground_truth, convert_to_tensor=True)
    cosine_sim = sbert_model.util.pytorch_cos_sim(pred_embedding, gt_embedding)
    return cosine_sim.item()

def rouge_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    # 检查 normalized_prediction 和 normalized_ground_truth 是否为空
    if not normalized_prediction or not normalized_ground_truth:
        return 0, 0, 0  # 返回默认值

    rouge = Rouge()
    scores = rouge.get_scores([normalized_prediction], [normalized_ground_truth])[0]
    return scores["rouge-1"]["f"], scores["rouge-2"]["f"], scores["rouge-l"]["f"]

# 遍历所有 ground-truth 找最优
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores = []
    if metric_fn.__name__ == "f1_score":
        for gt in ground_truths:
            f1, p, r = metric_fn(prediction, gt)
            scores.append((f1, p, r))
        return max(scores, key=lambda x: x[0])
    else:
        for gt in ground_truths:
            score = metric_fn(prediction, gt)
            scores.append(score)
        return max(scores)

def read_jsonl(file_path):
    with jsonlines.open(file_path, "r") as reader:
        return [obj for obj in reader]

def eval(config):
    data = read_jsonl(config["input_file"])
    print(f"Eval {len(data)} examples from {config['input_file']}")

    metrics = {}
    enabled_metrics = {}

    if config.get("use_em", False):
        enabled_metrics["em"] = exact_match_score
        metrics["em"] = 0
    if config.get("use_cover_em_1", False):
        enabled_metrics["cover_em_1"] = cover_exact_match_score_1
        metrics["cover_em_1"] = 0
    if config.get("use_cover_em_2", False):
        enabled_metrics["cover_em_2"] = cover_exact_match_score_2
        metrics["cover_em_2"] = 0
    if config.get("use_f1", False):
        enabled_metrics["f1"] = f1_score
        metrics["f1"] = 0
        metrics["precision"] = 0
        metrics["recall"] = 0
    if config.get("use_sbert_sim", False):
        enabled_metrics["sbert_sim"] = sbert_cosine_similarity
        metrics["sbert_sim"] = 0
    if config.get("use_rouge", False):
        enabled_metrics["rouge"] = rouge_score
        metrics["rouge-1"] = 0
        metrics["rouge-2"] = 0
        metrics["rouge-l"] = 0

    valid_count = 0
    total_retrieve_num = 0  # 用于统计总检索次数
    retrieve_num_list = []  # 用于存储所有检索次数，方便计算方差和中位数
    total_response_times_sum = 0  # 用于统计所有 response_times 的总和
    count_1 = 0
    total_generate_times_sum = 0
    total_retrieve_tokens_sum = 0
    total_reasoning_tokens_sum = 0
    total_response_times_count = 0
    total_generate_times_count = 0
    retrieve_num_count = defaultdict(int)  # 用于统计每个 retrieve_num 的出现次数


    flag=0
    for d in data:
        # flag+=1
        # print(flag)
        pred = d.get("pred_ans", None)
        # pred=pred.split(",")[0]
        retrieve_num = d.get("retrieve_num", None)  # 假设每个数据项中包含 retrieve_num 字段
        response_times = d.get("response_times", [])  # 获取 response_times 字段
        generate_times = d.get("generate_times", [])
        retrieve_tokens = d.get("retrieve_tokens", [])
        reasoning_tokens = d.get("reasoning_tokens", [])

        # 如果 pred 是 None 或 空字符串，跳过,只评估pred_ans有内容的数据
        if not pred or (isinstance(pred, str) and pred.strip() == ""):
            continue

        valid_count += 1
        if retrieve_num is not None:
            total_retrieve_num += retrieve_num
            retrieve_num_list.append(retrieve_num)  # 添加到列表中
            count_1 += 1

        # 统计 retrieve_num 的出现次数
        if retrieve_num is not None:
            retrieve_num_count[retrieve_num] += 1

        # 计算每条数据的平均响应时间
        if response_times:
            total_response_times_sum += sum(response_times)
            total_response_times_count += len(response_times)
        
        if generate_times:
            total_generate_times_sum += sum(generate_times)
            total_generate_times_count += len(generate_times)
        
        if retrieve_tokens:
            total_retrieve_tokens_sum += sum(retrieve_tokens)
        
        if reasoning_tokens:
            total_reasoning_tokens_sum += reasoning_tokens[-1]

        gts = d["answer"] if isinstance(d["answer"], list) else [d["answer"]]
        for name, func in enabled_metrics.items():
            if name == "f1":
                f1, p, r = metric_max_over_ground_truths(func, pred, gts)
                metrics["f1"] += f1
                metrics["precision"] += p
                metrics["recall"] += r
            elif name == "rouge":
                r1, r2, rl = metric_max_over_ground_truths(func, pred, gts)

                metrics["rouge-1"] += r1
                metrics["rouge-2"] += r2
                metrics["rouge-l"] += rl
            else:
                score = metric_max_over_ground_truths(func, pred, gts)
                metrics[name] += float(score)
    if valid_count == 0:
        print("没有有效的预测结果可以评估！")
        return metrics
    
    # 计算平均检索次数
    average_retrieve_num = total_retrieve_num / valid_count if valid_count > 0 else 0
    # 计算检索次数的方差和中位数
    retrieve_num_variance = statistics.variance(retrieve_num_list) if len(retrieve_num_list) > 1 else 0
    retrieve_num_median = statistics.median(retrieve_num_list) if retrieve_num_list else 0

    # 计算每一轮的平均响应时间
    overall_average_response_time = total_response_times_sum / total_response_times_count if total_response_times_count > 0 else 0

    print(f"\n有效评估样本数: {valid_count}")
    print("\nResult:")
    for k, v in metrics.items():
        if k in ["precision", "recall"]:
            continue
        value = round(v / valid_count * 100, 2)
        print(f"{k}: {value}")

    # 明确输出 recall 的值
    if "precision" in metrics:
        precision_value = round(metrics["precision"] / valid_count * 100, 2)
        print(f"precision: {precision_value}")

    if "recall" in metrics:
        recall_value = round(metrics["recall"] / valid_count * 100, 2)
        print(f"recall: {recall_value}")

    # 打印 retrieve_num 的统计结果
    print("\nRetrieve_num 统计结果：")
    for retrieve_num, count in retrieve_num_count.items():
        print(f"retrieve_num={retrieve_num}: {count} 条")
    
    # 打印平均检索次数、方差和中位数
    print(f"\n平均检索次数: {average_retrieve_num:.2f}")
    print(f"检索次数的方差: {retrieve_num_variance:.4f}")
    print(f"检索次数的中位数: {retrieve_num_median:.2f}")
    # 打印平均响应时间
    print(f"每一轮检索的平均响应时间: {overall_average_response_time:.4f}")
    print(f"每条数据的平均响应时间: { total_response_times_sum / count_1:.4f}")
    print(f"每一轮数据的平均生成时间: { total_generate_times_sum / total_generate_times_count:.4f}")
    print(f"每条数据的平均生成时间: { total_generate_times_sum / len(data):.4f}")
    print(f"每一轮数据的平均检索tokens: { total_retrieve_tokens_sum / total_response_times_count:.2f}")
    print(f"每条数据的平均检索tokens: { total_retrieve_tokens_sum / count_1:.2f}")
    print(f"每一轮数据的平均推理tokens: { total_reasoning_tokens_sum / total_generate_times_count:.2f}")
    print(f"每条数据的平均推理tokens: { total_reasoning_tokens_sum / len(data):.2f}")

    return metrics

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    eval(config)