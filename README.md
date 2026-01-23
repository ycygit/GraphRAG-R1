# GraphRAG-R1

[![arXiv](https://img.shields.io/badge/arXiv-2508.00304-b31b1b.svg)](https://arxiv.org/abs/2507.23581)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**This is the official implementation of the paper "GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning" (accepted by WWW '26).**

GraphRAG-R1 is a Graph Retrieval-Augmented Generation framework enhanced with Process-Constrained Reinforcement Learning, designed to improve the performance of large language models on complex multi-hop reasoning tasks. This repository contains the implementation for evaluation and inference.

## 📖 Table of Contents
- [Environment Setup](#environment-setup)
- [Getting Started](#getting-started)
  - [1. Deploy the Retrieval Service](#1-deploy-the-retrieval-service)
  - [2. Download Model Weights](#2-download-model-weights)
  - [3. Environment Configuration](#3-environment-configuration)
  - [4. Run Evaluation](#4-run-evaluation)
- [Examples](#examples)
- [License](#license)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)


## Getting Started

### 1. Deploy the Retrieval Service
We use a retrieval service based on HippoRAG. Please navigate to the `server/` directory and follow the instructions in its README to configure the environment and start the service. Once deployed, you will obtain an access URL (e.g., `http://127.0.0.1:8090`).


> **Note on Datasets**: The provided retrieval service and evaluation scripts are configured for the three in-domain datasets (HotpotQA, MuSiQue, and 2Wiki), which are sourced from [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG). To evaluate on the out-of-domain dataset PopQA (from [PropRAG](https://github.com/ReLink-Inc/PropRAG)), you can build a corresponding retrieval service following the same method and adjust the evaluation scripts accordingly.


### 2. Download Model Weights
Download the pre-trained GraphRAG-R1 model weights from the link below and place them in the `checkpoints/` directory:

> Download link: [Coming soon]

### 3. Environment Configuration

We recommend using Python 3.11. You can create a conda environment with:

```bash
conda create -n graphrag_r1 python=3.11
conda activate graphrag_r1
```

Clone this repository and install dependencies:

```bash
git clone https://github.com/ChuanyueYu/GraphRAG-R1.git
cd GraphRAG-R1
pip install -r requirements.txt
```


## 4. Run Evaluation

We provide evaluation scripts to reproduce the experimental results in the paper. Navigate to the `eval/` directory:

```bash
cd eval
```

### **Version 1: Qwen-2.5-7B (Base)**
Edit the evaluation script `qwen_base.py` and set the following paths and parameters:

```python
result_path = './result/qwen_base'         # Path to save results
checkpoint_path = '../checkpoints/qwen_base_v1'  # Path to model weights
search_url = 'http://127.0.0.1:8090'    # Retrieval service URL
```

Run the evaluation to generate intermediate results:

```bash
python qwen_base.py
```

> **Note on Result Stability**: The outputs from the base model may exhibit slightly higher variance across runs due to the inherent randomness in generation and the high temperature setting. 

### **Version 2: Qwen-2.5-7B-Instruct**
Recognizing the potential for further improvement, we conducted additional tuning on the instruction-tuned model (Qwen-2.5-7B-Instruct). This version demonstrates stronger instruction-following capabilities and more stable reasoning behavior.

The evaluation process is identical in structure. Use the script `qwen_instruct.py` and specify the corresponding checkpoint:

```python
result_path = './result/qwen_instruct'    # Path to save results
checkpoint_path = '../checkpoints/qwen_instruct'  # Path to V2 model weights
search_url = 'http://127.0.0.1:8090'    # Retrieval service URL
```

Run the evaluation:
```bash
python qwen_instruct.py
```

After completion, you will obtain a JSONL file containing model outputs in the directory specified by `result_path`.

We provide two evaluation methods:

#### Traditional Evaluation Metrics

Edit `config["input_file"]` in `config.json` to point to your output file path, then run:

```bash
python calc_rule.py
```

This script will compute traditional metrics such as F1 and provide statistics on retrieval behavior.

#### LLM-based Evaluation (LLM-as-Judge)

Edit `eval_online.py` and add your output file path to the `input_files` list, and replace the API key with your own. Run:

```bash
python eval_online.py
```

The script will invoke a large language model for scoring and generate a `*_judge_qwen.jsonl` file along with evaluation results.


## Examples

We provide additional examples in the `examples/` directory. 


## License
This project is released under the [MIT License](LICENSE).

## Citation

If you find this work helpful for your research, please cite our paper:

```bibtex
@inproceedings{yu2025graphrag,
  title={GraphRAG-R1: Graph Retrieval-Augmented Generation with Process-Constrained Reinforcement Learning},
  author={Yu, Chuanyue and Zhao, Kuo and Li, Yuhan and Chang, Heng and Feng, Mingjian and Jiang, Xiangzhe and Sun, Yufei and Li, Jia and Zhang, Yuzhi and Li, Jianxin and others},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year={2026}
}
```

## Acknowledgements

Our implementation is built upon and inspired by the following excellent open-source projects:

- [HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) – retrieval service foundation
- [R1-Searcher](https://github.com/RUCAIBox/R1-Searcher) – reinforcement learning training framework foundation

---

If you encounter any issues, please feel free to open an Issue.



