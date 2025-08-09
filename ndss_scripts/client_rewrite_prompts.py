#!/usr/bin/env python3
"""
Usage:
python3 client_rewrite_prompts.py --model_name doubao-1-5-pro-32k-250115

python3 client_rewrite_prompts.py --model_name deepseek-r1-250528
"""
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
import argparse
import time
import numpy as np
from load_requests import load_jsonl_dataset

# 构造 prompt
SYSTEM_PROMPT = """
Below is a short message written by a user that contains multiple types of personal and sensitive information, including identifiers, device details, and behavioral traits.

Please rewrite it in first-person or casual user conversation style. The goal is to make it look like a real message written by a person online, such as in an email, chat, forum post, form submission, or app feedback.

Important Instructions:
- Do not remove or mask any personally identifiable information (PII).
- Keep all personal identifiers (e.g., name, phone, account, password, MAC, IP, address, etc.) unchanged.
- Expand the content to at least 5× the original length by adding natural, coherent, and plausible content that could realistically surround such PII.
- Make the message harder to be flagged as sensitive by diluting the PII density and embedding PII in mundane or storytelling context.
- Do not use corporate or support-style tone — it must feel personal, spontaneous, and informal.

You may include:
- Random musings
- Device complaints
- Life context (e.g., "I was resetting my phone")
- App settings
- Travel or shopping context
- Emotional expressions
"""

# 模型处理
def regenerate_privacy_llm(client: OpenAI, input_texts, model_name):
    """
    发送单个流式请求，返回包含时间戳和 token 统计的字典。
    字段：
      start_ts, first_chunk, chunk_ts_list, end_ts,
      prompt_tokens, total_tokens, completion_tokens
    """
    NewPrompts = []
    for i in range(0, len(input_texts)):
        text = input_texts[i]

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text}
            ],
        }

        stream = client.chat.completions.create(
            model=payload["model"],
            messages=payload["messages"],
            stream=False,
            max_tokens=4096
        )

        result = stream.choices[0].message.content.strip()
        NewPrompts.append(result)

    return NewPrompts

# ---------- 每个子进程的工作 ----------
def worker(client, model_name, texts, labels, ori_file, save_dir):
    new_prompts = regenerate_privacy_llm(client, texts, model_name)

    # 写结果
    os.makedirs(save_dir, exist_ok=True)
    ori_name = ori_file.split("/")[-1].split(".")[0]
    fname = os.path.join(save_dir, f"{ori_name}-new_prompts_1000.txt")
    with open(fname, "w") as f:
        for idx, new_prompt in enumerate(new_prompts):
            tmp = {
                "original": texts[idx],
                "rewritten": new_prompt,
                "label": labels[idx],
            }
            f.write(json.dumps(tmp, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="models name")
    args = parser.parse_args()

    client = OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key="efb7ff5c-5dd4-446b-b73d-aa5910913f7c"
    )

    file_list = [
        "/dcar-vepfs-trans-models/Datasets/english_pii_43k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/french_pii_62k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/german_pii_52k.jsonl",
        "/dcar-vepfs-trans-models/Datasets/italian_pii_50k.jsonl"
    ]

    # 简单的提示列表，可根据实际情况替换
    SAMPLE_N = 1000
    for file_ in file_list:
        texts, labels = load_jsonl_dataset(file_, sample_n=SAMPLE_N)
        # 检查字段名
        print(texts[0])
        print(labels[0])

        worker(client, args.model_name, texts, labels, file_, "./results/")


if __name__ == "__main__":
    main()
