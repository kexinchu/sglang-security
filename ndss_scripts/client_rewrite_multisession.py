#!/usr/bin/env python3
"""
Usage:
python3 client_rewrite_multisession.py --model_name doubao-1-5-pro-32k-250115

python3 client_rewrite_prompts.py --model_name deepseek-r1-250528
"""
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from openai import OpenAI
import argparse
import random
import numpy as np
from load_requests import load_jsonl_dataset

# 构造 prompt
USER_PROMPT = """
You are an AI assistant designed for multi-turn conversations.
Generate a realistic multi-turn dialogue history (5 to 8 turns) between a user and an AI assistant.
The topic can be general knowledge, technology, travel, education, food, productivity, or daily life questions.
Avoid including any personal identifiable information (PII) such as names, phone numbers, addresses, or account credentials.

Format your output as a list of alternating "User:" and "Assistant:" lines, e.g.:

User: What are some good productivity apps?
Assistant: Some popular productivity apps include Notion, Todoist, Trello, and Evernote. They help you manage tasks, notes, and collaboration.
User: I’ve heard a lot about Notion. What makes it different?
Assistant: Notion combines note-taking, task management, and databases into a single customizable workspace...

Make the conversation natural and informative.
"""

# 模型处理
def regenerate_multi_session(client: OpenAI, NumSessions, model_name):
    """
    发送单个流式请求，返回包含时间戳和 token 统计的字典。
    字段：
      start_ts, first_chunk, chunk_ts_list, end_ts,
      prompt_tokens, total_tokens, completion_tokens
    """
    NewPrompts = []
    for i in range(0, NumSessions):
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": USER_PROMPT}
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
    multi_turn_histories = regenerate_multi_session(client, 200, model_name)

    # === 将历史对话按照高斯分布分配给真实请求 ===
    assigned_histories = []
    mean = 5
    std_dev = 2

    # 生成2000个整数，每个表示该request需要几个历史dialogue（可为0）
    num_history_per_request = np.clip(np.round(np.random.normal(mean, std_dev, size=2000)), 0, 10).astype(int)

    # 写结果 - warmup
    os.makedirs(save_dir, exist_ok=True)
    ori_name = ori_file.split("/")[-1].split(".")[0]
    fname = os.path.join(save_dir, f"{ori_name}-warmup.jsonl")
    with open(fname, "w") as f:
        for i, new_prompt in enumerate(multi_turn_histories):
            tmp = {
                "prompt": new_prompt,
                "label": 0
            }
            f.write(json.dumps(tmp, ensure_ascii=False) + "\n")


    for i in range(2000):
        history_count = num_history_per_request[i]
        assigned = random.choices(multi_turn_histories, k=history_count)
        assigned_histories.append(assigned)

    # === 构造requests
    os.makedirs(save_dir, exist_ok=True)
    ori_name = ori_file.split("/")[-1].split(".")[0]
    fname = os.path.join(save_dir, f"{ori_name}-actual.jsonl")
    with open(fname, "w") as f:
        for i, request in enumerate(texts):
            history_blocks = assigned_histories[i]
            try:
                tmp = {
                    "prompt": history_blocks[0] + "\n" + request,
                    "label": labels[i]
                }
                f.write(json.dumps(tmp, ensure_ascii=False) + "\n")
            except:
                continue

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
    SAMPLE_N = 2000
    for file_ in file_list:
        texts, labels = load_jsonl_dataset(file_, sample_n=SAMPLE_N)
        # 检查字段名
        print(texts[0])
        print(labels[0])

        worker(client, args.model_name, texts, labels, file_, "./results/")


if __name__ == "__main__":
    main()
