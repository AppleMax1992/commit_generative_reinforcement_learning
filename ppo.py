import random
import torch
import wandb
import time
import os
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import sys
sys.path.append('/root/autodl-tmp/commit_generative_reinforcement_learning')

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead

# -------------------------
# 0) PPO Config
# -------------------------
commit_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}

config = PPOConfig(
    model_name="/root/autodl-tmp/models/codet5-base",
    steps=51200,
    learning_rate=1.41e-5,
    remove_unused_columns=False,
    log_with="wandb",
    batch_size=32,
    mini_batch_size=32,
)

# ⚠️ 你原来 txt_in_len=5 太小了，会把 prompt 截成 5 个 token（几乎啥也没看到）
# 建议：输入长度至少 256/512（按显存调整）
txt_in_len = 512
txt_out_len = 64
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

policy_name = "/root/autodl-tmp/models/codet5-base"
tokenizer = AutoTokenizer.from_pretrained(policy_name)

policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(policy_name)
ref_policy = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(policy_name)

# -------------------------
# 1) 统一 Prompt（不使用 control）
# -------------------------
import re

def truncate_diff(diff: str, max_chars: int = 1500) -> str:
    diff = "" if diff is None else str(diff)
    return diff[:max_chars]

def build_gen_query(diff: str, max_chars: int = 1500) -> str:
    d = truncate_diff(diff, max_chars=max_chars)
    return (
        "<TASK>commit_message_generation</TASK>\n\n"
        "<DIFF>\n"
        f"{d}\n"
        "</DIFF>\n\n"
        "<MESSAGE>\n"
    )

def build_rm_prompt(diff: str, message: str, max_chars: int = 1500) -> str:
    d = truncate_diff(diff, max_chars=max_chars)
    m = (message or "").strip()
    return (
        "<TASK>security_patch_identification</TASK>\n\n"
        "<DIFF>\n"
        f"{d}\n"
        "</DIFF>\n\n"
        "<MESSAGE>\n"
        f"{m}\n"
        "</MESSAGE>\n"
    )

_TAG_RE = re.compile(r"</?(TASK|DIFF|MESSAGE)>", flags=re.IGNORECASE)

def extract_message(gen_text: str) -> str:
    if not gen_text:
        return ""
    text = gen_text.split("</MESSAGE>")[0]
    text = _TAG_RE.sub("", text)
    return text.strip()

# 你已有的语言质量 reward（保留你的版本也行）
def language_quality_reward(text: str) -> float:
    if text is None:
        return -0.5
    if len(text.split()) < 5:
        return -0.3
    blacklist = ["http", "www", "license", "copyright", "©"]
    if any(b in text.lower() for b in blacklist):
        return -0.5
    # 惩罚把 diff 片段吐出来（可选但很有效）
    if "diff --git" in text or "@@" in text:
        return -0.4
    if len(text) > 400:
        return -0.1
    return 0.0

# -------------------------
# 2) Load dataset (CSV) 并构造 PPO 需要字段
# -------------------------
dataset = load_dataset(
    "csv",
    data_files="/root/autodl-tmp/CommitFit/dataset/Ghadhab/dataset.csv",
    split="train",
)

# 你数据列名是 "diffs"（按你代码）
DIFF_COL = "diffs"

def map_to_ppo_fields(x):
    diff = x[DIFF_COL] if DIFF_COL in x else ""
    query = build_gen_query(diff)

    tok = tokenizer(
        query,
        truncation=True,
        max_length=txt_in_len,
        padding=False,               # 动态 padding 交给 collator
        return_attention_mask=True,
    )

    return {
        "diff": diff,                # ✅ 保留 diff 供 reward 拼 rm_prompt
        "query": query,              # ✅ 字符串（给 wandb logs/debug）
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
    }

dataset = dataset.map(map_to_ppo_fields, batched=False)
dataset = dataset.select(range(min(len(dataset), 20480)))

# 只保留用得到的列
keep_cols = ["diff", "query", "input_ids", "attention_mask"]
dataset = dataset.remove_columns([c for c in dataset.column_names if c not in keep_cols])

dataset.set_format("pytorch", columns=["input_ids", "attention_mask"])

def collator(data):
    # TRL 经典写法：返回 list；padding 由 PPOTrainer 内部或 generate 时处理
    return {k: [d[k] for d in data] for k in data[0]}

ppo_trainer = PPOTrainer(
    config=config,
    model=policy,
    ref_model=ref_policy,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
)

# -------------------------
# 3) Reward model pipeline
# -------------------------
if ppo_trainer.accelerator.num_processes == 1:
    pipe_device = 0 if torch.cuda.is_available() else -1
else:
    # 多卡时 pipeline 用 accelerator.device 有时会触发 transformers pipeline 的 bug
    # 更稳：仍用 0 或 -1（你若多卡，我建议先单卡跑通）
    pipe_device = 0 if torch.cuda.is_available() else -1

commit_pipe = pipeline(
    "text-classification",
    "/root/autodl-tmp/CommitFit/notebooks/E-3-best(70%)/my_awesome_model/checkpoint-390",
    device=pipe_device,
    truncation=True,
)

# 你 reward model 的 label 名必须确认一下
# print(commit_pipe.model.config.id2label)
# 假设你要奖励的目标类是 LABEL_1（你按自己任务改）
TARGET_LABEL = "LABEL_1"

def pipe_outputs_to_scores(outputs, target_label: str) -> list[float]:
    """
    outputs: List[List[Dict(label, score)]]
    return:  List[float] 每条样本 target_label 的 score
    """
    scores = []
    for out in outputs:
        # out 是一个 list[dict]，因为 top_k=None
        s = 0.0
        for d in out:
            if d["label"] == target_label:
                s = float(d["score"])
                break
        scores.append(s)
    return scores

# -------------------------
# 4) Generation kwargs
# -------------------------
generation_kwargs = {
    "min_length": 1,
    "top_k": 0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": txt_out_len,
    "eos_token_id": tokenizer.eos_token_id,
}

# reward 权重（可调）
w_cls = 1.0
w_lang = 0.2
reward_clip = 5.0

# -------------------------
# 5) PPO Training Loop（对齐后的版本）
# -------------------------
for epoch in range(10):
    for batch in tqdm(ppo_trainer.dataloader):
        logs, game_data = {}, {}

        # batch 是 dict of lists/tensors（collator 输出）
        diffs = batch["diff"]          # List[str]
        query_texts = batch["query"]   # List[str]（我们在 map 已经构造好了）

        game_data["query"] = query_texts

        # query_tensors: List[Tensor]
        query_tensors = batch["input_ids"]  # List[Tensor]（每条是 [L] 或 [1,L] 取决于 format）
        # 确保是 list[Tensor] 且在正确 device
        query_tensors = [qt.to(ppo_trainer.accelerator.device) for qt in query_tensors]

        # 1) generate response
        response_tensors = []
        for q in query_tensors:
            resp = ppo_trainer.generate(q, **generation_kwargs)
            response_tensors.append(resp.squeeze())

        responses = [
            tokenizer.decode(
                r,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            for r in response_tensors
        ]
        game_data["response"] = responses

        # 2) build RM texts (diff + extracted_message)
        clean_msgs = [extract_message(t) for t in responses]
        rm_texts = [build_rm_prompt(d, m) for d, m in zip(diffs, clean_msgs)]

        # 3) reward = pipeline_score + language_quality
        # pipeline batch 推理
        pipe_out = commit_pipe(rm_texts, **commit_pipe_kwargs, batch_size=len(rm_texts))
        cls_scores = pipe_outputs_to_scores(pipe_out, TARGET_LABEL)  # list[float]
        lang_scores = [language_quality_reward(t) for t in responses] # list[float]

        device_t = query_tensors[0].device
        rewards = []
        for cs, ls in zip(cls_scores, lang_scores):
            r = w_cls * cs + w_lang * ls
            r = float(max(min(r, reward_clip), -reward_clip))
            rewards.append(torch.tensor(r, device=device_t, dtype=torch.float32))

        # 4) PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # logging
        rs = torch.stack(rewards)
        stats["env/reward_mean"] = float(rs.mean().item())
        stats["env/reward_std"] = float(rs.std(unbiased=False).item())
        stats["env/reward_cls_mean"] = float(np.mean(cls_scores))
        stats["env/reward_lang_mean"] = float(np.mean(lang_scores))

        ppo_trainer.log_stats(stats, game_data, rewards)

# -------------------------
# 6) Save
# -------------------------
policy.save_pretrained("codet5-msgs-ctrl")  # 你可以改名 codet5-msgs-ppo
tokenizer.save_pretrained("codet5-msgs-ctrl")
