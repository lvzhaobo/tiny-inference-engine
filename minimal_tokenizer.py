"""
最小词表库（minimal_tokenizer.py）
==================================
minimal_llm.py 的配套：把"硬编码 ID"升级为"真查词表"。

设计目标（教学约定）：
- 词表 45 词（存于同目录 vocab.json，HF 字典格式 {词: ID}），覆盖 7 个领域 + 末尾追加苹果例句演示词（ID 42-44）
- 嵌入矩阵 45×4（存于同目录 embeddings.json）：词表查到 ID 后按 ID 取行得 4 维嵌入行
- encode 用"最长匹配"（教学简化版，真实 BPE 按 merges 规则合并）
- decode 反查词表还原文本
- 未登录词（不在词表）：先试单字，仍不在则输出 <UNK>

与真实 tokenizer 的对应关系：
- vocab.json ≈ 真实模型的 vocab.json（词 -> ID 映射表；本教学版用 JSON 数组，顺序即 ID）
- 最长匹配 ≈ 真实 BPE 的 merge 规则（优先用更长/更常见的组合）
- <UNK> 处理 ≈ 真实 tokenizer 的字节级兜底（真实模型极少走到 UNK，因为能拆到字节）

对接 minimal_llm.py 时需要改的：
- VOCAB = 6 改为 len(VOCAB_WORDS)
- ids = torch.tensor(tok.encode("上海东方明珠")) 替代硬编码 [0, 1, 2]
"""

import json
from pathlib import Path

# ---------------- 词表 + 嵌入矩阵（从 json 加载，顺序即 ID）----------------
DATA_DIR = Path(__file__).parent
with open(DATA_DIR / "vocab.json", encoding="utf-8") as f:
    _vocab = json.load(f)
# HF 字典格式 {词: ID} -> 按 ID 排序重建词表数组（顺序即 ID）
VOCAB_WORDS = [w for w, _ in sorted(_vocab.items(), key=lambda kv: kv[1])]
with open(DATA_DIR / "embeddings.json", encoding="utf-8") as f:
    EMBEDDINGS = json.load(f)


class MinimalTokenizer:
    """最小词表：word -> ID 查表 + 最长匹配切分。"""

    def __init__(self, words):
        self.word2id = {w: i for i, w in enumerate(words)}
        self.id2word = {i: w for w, i in self.word2id.items()}
        self.max_len = max(len(w) for w in words)
        self.unk_id = self.word2id["<UNK>"]
        self.eos_id = self.word2id["<EOS>"]

    def encode(self, text):
        """文本 -> ID 序列：从当前位置尝试最长的词表命中（最长匹配）。"""
        ids = []
        i = 0
        while i < len(text):
            # 从"剩余文本最长可能命中"的长度往下试
            matched = None
            for length in range(min(self.max_len, len(text) - i), 0, -1):
                frag = text[i:i + length]
                if frag in self.word2id:
                    matched = frag
                    break
            if matched is not None:
                ids.append(self.word2id[matched])
                i += len(matched)
            else:
                # 未登录：尝试单字，仍不在词表则记 <UNK>
                ids.append(self.unk_id)
                i += 1
        return ids

    def decode(self, ids):
        """ID 序列 -> 文本：逐 ID 反查词表后拼接。"""
        return "".join(self.id2word.get(i, "<UNK>") for i in ids)


class EmbeddingTable:
    """嵌入矩阵：ID -> 4 维嵌入行（45x4，来自 embeddings.json）。"""

    def __init__(self, rows):
        self.rows = rows

    def embed(self, ids):
        """ID 序列 -> 嵌入行列表（按 ID 当行号取行）。"""
        return [self.rows[i] for i in ids]


if __name__ == "__main__":
    tok = MinimalTokenizer(VOCAB_WORDS)
    print(f"词表大小：{len(VOCAB_WORDS)} 词；<UNK>={tok.unk_id}，<EOS>={tok.eos_id}")
    print()

    demos = [
        "上海东方明珠",          # 词表有"东方明珠"整词 -> 2 个 token
        "我在上海投资芯片",      # 日常 + 地名 + 财经 + 科技 混合
        "上海特斯拉工厂",        # "特斯拉""工厂"未登录 -> 单字兜底 / <UNK>
        "2025年上海股票市场",    # 数字整词 + 地名 + 财经
        "吃了苹果",              # 追加词全命中：吃/了/苹果 -> [31, 24, 42]
    ]
    for s in demos:
        ids = tok.encode(s)
        back = tok.decode(ids)
        pieces = [tok.id2word[i] for i in ids]
        print(f"输入：{s}")
        print(f"  ID 序列：{ids}")
        print(f"  切分：{pieces}")
        print(f"  decode 还原：{back}")
        print()

    # 粒度对比演示：词表里没有"东方明珠"整词时会切成什么
    print("粒度对比（假设词表里没有'东方明珠'整词）：")
    tmp = MinimalTokenizer([w for w in VOCAB_WORDS if w != "东方明珠"] + ["东方", "明珠"])
    ids2 = tmp.encode("上海东方明珠")
    print(f"  切成：{[tmp.id2word[i] for i in ids2]}，token 数 2 -> 3")
    print()

    # ID -> 向量：词表查到 ID 后，从嵌入矩阵按 ID 取行
    print("ID -> 嵌入行（embeddings.json，按 ID 取行）：")
    emb = EmbeddingTable(EMBEDDINGS)
    for i in tok.encode("上海东方明珠"):
        print(f"  {tok.id2word[i]}(ID {i}) -> {emb.embed([i])[0]}")
    print()
    print("对接 minimal_llm.py：VOCAB 改为 45，ids = torch.tensor(tok.encode('上海东方明珠'))")
