"""
model.py：引擎专用的模型加载与批量前向
====================================
从 train_minimal/infer.py 抽取并升级为"批量 + KV 缓存"形态。引擎要同时
服务多个请求，每个请求序列长度不同，所以前向必须支持：

1. 变长 batch：不同长度的序列 pad 到本批最大长度，pad 位置用掩码挡住
   （pad 位置既不被人看、也不看别人）
2. KV 缓存：prefill 时整段 prompt 走层并把 K/V 留下；decode 时新 token
   只算一行，K/V 追加进缓存（与 infer_kv.py 同一思想，升级为批量版）

与 train_minimal/infer.py 的架构约定完全一致（否则权重对不上）：
  - 972 参数：45 词表，6 维 3 头 3 层纯 attention + 残差，无 FFN/LayerNorm
  - 位置向量 P(pos, j) = 0.01 + 0.04*pos + 0.01*j（固定公式，不训练）
  - 因果掩码 + ÷√2 缩放；logits = X @ LM.t()

正确性保证：prefill_batch 对单请求的结果必须与 infer.py 的 forward 逐位
一致；decode 增量步的结果必须与"整段重算"一致（demo 里做了断言验证）。
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from minimal_tokenizer import MinimalTokenizer, VOCAB_WORDS

DATA_DIR = Path(__file__).parent

# ---------------- 模型尺寸（与 train_minimal 一致）----------------
D_MODEL = 6
N_HEADS = 3
D_HEAD = 2
N_LAYERS = 3


# ================= 权重加载 =================

def load_weights():
    """trained_weights.json（14 个张量，HF 参数名）-> 模型参数。

    引擎用同一份权重，保证与 train.py / infer.py / attention_measure.py
    产出的模型完全一致。
    """
    with open(DATA_DIR / "trained_weights.json", encoding="utf-8") as f:
        w = json.load(f)
    E = torch.tensor(w["embed_tokens.weight"])            # [45, 6]
    LM = torch.tensor(w["lm_head.weight"])                # [45, 6]
    WQ, WK, WV, WO = [], [], [], []
    for layer in range(N_LAYERS):
        p = f"layers.{layer}.self_attn."
        WQ.append(torch.tensor(w[p + "q_proj.weight"]))
        WK.append(torch.tensor(w[p + "k_proj.weight"]))
        WV.append(torch.tensor(w[p + "v_proj.weight"]))
        WO.append(torch.tensor(w[p + "o_proj.weight"]))
    return E, LM, WQ, WK, WV, WO


def position(pos):
    """位置向量 P(pos, j) = 0.01 + 0.04*pos + 0.01*j（与 train.py 同公式）。"""
    return torch.tensor([0.01 + 0.04 * pos + 0.01 * j for j in range(D_MODEL)])


# ================= 批量前向（变长序列）=================

def _make_mask(B, lengths, S):
    """构造 [B, S, S] 布尔掩码：位置 (q, k) 有效当且仅当
    - q 在请求自身的有效长度内（lengths[i] 之后是 pad）
    - k 在请求自身的有效长度内
    - q >= k（因果：不能看未来）
    """
    causal = torch.tril(torch.ones(S, S, dtype=torch.bool))            # [S, S]
    valid = torch.arange(S)[None, :] < lengths[:, None]                # [B, S]
    # [B,S,1] & [1,S,S] -> [B,S,S]：q 轴与 k 轴各自对齐
    return valid[:, :, None] & causal[None, :, :]                      # [B, S, S]


def prefill_batch(ids_list, E, LM, WQ, WK, WV, WO):
    """批量 prefill：多个新请求的完整 prompt 并行穿过 3 层。

    参数：
      ids_list: list[Tensor[Si]]，每个请求的词元 ID 序列（长度 Si 可不同）
    返回（每请求一项的 list）：
      dict(x_last=[6],        # 最后一个位置的隐藏状态（decode 的起点）
           logits=[45],       # 最后一个位置的 logits（可产出第一个 token）
           K_caches=[[S,6]x3],# 每层 K 缓存（prefill 一次全部写入）
           V_caches=[[S,6]x3])# 每层 V 缓存
    """
    B = len(ids_list)
    S = max(len(ids) for ids in ids_list)
    lengths = torch.tensor([len(ids) for ids in ids_list])

    # 1) 嵌入 + 位置：pad 位置填 0（会被掩码挡住，数值不影响结果）
    X = torch.zeros(B, S, D_MODEL)
    for i, ids in enumerate(ids_list):
        Si = ids.shape[0]
        X[i, :Si] = E[ids] + torch.stack([position(p) for p in range(Si)])

    # 2) 掩码：pad 不可见、因果（与单请求版逐行等价）
    mask = _make_mask(B, lengths, S)                       # [B, S, S]

    K_caches, V_caches = [], []
    for layer in range(N_LAYERS):
        q = X @ WQ[layer]                                  # [B, S, 6]
        k = X @ WK[layer]
        v = X @ WV[layer]
        # 每请求留下自己的 K/V（含 pad 行，用时按长度截断）
        K_caches.append([k[i, :lengths[i]] for i in range(B)])
        V_caches.append([v[i, :lengths[i]] for i in range(B)])

        qh = q.view(B, S, N_HEADS, D_HEAD)                 # [B, S, 3, 2]
        kh = k.view(B, S, N_HEADS, D_HEAD)
        vh = v.view(B, S, N_HEADS, D_HEAD)
        raw = torch.einsum("bqhd,bkhd->bhqk", qh, kh) / math.sqrt(D_HEAD)
        raw = raw.masked_fill(~mask[:, None, :, :], float("-inf"))
        attn = F.softmax(raw, dim=-1)                      # [B, 3, S, S]
        out = torch.einsum("bhqk,bkhd->bqhd", attn, vh).reshape(B, S, D_MODEL) @ WO[layer]
        X = X + out                                        # 残差

    # 3) 每请求取自己最后一个有效位置的 logits 与隐藏状态
    logits_all = X @ LM.t()                                # [B, S, 45]
    results = []
    for i in range(B):
        li = int(lengths[i].item())
        results.append({
            "x_last": X[i, li - 1].clone(),
            "logits": logits_all[i, li - 1].clone(),
            "K_caches": [c[i] for c in K_caches],          # 每层 [Si, 6]
            "V_caches": [c[i] for c in V_caches],
        })
    return results


def decode_batch(states, new_ids, E, LM, WQ, WK, WV, WO):
    """批量 decode：所有活跃请求各前进一步（生成一个新词元）。

    与 prefill 的区别（对应《5-Decode详解》）：
      - 只算新词元一行：新词元的 q/k/v 是 [B, 6] 而不是 [B, S, 6]
      - 新词元的 k/v 先追加进缓存，q 再查询含自己在内的全部缓存
      - 因果掩码天然满足（缓存里只有历史词元 + 自己，没有未来）

    参数：
      states: list[dict]，每个请求的状态（x_last + K/V 缓存）
      new_ids: list[int]，每个请求刚生成的新词元 ID
    返回：
      list[dict]，每请求更新后的状态（x_last 换成新位置、logits 可继续采样、
      K/V 缓存已追加新词元）
    """
    B = len(states)
    # 每个请求的序列总长度 = 缓存长度 + 1（新词元）
    lengths = torch.tensor([states[i]["K_caches"][0].shape[0] + 1 for i in range(B)])
    S = int(lengths.max().item())

    # 1) 新词元嵌入 + 位置：[B, 6]，每行是"自己序列的第 li-1 个位置"
    x_new = torch.zeros(B, D_MODEL)
    for i in range(B):
        pos = int(lengths[i].item()) - 1
        x_new[i] = E[new_ids[i]] + position(pos)

    new_K, new_V = [], []
    for layer in range(N_LAYERS):
        q = x_new @ WQ[layer]                              # [B, 6]
        k = x_new @ WK[layer]
        v = x_new @ WV[layer]

        # 2) 新词元的 K/V 先追加进缓存（标准 KV cache 流程：先写缓存再查询），
        #    总长度 = 历史 L + 1，且新词元的 q 能查到自己的 k（因果 q>=k，
        #    自注意力不能丢——实测头 0 就是自注意头，丢了会完全错乱）
        new_K.append([torch.cat([states[i]["K_caches"][layer], k[i:i + 1]], dim=0)
                      for i in range(B)])
        new_V.append([torch.cat([states[i]["V_caches"][layer], v[i:i + 1]], dim=0)
                      for i in range(B)])

        # 3) 更新后的缓存 pad 成 [B, S, 6]（pad 行填 0，掩码挡住）
        K_hist = torch.zeros(B, S, D_MODEL)
        V_hist = torch.zeros(B, S, D_MODEL)
        for i in range(B):
            li = new_K[layer][i].shape[0]                 # 含新词元的总长度
            K_hist[i, :li] = new_K[layer][i]
            V_hist[i, :li] = new_V[layer][i]

        # 4) 单行注意力：q [B,3,2] × K [B,S,3,2] -> [B,3,S]
        qh = q.view(B, N_HEADS, D_HEAD)
        kh = K_hist.view(B, S, N_HEADS, D_HEAD)
        raw = torch.einsum("bhd,bkhd->bhk", qh, kh) / math.sqrt(D_HEAD)
        # 掩码：只看自己缓存里真实存在的词元（含刚写入的新词元自己）
        valid = torch.arange(S)[None, :] < lengths[:, None]          # [B, S]
        raw = raw.masked_fill(~valid[:, None, :], float("-inf"))
        attn = F.softmax(raw, dim=-1)                      # [B, 3, S]

        # 5) 加权求和 V + 残差（注意残差加在新词元的输入 x_new 上）
        vh = V_hist.view(B, S, N_HEADS, D_HEAD)
        out = torch.einsum("bhk,bkhd->bhd", attn, vh).reshape(B, D_MODEL) @ WO[layer]
        x_new = x_new + out

    # 6) 更新每请求状态
    logits_all = x_new @ LM.t()                            # [B, 45]
    results = []
    for i in range(B):
        results.append({
            "x_last": x_new[i].clone(),
            "logits": logits_all[i].clone(),
            "K_caches": [c[i] for c in new_K],
            "V_caches": [c[i] for c in new_V],
        })
    return results
