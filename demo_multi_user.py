"""
demo_multi_user.py：多用户并发演示 + 串行 vs 连续批处理对比
============================================================
教学目的（回答"为什么 Qwen / vLLM 用连续批处理"）：

1. 正确性验证：引擎的批量前向必须与 infer.py 的单请求前向逐位一致，
   增量 decode 必须与"整段重算"一致。这是任何引擎的第一道关。

2. 时间线演示：8 个用户错峰到达，看每个请求排队 -> prefill -> 逐 token
   decode -> 完成的完整生命周期，理解"每迭代前进一步"的调度粒度。

3. 调度对比：同一批请求分别用两种方式服务
     [串行]   一次只服务一个请求，完整生成完才接下一个（静态 batch=1）
     [连续批处理] 每个迭代所有活跃请求各前进一步（vLLM 方式）
   对比虚拟总耗时、吞吐、平均 TTFT——连续批处理把短请求的"尾巴"插进
   长请求的 decode 空档里，整条流水线不再有空闲。

用法：
    python demo_multi_user.py
"""

import sys
from collections import deque

import torch

from engine import (DECODE_MS_PER_STEP, PREFILL_MS_PER_TOKEN, Engine, Request)
from model import (D_MODEL, N_LAYERS, decode_batch, load_weights,
                   prefill_batch)
from minimal_tokenizer import MinimalTokenizer, VOCAB_WORDS


# ================= Part 1：正确性验证 =================

def verify_correctness(tok):
    """引擎的批量前向必须与单请求前向（infer.py 的约定）完全一致。

    对比对象：train_minimal/infer.py 的 forward（单序列版）。这里直接
    在 batch 里放一个"独苗"请求，与单序列手算结果逐位比对。
    """
    E, LM, WQ, WK, WV, WO = load_weights()
    ids = torch.tensor(tok.encode("上海有"))

    # ---- 参考实现：单序列前向（infer.py 的 forward 逻辑）----
    def forward_single(ids):
        S = ids.shape[0]
        X = E[ids] + torch.stack(
            [torch.tensor([0.01 + 0.04 * p + 0.01 * j for j in range(D_MODEL)])
             for p in range(S)])
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool))
        for layer in range(N_LAYERS):
            q = X @ WQ[layer]
            k = X @ WK[layer]
            v = X @ WV[layer]
            qh = q.view(S, 3, 2)
            kh = k.view(S, 3, 2)
            vh = v.view(S, 3, 2)
            raw = torch.einsum("qhd,khd->hqk", qh, kh) / (2 ** 0.5)
            raw = raw.masked_fill(~mask, float("-inf"))
            attn = torch.softmax(raw, dim=-1)
            out = torch.einsum("hqk,khd->qhd", attn, vh).reshape(S, D_MODEL) @ WO[layer]
            X = X + out
        return X @ LM.t()

    # ---- 引擎实现：prefill_batch 单请求 = 单序列版？----
    res = prefill_batch([ids], E, LM, WQ, WK, WV, WO)[0]
    ref = forward_single(ids)
    diff_prefill = (res["logits"] - ref[-1]).abs().max().item()
    print(f"[验证] prefill_batch 与单序列 forward 最大误差: {diff_prefill:.2e}")

    # ---- 增量 decode = 整段重算？----
    # 贪心生成 3 步：用引擎的增量方式走；再整段重算相同序列对比 logits
    state = res
    new_ids = []
    for _ in range(3):
        nxt = int(torch.argmax(state["logits"]).item())
        new_ids.append(nxt)
        state = decode_batch([state], [nxt], E, LM, WQ, WK, WV, WO)[0]
    full_ids = torch.cat([ids, torch.tensor(new_ids)])
    ref_full = forward_single(full_ids)
    diff_decode = (state["logits"] - ref_full[-1]).abs().max().item()
    print(f"[验证] 增量 decode 与整段重算最大误差: {diff_decode:.2e}")
    print(f"[验证] 生成序列: {tok.decode(full_ids.tolist())}")

    ok = diff_prefill < 1e-4 and diff_decode < 1e-4
    # 容差 1e-4：float32 下经过 3 层 einsum + softmax 累计精度约 1e-6 量级，
    # 1e-4 足够判断"数学一致"而非逐位一致
    print(f"[验证] 结论: {'通过 ✓' if ok else '未通过 ✗'}")
    return ok


# ================= Part 2：多用户时间线 =================

def demo_timeline():
    """8 个用户错峰到达的完整时间线（用引擎 verbose 日志观察调度）。"""
    eng = Engine(verbose=True)
    print("\n" + "=" * 70)
    print("多用户时间线：8 个请求错峰到达（连续批处理）")
    print("=" * 70)
    # (prompt, max_new, 到达虚拟毫秒)
    requests = [
        ("上海有", 6, 0),
        ("我爱", 4, 0),
        ("北京", 8, 5),
        ("模型", 5, 15),
        ("推理", 7, 20),
        ("芯片", 4, 30),
        ("股票", 6, 45),
        ("上海有", 3, 60),
    ]
    for text, max_new, delay in requests:
        eng.submit(text, max_new=max_new, arrive_delay_ms=delay)
    eng.run()
    eng.summary()
    return eng


# ================= Part 3：串行 vs 连续批处理 =================

class SerialEngine:
    """串行调度器（静态 batch=1）：一次只服务一个请求，完整生成完
    才从队列里取下一个。对比对象：Engine 的连续批处理。

    实现上完全复用 model.py 的前向，只改调度策略：prefill 后立刻
    连续 decode 到完成，期间不看队列。
    """

    def __init__(self, tok):
        self.tok = tok
        self.E, self.LM, self.WQ, self.WK, self.WV, self.WO = load_weights()
        self.waiting = deque()
        self.finished = []
        self.clock = 0.0
        self._next_id = 0

    def submit(self, text, max_new, arrive_delay_ms=0.0):
        self._next_id += 1
        req = Request(self._next_id, torch.tensor(self.tok.encode(text)),
                      max_new, self.clock + arrive_delay_ms)
        self.waiting.append(req)
        return req.req_id

    def run(self):
        """串行调度：一次一请求，跑完才看下一个。"""
        while self.waiting:
            req = self.waiting.popleft()
            if req.arrive_time > self.clock:
                self.clock = req.arrive_time     # 空闲等待（时间照走）
            req.start_time = self.clock

            # prefill（串行模式一次只 prefill 一个请求）
            self.clock += req.ids.shape[0] * PREFILL_MS_PER_TOKEN
            with torch.no_grad():
                state = prefill_batch([req.ids], self.E, self.LM,
                                      self.WQ, self.WK, self.WV, self.WO)[0]
            req.first_token_time = self.clock

            # 连续 decode 直到完成：期间队列里的其他请求全部干等
            nxt = int(torch.argmax(state["logits"]).item())
            req.decode_times.append(self.clock)
            req.output_ids.append(nxt)
            while len(req.output_ids) < req.max_new:
                self.clock += DECODE_MS_PER_STEP
                with torch.no_grad():
                    state = decode_batch([state], [nxt], self.E, self.LM,
                                         self.WQ, self.WK, self.WV, self.WO)[0]
                nxt = int(torch.argmax(state["logits"]).item())
                req.decode_times.append(self.clock)
                req.output_ids.append(nxt)
            req.finish_time = self.clock
            req.status = "FINISHED"
            self.finished.append(req)
        return self.clock


def compare_schedulers():
    """同一批请求，串行 vs 连续批处理，对比总耗时与吞吐。"""
    tok = MinimalTokenizer(VOCAB_WORDS)
    requests = [
        ("上海有", 6, 0), ("我爱", 4, 0), ("北京", 8, 5), ("模型", 5, 15),
        ("推理", 7, 20), ("芯片", 4, 30), ("股票", 6, 45), ("上海有", 3, 60),
    ]

    print("\n" + "=" * 70)
    print("调度对比：同一批 8 个请求，两种调度方式")
    print("=" * 70)

    # ---- 串行 ----
    serial = SerialEngine(tok)
    for text, max_new, delay in requests:
        serial.submit(text, max_new=max_new, arrive_delay_ms=delay)
    serial_total = serial.run()
    n_tokens = sum(len(r.output_ids) for r in serial.finished)
    serial_ttft = sum(r.ttft for r in serial.finished) / len(serial.finished)

    # ---- 连续批处理（静默模式，不看逐条日志）----
    cb = Engine(tok=tok, verbose=False)
    for text, max_new, delay in requests:
        cb.submit(text, max_new=max_new, arrive_delay_ms=delay)
    cb.run()
    cb_total = cb.clock
    cb_ttft = sum(r.ttft for r in cb.finished) / len(cb.finished)

    # ---- 汇总对比 ----
    print(f"{'指标':<16s} {'串行(静态batch=1)':>20s} {'连续批处理(vLLM方式)':>22s}")
    print(f"{'总耗时(虚拟ms)':<14s} {serial_total:>20.0f} {cb_total:>22.0f}")
    print(f"{'吞吐(tokens/s)':<13s} "
          f"{n_tokens / (serial_total / 1000):>20.0f} "
          f"{n_tokens / (cb_total / 1000):>22.0f}")
    print(f"{'平均TTFT(ms)':<14s} {serial_ttft:>20.1f} {cb_ttft:>22.1f}")
    speedup = serial_total / cb_total
    print(f"\n连续批处理快 {speedup:.2f}x（虚拟时钟）")
    print("原因：串行时每个请求独占引擎，短请求完成后的空档没人补；")
    print("连续批处理每个迭代所有活跃请求各走一步，完成的让位、新请求")
    print("随时插入，流水线不空转。这与《7-连续批处理与PD分离详解》的")
    print("静态 batch 问题（短等长、GPU 空档）完全对应。")


if __name__ == "__main__":
    tok = MinimalTokenizer(VOCAB_WORDS)
    ok = verify_correctness(tok)
    if not ok:
        print("正确性验证未通过，停止演示")
        sys.exit(1)
    demo_timeline()
    compare_schedulers()
