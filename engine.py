"""
engine.py：教学版推理引擎（连续批处理调度器）
============================================
对应文档《7-连续批处理与PD分离详解》：vLLM 的调度核心概念在这里用
972 模型 + 纯 Python 复现，每一个状态切换都有日志可看。

真实引擎的四个关键角色（本文件对应前三个）：
  [调度器 Scheduler]  决定"这次迭代算哪些请求"——本文件的 Engine 类
  [KV 缓存池]         每个请求的 K/V 独立存放、用完释放——Request.state
  [执行器 Worker]     真正做矩阵乘法——model.py 的 prefill_batch / decode_batch
  [服务层]            对外 HTTP 接口——server.py

核心机制：连续批处理（continuous batching）
  静态 batch（老式服务）：一次凑满固定数量请求才开算，算完才放新请求；
  短请求要等长请求，GPU 有空档。
  连续批处理（vLLM 方式）：每个迭代只让每个活跃请求"前进一步"（产出
  1 个 token），新请求随时插入 batch。短请求提前完成就让位，长请求
  继续走——batch 永远满负荷。

PD 分离（prefill / decode 两个环节，逻辑上分开）：
  prefill（P 阶段）：新请求的整个 prompt 一次并行穿过 3 层，填充 KV 缓存，
                     并得到第一个 logits（产生首个 token 候选）
  decode（D 阶段）：  每迭代每个活跃请求只算新 token 一行，K/V 追加缓存
  真实引擎里两者跑在不同算力上（P 算力密集、D 访存密集），这里用
  两个独立函数体现分工，日志里分别标 [P] / [D]。

指标（对应文档《12-推理指标详解》）：
  TTFT  = 请求到达 → 第一个 token 产出的时间（首 token 延迟）
  TPOT  = decode 阶段相邻 token 的平均间隔（每 token 延迟）
  吞吐   = 生成的总 token 数 / 总耗时
  排队时长 = 提交 → 开始 prefill 的时间

虚拟时钟：972 模型在 CPU 上单次前向只有毫秒级，真实耗时看不出调度
差异。引擎用"虚拟时钟"模拟 GPU 的工作量：prefill 每个 token 记
PREFILL_MS 毫秒、decode 每步记 DECODE_MS 毫秒，让 TTFT / TPOT /
吞吐的对比有教学意义（打印时同时给真实耗时作对照）。

用法：
    python engine.py            # 直接跑内置演示（多用户 + 串行对比）
    python server.py            # 起 HTTP 服务（引擎在 server 内部驱动）
"""

import time
from collections import deque

import torch

from model import decode_batch, load_weights, prefill_batch
from minimal_tokenizer import MinimalTokenizer, VOCAB_WORDS

# 虚拟时钟常量（毫秒）：模拟 GPU 上每个计算单元的工作量，可调
PREFILL_MS_PER_TOKEN = 2   # prefill：每处理 1 个输入 token 的虚拟耗时
DECODE_MS_PER_STEP = 5     # decode：每生成 1 个 token 的虚拟耗时


class Request:
    """一个用户请求的状态机。

    状态流转（对应真实引擎的请求生命周期）：
      QUEUED（排队） -> RUNNING（已 prefill，decode 中） -> FINISHED（完成）
    每个请求携带自己的 KV 缓存和指标数据，互不干扰。
    """

    def __init__(self, req_id, ids, max_new, arrive_time):
        self.req_id = req_id
        self.ids = ids                      # prompt 词元 ID（Tensor）
        self.max_new = max_new              # 最多生成多少个新词元
        self.arrive_time = arrive_time      # 到达时刻（虚拟毫秒）

        # 状态与指标
        self.status = "QUEUED"
        self.start_time = None              # 开始 prefill 的时刻（虚拟）
        self.first_token_time = None        # 首个 token 产出的时刻（虚拟）
        self.finish_time = None             # 完成时刻（虚拟）
        self.state = None                   # prefill 后的 {x_last, logits, K/V 缓存}
        self.output_ids = []                # 已生成的新词元
        self.decode_times = []              # 每个 decode token 的产出时刻（虚拟）

    # ---- 指标 ----
    @property
    def ttft(self):
        """首 token 延迟：到达 -> 第一个 token 产出。"""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrive_time

    @property
    def tpot(self):
        """每 token 间隔：decode 阶段相邻 token 的平均虚拟耗时。"""
        if len(self.decode_times) < 2:
            return None
        gaps = [b - a for a, b in zip(self.decode_times, self.decode_times[1:])]
        return sum(gaps) / len(gaps)

    @property
    def queue_time(self):
        """排队时长：到达 -> 开始 prefill。"""
        if self.start_time is None:
            return None
        return self.start_time - self.arrive_time


class Engine:
    """连续批处理调度器：一个 run() 循环驱动所有请求。

    每次迭代做两件事（对应 PD 分离的两个环节）：
      1) [P] 从等待队列捞出所有新请求，批量 prefill
      2) [D] 把所有 RUNNING 请求打包成 batch，每请求前进一步（1 token）
    完成条件满足（EOS / max_new）的请求移出运行集，释放 KV 缓存。
    """

    def __init__(self, tok=None, verbose=True, max_batch=8):
        self.tok = tok or MinimalTokenizer(VOCAB_WORDS)
        self.E, self.LM, self.WQ, self.WK, self.WV, self.WO = load_weights()
        self.verbose = verbose
        self.max_batch = max_batch          # 单批最多服务多少请求（模拟显存上限）

        self.waiting = deque()              # QUEUED 请求队列
        self.running = []                   # RUNNING 请求（decode 中）
        self.finished = []                  # FINISHED 请求（保留指标供汇总）
        self.clock = 0.0                    # 虚拟时钟（毫秒）
        self.wall_start = None              # 真实计时（对照用）
        self.events = []                    # 时间线日志 [(clock, 描述)]
        self._next_id = 0

    # ---------- 对外接口 ----------

    def submit(self, text, max_new=10, tok=None, arrive_delay_ms=0.0):
        """接收一个用户请求（文本），编码后进等待队列。

        arrive_delay_ms：模拟请求错峰到达（demo 用），可指定该请求在
        虚拟时钟第几个毫秒到达；默认 0 = 立即到达。
        """
        tok = tok or self.tok
        ids = tok.encode(text)
        self._next_id += 1
        req = Request(self._next_id, torch.tensor(ids), max_new,
                      self.clock + arrive_delay_ms)
        self.waiting.append(req)
        if self.verbose:
            print(f"[t={req.arrive_time:>7.0f}ms] 收到请求 #{req.req_id} "
                  f"'{text}' -> 生成最多 {max_new} 个词元")
        return req.req_id

    def get(self, req_id):
        """查询请求结果（server 用）：未完成返回 None，完成返回文本。"""
        for r in self.finished:
            if r.req_id == req_id:
                return self.tok.decode([i for i in r.ids.tolist() + r.output_ids])
        return None

    # ---------- 调度循环 ----------

    def _log(self, msg):
        if self.verbose:
            print(f"[t={self.clock:>7.0f}ms] {msg}")
        self.events.append((self.clock, msg))

    def run(self, until_idle=True, max_steps=None):
        """驱动引擎直到（默认）队列空且没有活跃请求。

        until_idle=False 时可配合 max_steps 单步调试。
        """
        if self.wall_start is None:
            self.wall_start = time.time()
        step = 0
        while True:
            step += 1
            self._step_prefill()
            self._step_decode()
            if until_idle and not self.waiting and not self.running:
                break
            if max_steps is not None and step >= max_steps:
                break
            # 服务模式（until_idle=False）：空闲时让出 CPU，等新请求到来
            if not until_idle and not (self.waiting or self.running):
                time.sleep(0.01)
        return step

    def _step_prefill(self):
        """[P 阶段] 新请求批量 prefill。

        注意"批量"的价值：3 个请求同时 prefill，矩阵乘法把 batch 维
        一起算掉（GPU 上基本不增加额外耗时），这就是连续批处理吞吐
        高的第一个来源。
        """
        if not self.waiting:
            return
        # 一次最多 prefill max_batch 个（模拟显存上限；真实引擎受 KV 容量约束）
        batch = []
        while self.waiting and len(batch) < self.max_batch:
            req = self.waiting[0]
            if req.arrive_time > self.clock:
                break                       # 还没到到达时刻，留给未来迭代
            batch.append(self.waiting.popleft())

        if not batch:
            # 队列头还没到到达时刻：引擎空闲，虚拟时钟快进到最早到达时刻
            # （真实引擎里就是 GPU 在等下一个请求，时间照走）
            self.clock = max(self.clock, self.waiting[0].arrive_time)
            return

        # 虚拟耗时 = 所有请求的 prompt token 数之和 × 单 token 耗时
        n_tokens = sum(r.ids.shape[0] for r in batch)
        self.clock += n_tokens * PREFILL_MS_PER_TOKEN
        for req in batch:
            req.status = "RUNNING"
            req.start_time = self.clock
            self._log(f"[P] prefill 请求 #{req.req_id}（{req.ids.shape[0]} 个词元，"
                      f"本批共 {n_tokens} 词元，批量并行）")

        # 真正的批量前向（真实耗时发生在这里，虚拟时钟已在上面走完）
        with torch.no_grad():
            results = prefill_batch([r.ids for r in batch],
                                    self.E, self.LM, self.WQ, self.WK, self.WV, self.WO)
        for req, res in zip(batch, results):
            req.state = res

        # prefill 的产出：每个请求的第一个新 token 候选（这里贪心取 argmax，
        # 真实引擎此刻按采样参数采样）
        first_ids = [int(torch.argmax(r.state["logits"]).item()) for r in batch]
        for req, nxt in zip(batch, first_ids):
            req.first_token_time = self.clock
            req.decode_times.append(self.clock)
            self._log(f"[P] 请求 #{req.req_id} 首个 token 产出 "
                      f"'{self.tok.decode([nxt])}'（TTFT={req.ttft:.0f}ms）")
            # 先入运行集，再写入首 token；若 max_new=0 或遇 EOS，
            # _advance 会立即把它标记完成并从运行集移出
            self.running.append(req)
            self._advance(req, nxt)

    def _step_decode(self):
        """[D 阶段] 所有活跃请求打包成 batch，每请求前进一步。

        这是连续批处理的核心：不按"整个请求"为单位服务，而是按
        "一步 token"为单位。每个迭代 batch 里混合了不同进度的请求，
        完成的让位、新的进来，batch 保持满负荷。
        """
        if not self.running:
            return
        batch = self.running[:self.max_batch]   # 活跃请求（超出上限的留到下一步）

        # 虚拟耗时：decode 一步是访存密集（只算一行），按步计费
        self.clock += DECODE_MS_PER_STEP

        # 批量前进一步：所有请求的当前隐藏状态 + KV 缓存打包进 decode_batch
        new_ids = [int(torch.argmax(r.state["logits"]).item()) for r in batch]
        with torch.no_grad():
            results = decode_batch([r.state for r in batch], new_ids,
                                   self.E, self.LM, self.WQ, self.WK, self.WV, self.WO)
        for req, res, nxt in zip(batch, results, new_ids):
            req.state = res
            req.decode_times.append(self.clock)
            self._advance(req, nxt, log_token=True)

    def _advance(self, req, nxt, log_token=False):
        """把新词元写入请求输出，并判断是否完成。

        完成条件：生成 <EOS>（语料没教过 <EOS>，实际很少触发）或
        达到 max_new 上限。完成即移出运行集，KV 缓存随 Request 一起
        释放（真实引擎里 KV 块归还页式缓存池）。
        """
        eos_id = self.tok.eos_id
        if nxt == eos_id:
            req.finish_time = self.clock
            req.status = "FINISHED"
            self.running.remove(req)
            self.finished.append(req)
            self._log(f"请求 #{req.req_id} 完成（遇 <EOS>），共 {len(req.output_ids)} 个词元")
            return

        req.output_ids.append(nxt)
        if len(req.output_ids) >= req.max_new:
            req.finish_time = self.clock
            req.status = "FINISHED"
            self.running.remove(req)
            self.finished.append(req)
            text = self.tok.decode([i for i in req.ids.tolist() + req.output_ids])
            if log_token or self.verbose:
                self._log(f"请求 #{req.req_id} 完成：{text}")
        elif log_token:
            self._log(f"[D] 请求 #{req.req_id} 生成 '{self.tok.decode([nxt])}'"
                      f"（第 {len(req.output_ids)}/{req.max_new} 个）")

    # ---------- 指标汇总 ----------

    def summary(self):
        """汇总所有完成请求的指标表（TTFT / TPOT / 排队时长 / 词元数）。"""
        print("\n" + "=" * 70)
        print("指标汇总（虚拟时钟，毫秒）")
        print("=" * 70)
        print(f"{'请求':<6s} {'prompt':<10s} {'TTFT':>7s} {'TPOT':>7s} "
              f"{'排队':>7s} {'输出词元':>8s}")
        total_tokens = 0
        for r in self.finished:
            total_tokens += len(r.output_ids)
            prompt = self.tok.decode(r.ids.tolist())
            tpot = f"{r.tpot:.1f}" if r.tpot is not None else "-"
            print(f"#{r.req_id:<5d} {prompt:<10s} {r.ttft:7.0f} {tpot:>7s} "
                  f"{r.queue_time:7.0f} {len(r.output_ids):8d}")
        start = self.wall_start if self.wall_start is not None else time.time()
        wall = (time.time() - start) * 1000
        print(f"\n总生成 {total_tokens} 个词元；虚拟总耗时 {self.clock:.0f}ms；"
              f"真实耗时 {wall:.0f}ms")
        print(f"吞吐（虚拟）≈ {total_tokens / (self.clock / 1000):.0f} tokens/s")
        print("（972 模型在 CPU 上真实耗时仅毫秒级，吞吐对比看虚拟时钟才有意义）")


# ================= 内置演示：多用户错峰请求 =================

def demo_multiuser():
    """内置小演示：5 个用户在不同时刻到达，观察队列、prefill、decode
    的时间线。完整的多用户演示见 demo_multi_user.py（含串行对比）。
    """
    eng = Engine()
    print("=" * 70)
    print("多用户演示：5 个请求错峰到达")
    print("=" * 70)
    # 第 0ms 同时到达 3 个；第 20ms 再来 1 个；第 40ms 再来 1 个
    eng.submit("上海有", max_new=4)
    eng.submit("我爱", max_new=3)
    eng.submit("苹果", max_new=5)
    eng.submit("东方明珠", max_new=4, arrive_delay_ms=20)
    eng.submit("推理", max_new=6, arrive_delay_ms=40)
    eng.run()
    eng.summary()


if __name__ == "__main__":
    demo_multiuser()
