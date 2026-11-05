# tiny-inference-engine

> **用纯 Python 复现 vLLM 的调度核心——连续批处理、PD 分离、KV 缓存、指标。每次状态切换都有日志可看。**

972 个参数的模型 + 一套真实的调度器。不用 GPU、不用 vLLM、不装任何框架，CPU 上 8 秒钟跑完全部演示。

---

## 这是什么

| | |
|---|---|
| **调度器** | 请求队列 · 状态机（QUEUED / RUNNING / FINISHED）· 连续批处理 · 虚拟时钟 |
| **执行器** | 变长 batch 前向 · 增量 decode · 逐请求 KV 缓存 |
| **服务层** | HTTP 接口（提交即返回 / 轮询取结果）+ 指标 |
| **依赖** | 只需要 `torch` |
| **数据量** | 全部 9 个文件，104KB |

## 这不是什么

- **不是推理引擎产品**：无 PagedAttention（每请求连续缓存）、无采样参数（固定 argmax）、词表只有 45 词。
- **不追求吞吐**：972 个参数在 CPU 上真实耗时是毫秒级，看不出任何调度差异——所以引擎用**虚拟时钟**把差异放大。
- **不是 vLLM 的替代品**：它是 vLLM 调度逻辑的**可读版本**，用来回答"为什么 vLLM 不静态 batch"。

---

## 三条可验证的断言

每条都能用一条命令推翻。

| # | 断言 | 命令 | 实测 |
|--:|---|---|---|
| 1 | 批量 prefill 与单序列前向**逐位一致** | `python3 demo_multi_user.py` | 最大误差 **0.00e+00** |
| 2 | 增量 decode 与整段重算**等价** | 同上 | 最大误差 **3.81e-06**（float32 精度内） |
| 3 | 连续批处理比串行快 **2.35x** | 同上 | 总耗时 197→84ms，吞吐 218→512 tok/s，平均 TTFT **74.2→7.8ms** |

---

## 15 秒跑通

```bash
git clone <this-repo> && cd tiny-inference-engine
python3 -m pip install torch

python3 demo_multi_user.py     # 正确性验证 + 8 请求时间线 + 调度对比
python3 engine.py              # 内置小演示：5 请求错峰到达
```

服务模式（另开终端）：

```bash
python3 server.py              # 127.0.0.1:8000

curl -X POST http://127.0.0.1:8000/generate \
     -d '{"prompt": "上海有", "max_new_tokens": 8}'
# {"request_id": 1, "status": "QUEUED"}

curl "http://127.0.0.1:8000/result?id=1"
# {"done": true, "request_id": 1, "text": "上海有银行银行。..."}

curl "http://127.0.0.1:8000/metrics"
# 已完成请求的 TTFT / TPOT / 排队时长
```

---

## 文件与角色

对应真实推理引擎的四个角色：

| 文件 | 角色 | 职责 |
|---|---|---|
| `model.py` | 执行器 | 模型加载 + 批量前向（变长 mask、KV 增量 decode） |
| `engine.py` | 调度器 | 请求队列、状态机、连续批处理、虚拟时钟、指标 |
| `server.py` | 服务层 | 收请求、发结果，不碰模型计算 |
| `demo_multi_user.py` | 验证 | 正确性断言 + 多用户时间线 + 串行 vs 连续批处理对比 |
| `minimal_tokenizer.py` + `vocab.json` | 词表 | 45 词最长匹配（与模型同源） |
| `trained_weights.json` | 权重 | 972 参数，14 个张量（快照，见下） |

```
HTTP 请求 ──▶ server.py（服务层，每请求一线程）──▶ engine.py（调度器）
                                                     │ waiting 队列（QUEUED）
                                                     │ running 列表（RUNNING）
                                                     │ finished 列表（FINISHED）
                                                     ▼
                                             model.py（执行器）
                                             prefill_batch / decode_batch
```

## 四个核心机制

**1. 连续批处理。**每迭代所有活跃请求各走一步（产出 1 token），完成的让位、新请求随时插入。实测同一批 8 请求比串行快 2.35x——这就是"为什么 vLLM 不用静态 batch"。

**2. PD 分离。**prefill（整段 prompt 并行，算力密集）与 decode（每步一行，访存密集）是两个独立函数，日志分别标 `[P]` / `[D]`。

**3. KV cache 的正确语义。**新词元的 k/v 必须先写入缓存、q 再查询含自己在内的全部缓存。丢掉"自己"等于丢掉自注意力，输出完全错乱。`demo_multi_user.py` 的 Part 1 把增量 decode 与整段重算逐位比对。

**4. 虚拟时钟。**972 在 CPU 上真实耗时毫秒级，看不出调度差异。引擎用虚拟毫秒模拟 GPU 工作量（prefill 每 token 2ms、decode 每步 5ms，`engine.py` 顶部常量可调），让 TTFT/TPOT/吞吐对比有意义。

## 与真实 vLLM 的对应

| 本仓库 | vLLM |
|---|---|
| `engine.py` 的 waiting/running/finished | `Scheduler` 的 waiting/running 队列 |
| 每迭代全活跃请求各走一步 | continuous batching |
| `prefill_batch` / `decode_batch` | prefill / decode 两阶段 |
| 每请求连续 KV 缓存，完成整体释放 | PagedAttention（分块 + 池 + 碎片统计，**未实现**） |
| 虚拟时钟 | GPU 真实算子耗时 |
| TTFT / TPOT / 排队时长 | 同名指标 |

## 权重快照说明

本仓库自带一份 `trained_weights.json`（28KB），是 [tiny-llm-stack](https://github.com/lvzhaobo) 里 972 模型训练产出的**冻结快照**：

```
md5 = a40b16a2e4f853dff04be37547af697e
```

上游权重若变化，用 `tools/sync-weights.sh` 同步（脚本会打印新 md5）。

> 模型只有 972 个参数、训练可复现到字节级，所以快照不会漂移；这是它能独立成仓的前提。

## 能力边界

| 边界 | 说明 |
|---|---|
| 无 PagedAttention | 每请求连续缓存，完成即整体释放。页式（固定块+池+碎片统计）是下一步 |
| 无采样参数 | 固定贪心 argmax；温度/top-k/top-p 属服务层扩展，与调度无关 |
| 词表 45 词 | prompt 含词表外字符会编码成 `<UNK>`（server 会拒绝） |
| 单机单进程 | 无分布式、无多卡、无真实并发压测 |

## 下一步

| 想继续看 | 去哪 |
|---|---|
| 这个模型是怎么训出来的（词表 / Attention / 训练 / 发布） | `tiny-llm-stack` |
| 训练与推理的 16 篇专题 | `tiny-llm-stack/docs/` |
| 原理能不能看得见 | `llm-visual-lab` |
| 网关 / 监控 / 灰度，上线之后怎么活 | `llm-ops-workshop` |

## License

待定（发布前必须补）。
