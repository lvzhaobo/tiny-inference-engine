"""
server.py：HTTP 服务层（教学版引擎的对外接口）
============================================
用 Python 标准库 http.server 实现（零第三方依赖），对应真实引擎的
OpenAI 兼容 /v1/chat/completions 接口的极简形态。

接口设计（对照真实服务的最小子集）：
  POST /generate       提交生成请求，返回 request_id（异步模式）
       body: {"prompt": "上海有", "max_new_tokens": 10}
  GET  /result?id=3    查询结果：done=false 表示还没算完
  GET  /metrics        引擎指标汇总（TTFT/TPOT/吞吐）

架构关系（与真实 vLLM 服务对照）：
  用户 HTTP 请求 -> 服务层线程 -> Engine.submit() 进队列 -> 引擎循环
  线程批量 prefill / decode -> 结果落 Engine.finished -> 服务层查询返回
真实服务里这一步由 asyncio + FastAPI 完成，这里用"每请求一个线程 +
轮询"的最小实现，只为展示调度边界：服务层只负责收请求、发结果，
完全不碰模型计算。

用法：
    python server.py
    另开终端：curl -X POST http://127.0.0.1:8000/generate \
              -d '{"prompt": "上海有", "max_new_tokens": 8}'
              curl "http://127.0.0.1:8000/result?id=1"
              curl "http://127.0.0.1:8000/metrics"
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from engine import Engine

HOST, PORT = "127.0.0.1", 8000

# 全局引擎实例：服务层与引擎循环各占一个线程
ENGINE = Engine(verbose=True)
ENGINE_LOCK = threading.Lock()          # submit / 查询的互斥锁（引擎本身单线程循环）


class Handler(BaseHTTPRequestHandler):
    """请求处理：每个 HTTP 请求一个线程，只做"收/发"，不算模型。"""

    def _send_json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """POST /generate：收一个生成请求，返回 request_id（异步）。"""
        if self.path != "/generate":
            return self._send_json(404, {"error": "只有 POST /generate"})
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        prompt = payload.get("prompt", "")
        max_new = int(payload.get("max_new_tokens", 10))

        # 校验：prompt 里不能有词表外字符（972 模型词表 45 词）
        ids = ENGINE.tok.encode(prompt)
        if 0 in ids:
            return self._send_json(400, {"error": "prompt 含词表外字符（<UNK>）"})

        with ENGINE_LOCK:
            req_id = ENGINE.submit(prompt, max_new=max_new)
        self._send_json(200, {"request_id": req_id, "status": "QUEUED"})

    def do_GET(self):
        """GET /result?id=N 查结果；GET /metrics 看指标汇总。"""
        if self.path.startswith("/result"):
            # 解析 ?id=3
            qs = self.path.split("?", 1)[1] if "?" in self.path else ""
            params = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
            req_id = int(params.get("id", 0))
            with ENGINE_LOCK:
                text = ENGINE.get(req_id)
            if text is None:
                self._send_json(200, {"done": False, "request_id": req_id})
            else:
                self._send_json(200, {"done": True, "request_id": req_id, "text": text})
            return

        if self.path == "/metrics":
            # 输出已完成请求的指标（引擎的虚拟时钟统计）
            with ENGINE_LOCK:
                rows = [
                    {"request_id": r.req_id,
                     "prompt": ENGINE.tok.decode(r.ids.tolist()),
                     "ttft_ms": r.ttft, "tpot_ms": r.tpot,
                     "queue_ms": r.queue_time, "out_tokens": len(r.output_ids)}
                    for r in ENGINE.finished
                ]
            self._send_json(200, {"finished": rows, "virtual_clock_ms": ENGINE.clock})
            return

        self._send_json(404, {"error": "路径：/result?id=N 或 /metrics"})


def run_server():
    """启动引擎循环线程 + HTTP 服务。"""
    # 引擎循环独立线程：持续驱动队列（prefill/decode），HTTP 线程只提交
    t = threading.Thread(target=ENGINE.run, kwargs={"until_idle": False},
                         daemon=True)
    t.start()

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"引擎已启动：http://{HOST}:{PORT}")
    print("  POST /generate  {\"prompt\": \"上海有\", \"max_new_tokens\": 8}")
    print("  GET  /result?id=1    GET /metrics")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
