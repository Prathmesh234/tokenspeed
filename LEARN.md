# TokenSpeed: From-Scratch Walkthrough + Print-Trace Plan

**Audience:** you, on a single RTX 6000 Ada (SM 8.9, 48 GB), running a small
dense model (Qwen3-1.7B / Llama-3.2-1B) with `--attention-backend triton`.
**Goal:** understand the whole engine end-to-end, then verify each stage with
targeted prints so you can watch a single token travel from HTTP to GPU and
back.

All file paths below are relative to the repo root
`/Users/prathmeshbhatt/Desktop/tokenspeed`.

> **Print convention.** Every print statement in this doc uses the prefix
> `[TS]` followed by a tag like `[TS][http]`, `[TS][async]`, `[TS][sched]`,
> `[TS][exec]`, `[TS][model]`, `[TS][attn]`, `[TS][triton]`, `[TS][sampler]`,
> `[TS][output]`. After running, you can `grep -E "^\[TS\]"` your log and
> reconstruct a timeline.

---

## 0. Mental model in one paragraph

TokenSpeed is a **3-process** OpenAI-compatible LLM server:

1. **Frontend** (main Python process). FastAPI/uvloop receives HTTP, an
   `AsyncLLM` object tokenizes and ZMQ-PUSHes the request to the scheduler.
2. **Scheduler** (separate process). A Python `EventLoop` ticks forever; on
   each tick it pulls new requests, asks a **C++ `Scheduler`** (pybind11) for
   the next `ExecutionPlan`, dispatches it to the worker(s), and feeds the
   results back into the C++ FSM.
3. **Worker(s)** (one per GPU/TP rank). A `ModelExecutor` materializes the
   `ExecutionPlan` into batched tensors, runs the model `forward()`
   (which calls Triton attention + GEMMs), samples a token, and returns it.

There are **two kinds of plans** per tick: prefill (a chunked context for a
new request) and decode (one token per active request). The C++ FSM owns
which requests are in `WAITING` / `PREFETCH` / `PREFILLING` / `DECODING` /
`RETRACTED` / `FINISHED` and decides what's safe to schedule.

The "speed-of-light" claim rests on (a) fused/specialized kernels — gated to
Hopper/Blackwell — and (b) the tight C++ FSM that lets KV pages be reused
without locks. On Ada you only get the Triton path, but the **scheduler/FSM
and engine architecture are identical** — that's exactly what's worth
studying.

---

## 1. Repository layout (the four packages)

```
tokenspeed/
├── python/                        # Package: tokenspeed (the runtime)
│   └── tokenspeed/
│       ├── cli.py                 # tokenspeed serve / bench / env / version
│       ├── runtime/
│       │   ├── entrypoints/       # FastAPI server + OpenAI compat layer
│       │   ├── engine/            # AsyncLLM + scheduler-process EventLoop
│       │   ├── execution/         # ModelExecutor / ModelRunner / CUDA-graph
│       │   ├── layers/            # paged_attention, attention/backends, MoE
│       │   ├── models/            # llama.py, qwen3.py, deepseek_v3/v4, ...
│       │   ├── cache/             # Python KV pool / req-to-token-pool
│       │   ├── sampling/          # Greedy / FlashInfer sampling backends
│       │   └── distributed/       # TP/EP/PP comm + process-group manager
│
├── tokenspeed-scheduler/          # Package: tokenspeed_scheduler (C++ + pybind)
│   ├── csrc/
│   │   ├── core/                  # token_container.{h,cpp}
│   │   ├── fsm/                   # forward_/cache_/pd_ states & events
│   │   ├── resource/              # page_allocator, radix_tree, KV prefix cache
│   │   └── scheduler/             # scheduler.{h,cpp}, request, execution_plan
│   ├── bindings/                  # python_module.cpp (pybind11)
│   └── python/tokenspeed_scheduler/
│
├── tokenspeed-kernel/             # Package: tokenspeed_kernel (Triton/CUDA ops)
│   └── python/tokenspeed_kernel/
│       ├── registry.py            # Priority bands, register_kernel()
│       ├── platform.py            # ArchVersion, CapabilityRequirement
│       ├── selection.py           # selects best kernel for current platform
│       └── ops/                   # gemm/, moe/, attention/, layernorm/, ...
│           └── attention/triton/  # ★ the kernels you'll actually run
│
└── tokenspeed-mla/                # Package: tokenspeed_mla (CuTe DSL MLA, SM10.0)
    └── python/tokenspeed_mla/     # not used on Ada
```

---

## 2. The three processes — what owns what

```
┌───────────────────────────────── MAIN PROCESS (Frontend) ─────────────────────────────────┐
│                                                                                            │
│  uvicorn HTTP server (uvloop+FastAPI)                                                      │
│   └── /v1/chat/completions                                                                 │
│         └── OpenAIServingChat.handle_request(...)                                          │
│               └── AsyncLLM.generate_request(...)                                           │
│                     │  tokenize, build GenerateReqInput                                    │
│                     ▼                                                                      │
│             EngineCoreClient.send_to_scheduler  (ZMQ PUSH)                                 │
│                                                                                            │
│  background asyncio task: _result_dispatcher                                               │
│             ◀── recv_from_detokenizer  (ZMQ PULL)  BatchTokenIDOut / BatchStrOut           │
│             OutputProcessor.handle_batch_output → per-request asyncio.Queue                │
│             ──► HTTP SSE chunks back to client                                             │
└────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                             │ ZMQ
                                             ▼
┌─────────────────────────────── SCHEDULER PROCESS ──────────────────────────────────────────┐
│                                                                                            │
│  EventLoop.event_loop()  (run_event_loop in event_loop.py)                                 │
│  ┌──────────────────────────────────────────────────────────────────────────────────────┐  │
│  │  while True:                                                                         │  │
│  │      _process_new_requests()             # ZMQ recv → C++ scheduler.submit_requests │  │
│  │      _commit_cache_results()             # turn cache events into FSM transitions   │  │
│  │      plan = scheduler.next_execution_plan()  # ★ C++ Scheduler::NextExecutionPlan() │  │
│  │      _submit_cache_ops(plan)             # any LoadBack/WriteBack KV ops            │  │
│  │      forward_op = _get_forward_op(plan)                                              │  │
│  │      results, _ = _dispatch_forward(forward_op, ...)   # blocks on worker           │  │
│  │      changes = _commit_forward_results(forward_op, results, ...)                    │  │
│  │      advance_forward(scheduler, changes)  # ★ C++ Scheduler::Advance(events)        │  │
│  └──────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                            │
│  Owns: C++ Scheduler instance (FSM + KV page allocator + radix prefix cache),             │
│        OutputProcesser, IncrementalDetokenizer (rank-0 only).                              │
└────────────────────────────────────────────┬───────────────────────────────────────────────┘
                                             │ Python in-process call (TP rank 0) +
                                             │ torch.distributed broadcast (other ranks)
                                             ▼
┌─────────────────────────────── WORKER PROCESS(ES) ─────────────────────────────────────────┐
│                                                                                            │
│  ModelExecutor                                                                             │
│   ├── InputBuffers.fill_input_buffers(forward_op, ...)   # build positions/cache_loc/...  │
│   ├── CudaGraphWrapper.replay(...) OR eager forward                                        │
│   ├── ModelRunner.forward(ctx, input_ids, positions, ...)                                  │
│   │     └── Llama/Qwen3 model.forward                                                      │
│   │           └── for each decoder layer:                                                  │
│   │                 ├── self_attn = QKV → RoPE → PagedAttention                            │
│   │                 │     └── ctx.attn_backend.forward(...)  # TritonAttnBackend           │
│   │                 │           ├── token_to_kv_pool.set_kv_buffer(...)                    │
│   │                 │           └── prefill_attention_fwd(...) | decode_attention_fwd(...)│
│   │                 │                  └── @triton.jit kernel                             │
│   │                 └── MLP = gate_up_proj → SiLU → down_proj                              │
│   ├── LogitsProcessor.forward(...)   # gather last-token hidden → logits                   │
│   └── SamplingBackend.sample(...) | .verify(...)                                           │
│   └── return tokens + (optionally) hidden states / logprobs                                │
│                                                                                            │
│  Owns: torch.nn.Module weights, KV page pool tensor, attention backend metadata,          │
│        captured CUDA graphs.                                                               │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

This 3-tier separation is the single most important thing to internalize.
**Frontend** never blocks on GPU. **Scheduler** never holds GPU memory.
**Worker** never tokenizes or talks to HTTP.

---

## 3. The C++ Scheduler is the brain

Even though we won't print into C++ this round, you must know its surface
because everything in Python ultimately calls it. From
`tokenspeed-scheduler/csrc/scheduler/scheduler.h:47`:

```cpp
class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config);
    void SubmitRequests(const std::vector<RequestSpec>& request_specs);
    ExecutionPlan NextExecutionPlan();              // ★ called every tick
    void Advance(const ExecutionEvent& event);      // ★ feeds back results

    std::size_t WaitingSize() const;
    std::size_t DecodingSize() const;
    std::size_t AvailableKvPages() const;
    // ...
private:
    PageAllocator device_allocator_;       // KV pages on device
    PageAllocator host_allocator_;         // KV pages on host (for L2 cache)
    KVPrefixCache kv_prefix_cache_;        // radix tree of cached prefixes
    ReqPoolAllocator req_pool_allocator_;
    std::unordered_map<std::string, std::unique_ptr<Request>> requests_;
};
```

**Mental model of one tick:**

1. `SubmitRequests` adds the new requests to `requests_` map (state = WAITING).
2. `NextExecutionPlan` walks `requests_`, asks the FSM what each request can
   do *now* given KV-page availability and the chunked-prefill budget, and
   emits a list of `PrefillOperation` + `DecodeOperation` + `PrefetchOperation`
   + (sometimes) `WriteBackOperation`s. Crucially **this is pure planning** —
   no GPU work happens in C++.
3. `Advance(ExecutionEvent)` is called by Python *after* the worker reports
   results (`forward::ExtendResult`, `forward::Finish`, etc.). This drives
   FSM transitions: e.g. PREFILLING → DECODING when last chunk done, or
   DECODING → FINISHED on EOS.

The bindings live in `tokenspeed-scheduler/bindings/python_module.cpp`.

---

## 4. End-to-end trace of one HTTP request

This is the canonical journey we'll instrument. Each phase has:

- **What it does** — one paragraph
- **File & function** — exact location to read first
- **Where to print** — the line you'll insert a `print(...)` on
- **What you'll learn** — what the print confirms

We'll instrument **all 10 phases**, producing one line per phase per token.
At the end you have a complete cycle log per generated token.

---

### PHASE 1 — HTTP arrives

The user sends a `POST /v1/chat/completions` to FastAPI.

- **File:** `python/tokenspeed/runtime/entrypoints/http_server.py` (route
  `openai_v1_chat_completions`, ~line 370–380).
- **What:** receives JSON, hands off to `OpenAIServingChat.handle_request`.

**Print 1.1** — at the top of the route handler:

```python
print(f"[TS][http] POST /v1/chat/completions  model={request.model}  "
      f"messages={len(request.messages)}  stream={request.stream}",
      flush=True)
```

**Print 1.2** — in
`python/tokenspeed/runtime/entrypoints/openai/serving_chat.py:266` inside
`_convert_to_internal_request`, after the `GenerateReqInput` is built:

```python
print(f"[TS][http] _convert_to_internal_request: rid={adapted.rid} "
      f"input_ids_len={len(adapted.input_ids) if adapted.input_ids else None} "
      f"sampling={sampling_params}", flush=True)
```

> **What you'll see:** the OpenAI request landing, getting a `rid` (request
> id used everywhere downstream), and being shaped into an internal
> `GenerateReqInput`.

---

### PHASE 2 — AsyncLLM intake

The frontend's `AsyncLLM.generate_request` is the single chokepoint where
*every* request enters the engine — sync (`LLM`) or async, OpenAI or raw.

- **File:** `python/tokenspeed/runtime/engine/async_llm.py:269–315`.
- **What:** validates, runs `_tokenize_one_request`, then `_send_one_request`
  which ZMQ-PUSHes a tokenized req to the scheduler.

**Print 2.1** — top of `AsyncLLM.generate_request`:

```python
print(f"[TS][async] generate_request rid={obj.rid} prompt_len="
      f"{len(obj.input_ids) if obj.input_ids else len(obj.text or '')}",
      flush=True)
```

**Print 2.2** — inside `_tokenize_one_request` (line ~303):

```python
print(f"[TS][async] tokenized rid={obj.rid} -> "
      f"{len(tokenized.input_ids)} tokens, "
      f"first5={tokenized.input_ids[:5]}", flush=True)
```

**Print 2.3** — inside `_send_one_request` just before
`send_to_scheduler.send_pyobj(...)`:

```python
print(f"[TS][async] PUSH rid={tokenized.rid} -> scheduler ZMQ", flush=True)
```

> **What you'll see:** prompt → tokens, then the handoff over ZMQ. After
> Phase 2 the frontend Python returns to the asyncio loop and waits for
> output.

---

### PHASE 3 — Scheduler-process intake

The scheduler process pulls ZMQ messages, broadcasts to all TP ranks (only
rank 0 talks to ZMQ), then hands to the C++ scheduler.

- **File:** `python/tokenspeed/runtime/engine/event_loop.py:606–707`
  (`_init_interprocess_comm`, `_process_new_requests`).
- **File:** `python/tokenspeed/runtime/engine/request_handler.py:133–244`
  (`recv_reqs`, `process_requests`, `handle_generate_request`).

**Print 3.1** — top of `EventLoop._process_new_requests`:

```python
print(f"[TS][sched] poll new reqs (rank={self.tp_rank})", flush=True)
```

**Print 3.2** — in `RequestHandler.recv_reqs` after a request is received
(line ~140):

```python
print(f"[TS][sched] recv rid={req.rid} state=WAITING "
      f"prompt_tokens={len(req.input_ids)}", flush=True)
```

**Print 3.3** — in `RequestHandler.handle_generate_request` after the
`RequestSpec` is built:

```python
print(f"[TS][sched] handoff to C++ rid={spec.id} "
      f"max_new_tokens={spec.sampling_params.max_new_tokens}", flush=True)
```

**Print 3.4** — in `_process_new_requests` after
`scheduler.submit_requests(...)`:

```python
print(f"[TS][sched] C++ Scheduler.SubmitRequests batch={len(specs)} "
      f"waiting_size_now={self.scheduler.waiting_size()}", flush=True)
```

> **What you'll see:** the request crossing into C++. After this, the C++
> FSM owns the request lifecycle.

---

### PHASE 4 — The tick: NextExecutionPlan

The scheduler's heart is the `event_loop` while-true at
`event_loop.py:839–887`. Every tick:

1. ingest new requests,
2. drain completed cache ops,
3. **ask C++ for a plan**,
4. submit cache ops,
5. **dispatch a forward** (the GPU work),
6. commit results,
7. **advance the FSM**.

**Print 4.1** — at the top of `event_loop` (line 841):

```python
print(f"[TS][sched] tick begin "
      f"waiting={self.scheduler.waiting_size()} "
      f"decoding={self.scheduler.decoding_size()} "
      f"avail_pages={self.scheduler.available_kv_pages()}", flush=True)
```

**Print 4.2** — right after `execution_plan = self.scheduler.next_execution_plan()`
(line 844):

```python
fwd = execution_plan.forward
cache = execution_plan.cache
print(f"[TS][sched] plan forward_ops={len(fwd)} cache_ops={len(cache)} "
      f"req_ids={[op.id for op in (fwd[0].operations if fwd else [])]}",
      flush=True)
```

**Print 4.3** — in `_get_forward_op` once `forward_op` is decided (a
prefill chunk vs a decode batch):

```python
if forward_op is not None:
    print(f"[TS][sched] forward_op type={type(forward_op).__name__} "
          f"batch_size={len(forward_op.request_ids)} "
          f"num_tokens={getattr(forward_op, 'num_tokens', '?')}", flush=True)
```

**Print 4.4** — at the very end of the tick, after `advance_forward`:

```python
print(f"[TS][sched] tick end "
      f"changes={len(request_changes)}", flush=True)
```

> **What you'll see:** the tick rate, plan composition, and how many request
> changes the FSM applied. If you see prefill chunks and decode batches
> alternating you've understood **chunked prefill**: long prompts are split
> across ticks so decode latency on other requests doesn't suffer.

---

### PHASE 5 — Dispatching the forward (scheduler → worker)

In the same process (rank 0) the scheduler calls `ModelExecutor`. On other
TP ranks the dispatch happens via `torch.distributed` broadcast.

- **File:** `python/tokenspeed/runtime/engine/event_loop.py:428–527`
  (`_dispatch_forward`).
- **File:** `python/tokenspeed/runtime/execution/model_executor.py`
  (`execute_forward_op_with_log` → `_forward_step` at line 361–414).

**Print 5.1** — top of `_dispatch_forward`:

```python
print(f"[TS][exec] dispatch forward "
      f"mode={forward_op.forward_mode.name if hasattr(forward_op,'forward_mode') else '?'} "
      f"bs={len(forward_op.request_ids)}", flush=True)
```

**Print 5.2** — at the top of `ModelExecutor._forward_step` (line ~361):

```python
print(f"[TS][exec] _forward_step batch_size={batch.batch_size} "
      f"num_tokens={batch.input_ids.numel()} "
      f"forward_mode={batch.forward_mode.name}", flush=True)
```

> **What you'll see:** the boundary where the scheduler hands the GPU work
> to the executor. Compare timestamps with PHASE 4 — anything between is
> Python overhead in the scheduler process.

---

### PHASE 6 — Building the forward batch

Before the model runs, raw scheduler operations are turned into actual GPU
tensors: `input_ids`, `positions`, `out_cache_loc` (where in the KV pool to
write), `req_pool_indices`, `seq_lens`.

- **File:** `python/tokenspeed/runtime/execution/input_buffer.py:101–260`
  (`InputBuffers.fill_input_buffers`).
- **File:** `python/tokenspeed/runtime/execution/forward_batch_info.py:32–69`
  (`ForwardMode`).

**Print 6.1** — at the bottom of `fill_input_buffers` once tensors exist:

```python
print(f"[TS][exec] fill_input_buffers "
      f"input_ids.shape={tuple(input_ids.shape)} "
      f"positions.range=({int(positions.min())},{int(positions.max())}) "
      f"out_cache_loc.shape={tuple(out_cache_loc.shape)} "
      f"seq_lens={seq_lens.tolist()[:8]}{'...' if len(seq_lens)>8 else ''}",
      flush=True)
```

> **What you'll see:** the exact tensors the model is about to consume —
> particularly `out_cache_loc`, which tells you which KV-pool slots will
> receive the new K/V vectors this step. This is *the* critical piece of
> paged attention.

---

### PHASE 7 — The model forward (Llama / Qwen3)

`ModelRunner.forward` (`execution/model_runner.py:115–146`) just routes the
call into the `nn.Module`. From there each decoder layer does:

```
x → RMSNorm → QKV projection → RoPE → PagedAttention → output proj →
   residual → RMSNorm → MLP(SiLU) → residual
```

- **File:** `python/tokenspeed/runtime/models/llama.py:191–208`
  (`LlamaAttention.forward`) and `:101–107` (`LlamaMLP.forward`).
- **File:** `python/tokenspeed/runtime/models/base/decoder_layer.py:123–226`
  (the base loop calling `forward_attn` then `forward_mlp`).

**Print 7.1** — at the top of `ModelRunner.forward`
(`execution/model_runner.py:115`):

```python
print(f"[TS][model] ModelRunner.forward "
      f"input_ids.shape={tuple(input_ids.shape)} "
      f"is_capture={get_is_capture_mode() if 'get_is_capture_mode' in globals() else 'n/a'}",
      flush=True)
```

**Print 7.2** — inside `LlamaAttention.forward` (the per-layer self-attn),
just *before* `self.attn(q, k, v, ...)` is called:

```python
print(f"[TS][model] layer={self.layer_id} pre-attn "
      f"q.shape={tuple(q.shape)} k.shape={tuple(k.shape)} "
      f"v.shape={tuple(v.shape)}", flush=True)
```

> **Why guard with `layer_id == 0`?** A 28-layer model with one decode
> token will print 28 times *per token*. To reduce noise, prefer:
> `if self.layer_id in (0, self.num_layers - 1): print(...)`. Add the prefix
> `[TS][model][L{self.layer_id}]` so you can grep one layer.

**Print 7.3** — inside `LlamaMLP.forward`, after the down projection:

```python
if self.layer_id == 0:
    print(f"[TS][model][L0] mlp out.shape={tuple(out.shape)} "
          f"out.norm={out.norm():.4f}", flush=True)
```

> **What you'll see:** the model is just a Python orchestration of layer
> calls. Most "speed" comes one level deeper, in the attention backend and
> Triton kernels. This phase confirms layer order and the QKV/MLP shapes.

---

### PHASE 8 — Attention backend dispatch (Python wrapper)

`PagedAttention.forward` (`layers/paged_attention.py:59–89`) just delegates
to `ctx.attn_backend.forward(...)`. The base
`AttentionBackend.forward` (`layers/attention/backends/base.py:113–163`)
splits decode vs extend and calls one of `forward_decode` or `forward_extend`.

For us that's `TritonAttnBackend.forward_decode` / `forward_extend` in
`layers/attention/backends/triton.py:719–819`.

**Print 8.1** — top of `PagedAttention.forward`
(`layers/paged_attention.py:59`):

```python
print(f"[TS][attn] PagedAttention layer={self.layer_id} "
      f"backend={type(ctx.attn_backend).__name__} "
      f"mode={ctx.forward_mode.name}", flush=True)
```

**Print 8.2** — inside `AttentionBackend.forward` (base.py:113), right at
the branch point:

```python
print(f"[TS][attn] {type(self).__name__}.forward branch="
      f"{'decode' if forward_mode.is_decode() else 'extend'}", flush=True)
```

**Print 8.3** — at the top of `TritonAttnBackend.forward_decode`
(`backends/triton.py:772`):

```python
print(f"[TS][attn][triton] forward_decode q.shape={tuple(q.shape)} "
      f"qk_dim={layer.qk_head_dim} v_dim={layer.v_head_dim} "
      f"save_kv={save_kv_cache} "
      f"sliding_window={layer.sliding_window_size}", flush=True)
```

**Print 8.4** — at the top of `TritonAttnBackend.forward_extend`
(`backends/triton.py:719`):

```python
print(f"[TS][attn][triton] forward_extend q.shape={tuple(q.shape)} "
      f"max_extend_len={self.forward_metadata.max_extend_len} "
      f"qo_indptr.numel={self.forward_metadata.qo_indptr.numel() if self.forward_metadata.qo_indptr is not None else 0}",
      flush=True)
```

**Print 8.5** — right after `set_kv_buffer` in either branch (writing K/V
into the paged pool):

```python
print(f"[TS][attn][triton] set_kv_buffer layer={layer.layer_id} "
      f"out_cache_loc.shape={tuple(out_cache_loc.shape)} "
      f"k.shape={tuple(k.shape)}", flush=True)
```

> **What you'll see:** the moment the freshly-computed K/V is appended into
> the global KV pool tensor at `out_cache_loc`. After this point those K/V
> rows are queryable by future tokens.

---

### PHASE 9 — The Triton kernel itself

This is what you actually want to learn, since you're studying Triton.
Two kernels matter:

#### 9a. Decode (one query token per request, attending over many KV)

- **File:** `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/mha_decode.py`
- Wrapper: `_decode_att_m_fwd` at line 171–245
- Kernel: `@triton.jit _fwd_kernel_stage1` at line 33–170
- Grouped variant: `_fwd_grouped_kernel_stage1` at line 247+

The decode is **two-stage** (split-KV reduction): stage 1 computes per-split
partial logits + LSE, stage 2 reduces them.

**Print 9.1** — at the top of `_decode_att_m_fwd` (the host-side launcher):

```python
print(f"[TS][triton][decode] launch grid=(B={cache_seqlens.shape[0]}, "
      f"H={q.shape[1]}, MAX_KV_SPLITS={max_kv_splits}) "
      f"BLOCK_DMODEL={triton.next_power_of_2(k_buffer.shape[-1])} "
      f"page_size={page_size}", flush=True)
```

> **What you'll see:** the three-dim grid `(batch, heads, kv_splits)`. This
> grid is the central concept — each program (i.e. each thread block / CTA)
> handles one (request, head, kv-shard).

> **Why "stage 1"?** With long sequences the K/V is split across
> `num_kv_splits` programs along the sequence dimension. Each computes a
> partial softmax+matmul; stage 2 then reduces the splits per (batch, head)
> into the final output. This is the trick that keeps decode latency low for
> very long contexts.

#### 9b. Prefill (many query tokens, ragged across requests, packed KV)

- **File:** `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/mha_prefill.py`
- Kernel: `@triton.jit _fwd_kernel` at line ~32–220
- Wrapper: `prefill_attention_fwd` lower in the file

**Print 9.2** — at the top of `prefill_attention_fwd`:

```python
print(f"[TS][triton][prefill] launch q.shape={tuple(q.shape)} "
      f"k.shape={tuple(k.shape)} v.shape={tuple(v.shape)} "
      f"max_extend_len={max_extend_len} "
      f"causal={is_causal} sliding_window={sliding_window_size}",
      flush=True)
```

#### 9c. Inside the GPU kernel (be careful)

You **cannot** put a `print(...)` inside a `@triton.jit` function in the
normal way — Python prints don't run on the GPU. Triton offers
`tl.device_print(name, value)` which is the only sanctioned in-kernel
print, but it floods stderr fast. Use it sparingly:

```python
@triton.jit
def _fwd_kernel_stage1(...):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    if cur_batch == 0 and cur_head == 0 and split_kv_id == 0:
        tl.device_print("[TS][triton][gpu] cache_len=", cache_len)
        tl.device_print("[TS][triton][gpu] kv_start_offset=", kv_start_offset)
    ...
```

The `if` guard is critical — without it you'll print thousands of times.

> **Reading order for the kernel itself.** Open `mha_decode.py` and read
> top-to-bottom for `_fwd_kernel_stage1` (lines 33–170): note
> `cur_batch = program_id(0)`, `cur_head = program_id(1)`,
> `split_kv_id = program_id(2)`, the strided loads from `K_Buffer` /
> `V_Buffer` via the `page_table`, the online softmax (running `e_max`,
> `e_sum`), and finally the `tl.store(Att_Lse, e_max + tl.log(e_sum))` at
> line 165–168. That last line is where the LSE for stage-2 reduction is
> persisted.

---

### PHASE 10 — Logits, sampling, and the output return path

After the last decoder layer you have a `hidden_states` tensor. Steps:

1. **LogitsProcessor.forward** prunes to the last token of each sequence,
   does the LM-head matmul, and returns logits.
   - File: `python/tokenspeed/runtime/layers/logits_processor.py:157+`
2. **SamplingBackend.sample** turns logits into next-token ids.
   - File: `python/tokenspeed/runtime/sampling/backends/greedy.py:153–179`
3. Tokens go back to the scheduler-process side as a `forward.ExtendResult`
   or `forward.Finish` event.
4. `_commit_forward_results` (`event_loop.py:708–731`) calls the output
   processor; `OutputProcesser.handle_batch_output` detokenizes and emits
   `BatchTokenIDOut` or `BatchStrOut` over ZMQ to the frontend.
5. `AsyncLLM._result_dispatcher` puts the chunk on the per-rid asyncio queue;
   the HTTP route reads it and emits an SSE chunk.

**Print 10.1** — top of `LogitsProcessor.forward`:

```python
print(f"[TS][logits] forward hidden.shape={tuple(hidden_states.shape)} "
      f"prune_indices.numel={prune_indices.numel() if prune_indices is not None else 'all'}",
      flush=True)
```

**Print 10.2** — top of `GreedySamplingBackend.sample`
(`sampling/backends/greedy.py:153`):

```python
print(f"[TS][sampler] greedy.sample logits.shape={tuple(logits.shape)} "
      f"argmax_first5={logits.argmax(-1)[:5].tolist()}", flush=True)
```

**Print 10.3** — in `event_loop._commit_forward_results` after tokens are
gathered for output:

```python
print(f"[TS][output] commit batch tokens={[t for t in new_tokens[:8]]}{'...' if len(new_tokens)>8 else ''} "
      f"finished_rids={finished_rids}", flush=True)
```

**Print 10.4** — in `OutputProcesser.handle_batch_output` (top of method):

```python
print(f"[TS][output] handle_batch_output kind={type(batch).__name__} "
      f"size={len(batch.rids) if hasattr(batch,'rids') else '?'}", flush=True)
```

**Print 10.5** — in `AsyncLLM._result_dispatcher` (bottom of file, around
line 227–245), right after a chunk lands on the per-rid queue:

```python
print(f"[TS][output] dispatch rid={rid} -> queue (kind={type(payload).__name__})",
      flush=True)
```

**Print 10.6** — back in the HTTP streaming generator
(`http_server.py` `/v1/chat/completions` SSE loop):

```python
print(f"[TS][http] SSE chunk rid={rid} "
      f"delta_text={chunk.choices[0].delta.content!r}", flush=True)
```

> **What you'll see end-to-end for one token:**
>
> ```
> [TS][http] POST /v1/chat/completions ...
> [TS][async] generate_request rid=...
> [TS][async] tokenized rid=...
> [TS][async] PUSH rid=... -> scheduler ZMQ
> [TS][sched] poll new reqs
> [TS][sched] recv rid=... state=WAITING
> [TS][sched] handoff to C++ rid=...
> [TS][sched] C++ Scheduler.SubmitRequests batch=1
> [TS][sched] tick begin waiting=1 decoding=0 ...
> [TS][sched] plan forward_ops=1 cache_ops=0
> [TS][sched] forward_op type=PrefillForwardOp bs=1
> [TS][exec] dispatch forward mode=EXTEND
> [TS][exec] _forward_step ...
> [TS][exec] fill_input_buffers ...
> [TS][model] ModelRunner.forward ...
> [TS][model][L0] pre-attn ...
> [TS][attn] PagedAttention layer=0 backend=TritonAttnBackend
> [TS][attn][triton] forward_extend ...
> [TS][attn][triton] set_kv_buffer ...
> [TS][triton][prefill] launch ...
> [TS][logits] forward ...
> [TS][sampler] greedy.sample ...
> [TS][output] commit batch tokens=[...]
> [TS][output] handle_batch_output ...
> [TS][output] dispatch rid=... -> queue
> [TS][http] SSE chunk rid=... delta_text='Hello'
> ```
>
> Every line is one round-trip stage. If a stage is missing, that's where to
> look.

---

## 5. Special topics worth a separate read

### 5.1 KV pool & paging

- `python/tokenspeed/runtime/cache/allocator.py` — Python-side KV allocator
- `python/tokenspeed/runtime/cache/req_to_token_pool.py` — request→token map
- `tokenspeed-scheduler/csrc/resource/allocator/page_allocator.h` — C++ side
- `tokenspeed-scheduler/csrc/resource/radix_tree/` — prefix-cache radix tree

The paging model: GPU memory is divided into **pages** of `page_size` tokens
(default 64 in TokenSpeed). Each page holds K and V for one layer. The
"page table" is a `[num_requests, max_pages_per_request]` int32 tensor that
the Triton kernel uses to gather scattered K/V into a contiguous block.
`out_cache_loc` (Phase 6) is the flat list of page slots receiving the new
tokens *this step*.

**Print:** in `BaseTokenToKVPool.set_kv_buffer` (find via
`grep -n "def set_kv_buffer" python/tokenspeed/runtime/layers/attention/kv_cache/`):

```python
print(f"[TS][kv] set_kv_buffer L={layer.layer_id} "
      f"loc[:5]={loc[:5].tolist()} k.shape={tuple(k.shape)}", flush=True)
```

### 5.2 CUDA graphs

For decode-only batches the engine captures a graph per supported batch
size (1, 2, 4, ..., max_bs). Replay is much faster than eager because
launch overhead dominates at decode.

- `python/tokenspeed/runtime/execution/cuda_graph_wrapper.py:228–420`

**Print:** top of `CudaGraphWrapper.replay`:

```python
print(f"[TS][cuda_graph] replay bs={bs} "
      f"captured_sizes={sorted(self._captured_bs)}", flush=True)
```

> **Note for learning:** during graph capture, your `print(...)` statements
> in the model forward will fire during capture (once per graph), then *not
> fire on replay* — because graph replay doesn't re-execute Python. To see
> per-step prints, run with `--disable-cuda-graph` first.

### 5.3 Chunked prefill

If a prompt is longer than `--chunked-prefill-size`, the scheduler emits
one chunk per tick and keeps the request in PREFILLING state until the last
chunk. This lets decode of *other* requests run in interleaved ticks.
Watch this with Print 4.2 — you'll see `PrefillChunkOp` for several ticks
on the same `rid`, then a `DecodeOp` once it finishes prefill.

### 5.4 Sampling-batch info

Per-request sampling params (`temperature`, `top_p`, `top_k`, `max_new_tokens`,
`stop`) live in `python/tokenspeed/runtime/sampling/sampling_batch_info.py`.
On the greedy path most are ignored; on flashinfer they're packed into a
batched kernel call. We're using greedy.

### 5.5 Forward modes

`ForwardMode` (`forward_batch_info.py:32–69`):

- `EXTEND` — first chunk of a new prefill (no prior KV)
- `EXTEND_CONTINUE` — subsequent chunks of a long prompt
- `DECODE` — one query token, attending over all prior KV
- `IDLE` — no real work this tick (DP sync placeholder)
- `TARGET_VERIFY` — speculative decoding verify step
- `DRAFT_EXTEND` — speculative decoding draft step

For your single-stream learning, you'll only see `EXTEND` then a long
sequence of `DECODE`s.

---

## 6. How the kernel registry actually picks `triton`

`tokenspeed_kernel.selection.SelectedKernel` is the resolver. Roughly:

1. Look up all kernels registered under family `"attention"`,
   solution name `"mha_decode_with_kvcache"` (or `"mha_prefill"` etc.).
2. Filter by `CapabilityRequirement.satisfied_by(current_platform())`.
3. Filter by traits (sliding window, sinks, dtypes).
4. Of survivors, pick the highest `priority` value.
5. If none survive, raise.

On RTX 6000 Ada (SM 8.9):

- `flashinfer_*` (priority `SPECIALIZED+2`, `min_arch_version=ArchVersion(10,0)`) → filtered out
- `fa3_*` (priority `SPECIALIZED+3`, `min_arch_version=ArchVersion(9,0)`) → filtered out
- `triton_*` (priority `PERFORMANT`, no min) → ✅ **chosen**

That's why your runs all hit the Triton path.

**Print:** in `tokenspeed_kernel/selection.py` `select` method, after the
final selection:

```python
print(f"[TS][registry] select family={family} solution={op_name} "
      f"-> {chosen.name} (priority={chosen.priority})", flush=True)
```

This prints once per unique (family, op, traits) tuple at first call —
exactly what you want for "did the right kernel win?"

---

## 7. Recommended print-instrumentation procedure

You don't add all 22 prints at once. Do it in **three rounds** so output
stays readable.

### Round A — End-to-end skeleton (8 prints)

Phases 1.1, 2.1, 3.2, 4.1, 4.2, 5.2, 7.1, 10.6.

Goal: see one full HTTP→SSE cycle in 8 lines per generated token. If any
line is missing, you immediately know which subsystem broke.

### Round B — Attention deep-dive (6 prints)

Phases 8.1 (gated to `layer_id == 0`), 8.3, 8.4, 8.5, 9.1, 9.2.

Goal: see the attention backend pick decode vs extend, the kernel launch
shapes, and the KV-cache appends. Combined with Round A you can write
out the per-token timeline of the whole engine.

### Round C — Triton GPU internals (gated `tl.device_print`)

Pick **one** kernel and add 2–3 `tl.device_print` lines guarded by
`program_id(0)==0 and program_id(1)==0 and program_id(2)==0`. Re-launch
once with `--max-num-seqs 1` and a short prompt so the kernel fires only a
handful of times. This is the only way to peek at the actual GPU-side state.

> **Triton print quirk:** `tl.device_print` flushes asynchronously. If your
> server keeps running it'll print fine; if you Ctrl+C right after launch
> some output may be lost.

---

## 8. Concrete launch recipe for your RTX 6000 Ada

Once you've added Round A prints:

```bash
# Disable CUDA graphs while learning so prints fire every step
tokenspeed serve Qwen/Qwen3-1.7B \
  --attention-backend triton \
  --moe-backend triton \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --chunked-prefill-size 512 \
  --max-num-seqs 1 \
  --disable-cuda-graph \
  --load-format dummy \
  --host 0.0.0.0 --port 8000 \
  2>&1 | tee /tmp/tokenspeed-trace.log
```

> **Note on `--disable-cuda-graph`:** I'm using the conventional name; check
> `tokenspeed serve --help | grep -i graph` and use whatever flag the CLI
> actually exposes (very likely `--disable-cuda-graph` or
> `--enable-cuda-graph false`).

Then in another terminal:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role":"user","content":"Say hi in 3 words."}],
    "max_tokens": 5,
    "stream": true,
    "temperature": 0
  }'
```

Inspect the trace:

```bash
grep -E "^\[TS\]" /tmp/tokenspeed-trace.log | head -120
```

You should see one EXTEND tick, then ~5 DECODE ticks, then `Finish`.

---

## 9. Reading order summary

If you only have time for a focused study session, read in this order:

1. `python/tokenspeed/cli.py` (10 min)
2. `python/tokenspeed/runtime/entrypoints/engine.py` — Engine class (15 min)
3. `python/tokenspeed/runtime/engine/async_llm.py` — top docstring + class
   header + `generate_request` (20 min)
4. `tokenspeed-scheduler/csrc/scheduler/scheduler.h` — entire header (15 min)
5. `python/tokenspeed/runtime/engine/event_loop.py` — `event_loop` method
   and `_dispatch_forward` (30 min)
6. `python/tokenspeed/runtime/execution/model_runner.py` — `__init__` and
   `forward` (10 min)
7. `python/tokenspeed/runtime/models/llama.py` — read the whole file, it's
   316 lines and the simplest model (30 min)
8. `python/tokenspeed/runtime/layers/attention/backends/triton.py` —
   `forward_decode` and `forward_extend` (15 min)
9. `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/triton/mha_decode.py`
   — read the kernel itself line by line, this is the Triton learning payoff
   (45 min)

Total: about 3 hours of careful reading gets you from zero to "I know how
TokenSpeed produces a token, all the way down to the GPU instructions."

---

## 10. Things to watch out for while instrumenting

1. **Don't print inside hot inner loops.** A 28-layer 4-head model
   running 100 decode steps × 2 attn calls = 5,600 attention prints. Always
   guard with `if layer.layer_id == 0:` or `if step_idx % 16 == 0:`.

2. **`flush=True` matters.** Python buffers stdout when piped to file. Use
   `print(..., flush=True)` everywhere, or set `PYTHONUNBUFFERED=1`.

3. **The scheduler/worker are separate processes.** Your prints will appear
   in a *combined* stream if you use `tee`, but the order across processes
   is not strictly causal — small reorderings are normal. To get strict
   ordering, prefix with a clock: `f"[TS] t={time.monotonic():.4f} ..."`.

4. **CUDA graph capture replays Python only at capture time.** With graphs
   enabled your model-side prints fire only during the warmup capture passes,
   not during steady-state replay. Use `--disable-cuda-graph` while learning;
   re-enable it later to compare timings.

5. **Triton autotune.** First kernel launch may take several seconds while
   Triton autotunes; subsequent launches are fast. Don't mistake this for a
   bug.

6. **Dummy weights produce gibberish.** `--load-format dummy` gives you
   *random* weights so the engine is fully exercised but generated text is
   meaningless. Switch to real weights once your trace is clean.

7. **The codebase is preview-grade.** README literally warns "do not use for
   production" and "several major PRs still in progress." If you hit an
   import error in some optional dependency (e.g. flashinfer build failing),
   work around it locally — don't assume your setup is wrong.

---

## Appendix A — Print-statement quick reference (copy/paste)

A condensed list of the prints, by file, ready to paste:

```text
http_server.py            : route handler (Print 1.1) + SSE loop (10.6)
serving_chat.py:266       : _convert_to_internal_request (1.2)
async_llm.py:269          : generate_request (2.1)
async_llm.py:303          : _tokenize_one_request (2.2)
async_llm.py:316+         : _send_one_request (2.3)
async_llm.py:227          : _result_dispatcher (10.5)
event_loop.py:606         : _init_interprocess_comm           (Phase 3 init)
event_loop.py:623         : _process_new_requests (3.1)
event_loop.py:839         : event_loop (4.1)
event_loop.py:844         : after next_execution_plan (4.2)
event_loop.py:847         : after _get_forward_op (4.3)
event_loop.py:887         : after advance_forward (4.4)
event_loop.py:428         : _dispatch_forward (5.1)
event_loop.py:708         : _commit_forward_results (10.3)
request_handler.py:140    : recv_reqs (3.2)
request_handler.py:194    : handle_generate_request (3.3)
model_executor.py:361     : _forward_step (5.2)
input_buffer.py:101+      : end of fill_input_buffers (6.1)
model_runner.py:115       : ModelRunner.forward (7.1)
models/llama.py:191       : LlamaAttention.forward (7.2)
models/llama.py:101       : LlamaMLP.forward (7.3)
layers/paged_attention.py:59  : PagedAttention.forward (8.1)
layers/attention/backends/base.py:113 : AttentionBackend.forward (8.2)
layers/attention/backends/triton.py:719 : forward_extend (8.4)
layers/attention/backends/triton.py:772 : forward_decode (8.3)
layers/attention/backends/triton.py:736/793 : after set_kv_buffer (8.5)
ops/attention/triton/mha_prefill.py : prefill_attention_fwd (9.2)
ops/attention/triton/mha_decode.py:171  : _decode_att_m_fwd (9.1)
ops/attention/triton/mha_decode.py:33   : tl.device_print inside kernel (9c, optional)
layers/logits_processor.py:157 : LogitsProcessor.forward (10.1)
sampling/backends/greedy.py:153 : GreedySamplingBackend.sample (10.2)
generation_output_processor.py:110 : handle_batch_output (10.4)
execution/cuda_graph_wrapper.py:?  : replay (5.2 cuda-graph print)
tokenspeed_kernel/selection.py     : select (registry pick print)
```

---

## Appendix B — Open questions to answer with your trace

Once Round A prints are in and you've sent one request, you should be able
to answer these from the log alone:

1. How long (wall-clock) does each phase take? (Use `time.monotonic()` in
   each print.)
2. How many ticks pass before the first token is sampled? (Count `tick begin`
   between PUSH and `commit batch tokens`.)
3. How many KV pages does a 100-token prompt consume? (Subtract
   `avail_pages` before vs after.)
4. How many decoder layers fire per step? Confirm by removing the
   `layer_id == 0` guard once.
5. With `--max-num-seqs 4` and 4 concurrent requests, does the scheduler
   batch their decodes into a single forward, or run them serially?

Answering these from the log is the real proof you understand the engine.

---

*End of document. When ready, ask me to generate the actual patch file
(`print_everywhere.patch`) that applies all Round A prints in one go, or
walk you through any one phase in deeper detail.*
