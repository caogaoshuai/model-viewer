# Model Viewer

> 模型结构可视化与对账工具，服务于训练团队 ↔ 部署团队的跨团队协作。

## 一句话说明

**让任何一对模型（训练态 / 部署态 / 不同版本 / 不同量化方案）的结构差异，能在 30 秒内被一张图说清楚。**

## 项目状态

- **当前阶段**：CLI 原型可运行
- **完整 PRD**：[`PRD.md`](./PRD.md)
- **视图范例**：[`docs/examples.md`](./docs/examples.md)

## 核心能力（规划中）

| 能力 | 说明 |
|---|---|
| 字符 tree + 折叠语法 | 精确表达每一层 / 每个 tensor 的 shape、dtype、参数量；同构层自动折叠 |
| 多视图渲染 | Overview / Heatmap / Layer Detail / Key Mapping / Memory / Raw Tree / Key Patterns / Blocks |
| 多维度 Diff | 结构 / 命名 / 参数量 / 异构层 / 显存 五个维度对比 |
| 量化感知 | 自动识别 GPTQ / AWQ / fp8 / LoRA 的辅助 tensor，归属到主 weight |
| MoE / 混合注意力 | 专家组折叠、State Cache 估算、`[A]/[L]` 层标记 |
| 跨格式输出 | term / markdown / mermaid / draw.io XML / html / json |
| 显存估算 | 基于模型参数、dtype、KV cache 配置估算权重与推理显存 |

## 使用说明

更完整的命令说明、输入格式、输出解释和常见场景见 [`docs/usage.md`](./docs/usage.md)。

### 安装与运行

```bash
# 方式 1：仓库内直接运行，无需安装
./mad --help

# 方式 2：安装为 mad 命令
python3 -m pip install -e .
mad --help
```

### 支持的输入

| 输入 | 示例 | 适合场景 |
|---|---|---|
| 模型目录 | `/path/to/Qwen3-0.6B` | 目录里有 `config.json`、`.safetensors` 或 index |
| 配置文件 | `/path/to/config.json` | 只做结构级对比，速度最快 |
| safetensors 文件 | `/path/to/model.safetensors` | 从 header 读取真实 tensor key、shape、dtype |
| safetensors index | `/path/to/model.safetensors.index.json` | 对比权重 key 集合，不下载大 shard |
| 快照 JSON | `model.snapshot.json` | CI 或离线回归对比 |
| 远程模型 ID | `hf://Qwen/Qwen3-0.6B`、`ms://Qwen/Qwen3-1.7B` | 需要安装 `model-viewer[hub]` |

如果只有 `config.json`，工具会按配置合成结构 tensor，这适合看层数、hidden size、MLP size、参数量和显存估算；如果要确认真实权重 key 或 dtype，请提供 `.safetensors` 或 index。

### 常用命令

```bash
# 1. 展示单模型结构，输出 Markdown
mad show /path/to/model --view overview,tree,memory --format markdown

# 2. 只输出 Mermaid 结构图，适合贴到文档
mad show /path/to/model --view overview --format mermaid

# 3. 对比两个模型，输出完整报告
mad diff /path/to/model_a /path/to/model_b --view all --fuzzy-match -o model-diff.md

# 4. 导出快照，后续可以离线 diff
mad snapshot /path/to/model -o model.snapshot.json

# 5. 估算部署侧显存，包含 KV cache
mad memory /path/to/model --mode deploy --seq-len 40960 --batch-size 1

# 6. 展示 safetensors key 折叠图
mad show /path/to/model --view patterns --format markdown

# 7. 展示字符结构图，直观看每个结构块
mad show /path/to/model --view blocks --format markdown
```

### 对比 Qwen3-0.6B 和 Qwen3-1.7B

```bash
# 0.6B 如果没有 index，需要下载 model.safetensors 才能读取真实 key/shape/dtype
modelscope download Qwen/Qwen3-0.6B \
  config.json model.safetensors \
  --local_dir ~/Documents/project/Qwen3-0.6B

# 1.7B 有 index 时，只拉 config + index 即可做 key 集合检查
modelscope download Qwen/Qwen3-1.7B \
  --include config.json '*.safetensors.index.json' \
  --local_dir ~/Documents/project/Qwen3-1.7B

# 结构级对比：建议直接传 config.json，避免把未下载 shard 误判为缺 shape
mad diff \
  ~/Documents/project/Qwen3-0.6B/config.json \
  ~/Documents/project/Qwen3-1.7B/config.json \
  --view overview,memory,tree \
  --format markdown \
  -o ~/Documents/project/qwen3-0.6b-vs-1.7b.md
```

### 视图与输出格式

| 参数 | 说明 |
|---|---|
| `--view overview` | Mermaid 模型框图 |
| `--view heatmap` | diff 热力图，只用于 `mad diff` |
| `--view detail --layer 0` | 单层模块形状对比 |
| `--view mapping` | key 映射表，显示 exact、fused、tied、left-only、right-only |
| `--view memory` | 权重和 KV cache 显存估算 |
| `--view tree` | 折叠后的结构树 |
| `--view patterns` | safetensors key 折叠图，把数字变化位置折叠成 `{0..N}` |
| `--view blocks` | 字符结构图，展示 Embedding、Decoder、Attention、MLP/MoE、混合层调度、ViT/MTP、Norm、LM Head |
| `--view all` | 输出所有核心视图 |
| `--format markdown` | 适合写报告或贴文档 |
| `--format json` | 适合 CI 消费 |
| `--format drawio` | 输出 draw.io XML |

### 当前实现状态

| 能力 | 当前状态 |
|---|---|
| 本地 `config.json` | 已支持；无权重 metadata 时按配置合成结构 tensor |
| `.safetensors` / index | 已支持读取 header，不加载权重内容 |
| 快照 JSON | 已支持 `mad snapshot` 导出与 `mad diff` 输入 |
| HF / ModelScope ID | 可选支持；安装 `model-viewer[hub]` 后拉取轻量 metadata |
| 结构 Diff | exact / dtype / shape / left-only / right-only |
| Fuzzy 对账 | qkv fuse、gate/up fuse、tied embedding、量化辅助 tensor |
| 输出格式 | term / markdown / mermaid / draw.io XML / html / json |

### safetensors key 折叠图

`patterns` 视图会按 key 中纯数字 token 的变化位置自动折叠，适合快速确认层、专家、分片等重复结构：

```text
原始 key:
model.layers.0.mlp.experts.0.weight
model.layers.0.mlp.experts.1.weight
...
model.layers.0.mlp.experts.63.weight

折叠后:
model.layers.{0}.mlp.experts.{0..63}.weight  x64
```

连续数字会显示成 `{0..63}`；固定数字但属于同一折叠组时显示成 `{0}`；不连续数字会显示成 `{0..3,28..31}`。

### 字符结构图

`blocks` 视图用字符盒图展示模型主干和 Decoder Block 内部结构，适合快速看清结构块和数据流：

```bash
mad show /path/to/model --view blocks --format markdown
```

输出会包含：

- `TOKEN EMBEDDING`
- `LANGUAGE DECODER STACK x N`
- 普通 GQA/Attention 下的 `q_proj / k_proj / v_proj / o_proj`
- Qwen3.5 等混合模型的 `HYBRID LAYER SCHEDULE`、DeltaNet/GQA 宏块比例、KV Cache/State Cache 层数
- 多模态 Qwen3.5 的 `MULTIMODAL INPUT ROUTER`、ViT blocks、visual merger、MTP side head
- `Dense SwiGLU MLP` 或 `SwiGLU MoE MLP`，MoE 会展示 router、Top-K、experts、shared expert
- `FINAL NORM`
- `LM HEAD`

Qwen3.5 示例：

```bash
mad show ~/Documents/project/Qwen3.5-0.8B/Qwen3.5-0.8B \
  --view blocks,patterns \
  --format markdown
```

典型输出会把 `layer_types` 折成：

```text
HYBRID LAYER SCHEDULE
DeltaNet/linear=18  GQA/full=6  O(T^2) share=25.0%
macro-block x6: [L1:DeltaNet -> L2:DeltaNet -> L3:DeltaNet -> L4:GQA]
DeltaNet layers: {0..2,4..6,8..10,12..14,16..18,20..22}
GQA layers: {3,7,11,15,19,23}
KV Cache layers=6; State Cache layers=18
```
