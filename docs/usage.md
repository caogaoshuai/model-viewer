# Model Viewer 使用指南

本文说明如何用 `mad` 做模型结构展示、模型对比、快照导出和显存估算。

## 1. 安装

在仓库目录内可以直接运行：

```bash
./mad --help
```

也可以安装成本机命令：

```bash
python3 -m pip install -e .
mad --help
```

远程模型 ID 解析依赖可选包：

```bash
python3 -m pip install -e '.[hub]'
```

## 2. 输入格式

`mad` 的输入可以是模型目录、单个配置文件、safetensors 文件、safetensors index 或快照 JSON。

| 输入 | 示例 | 说明 |
|---|---|---|
| 模型目录 | `/models/Qwen3-0.6B` | 自动查找 `config.json`、`.safetensors` 和 index |
| `config.json` | `/models/Qwen3-0.6B/config.json` | 只做结构级分析；不会读取真实权重 key |
| `.safetensors` | `/models/Qwen3-0.6B/model.safetensors` | 从 header 读取真实 key、shape、dtype，不加载 tensor 内容 |
| index JSON | `/models/Qwen3-1.7B/model.safetensors.index.json` | 读取权重 key 和 shard 映射；如果 shard 不在本地，shape/dtype 依赖 config 推导 |
| 快照 JSON | `baseline.snapshot.json` | 由 `mad snapshot` 生成，用于 CI 和离线 diff |

建议：

- 只关心层数、hidden size、MLP size、参数量时，传 `config.json` 最快。
- 关心真实权重 key、dtype、单 tensor shape 时，传模型目录或 `.safetensors`。
- 只有 index 没有 shard 时，适合做 key 集合检查；结构报告建议直接传 `config.json`。

## 3. 展示单模型

输出 Markdown 报告：

```bash
mad show /path/to/model --view overview,tree,memory --format markdown -o model.md
```

只输出 Mermaid：

```bash
mad show /path/to/model --view overview --format mermaid
```

查看指定层：

```bash
mad show /path/to/model --view detail --layer 0 --format markdown
```

导出 draw.io XML：

```bash
mad show /path/to/model --view overview --format drawio -o model.drawio.xml
```

输出 safetensors key 折叠图：

```bash
mad show /path/to/model --view patterns --format markdown
```

`patterns` 视图按 key 中纯数字 token 的变化位置自动折叠。例如：

```text
原始:
model.layers.0.mlp.experts.0.weight
model.layers.0.mlp.experts.1.weight
...
model.layers.0.mlp.experts.63.weight

折叠:
model.layers.{0}.mlp.experts.{0..63}.weight  x64
```

规则：

- 只有纯数字分段会折叠，例如 `layers.0`、`experts.63`。
- 连续数字显示为 `{start..end}`。
- 固定数字但属于同一折叠组时显示为 `{0}`。
- 不连续数字显示为 `{0..3,28..31}`。
- shape 或 dtype 不同时会拆成不同折叠组，避免隐藏真实结构差异。

输出字符结构图：

```bash
mad show /path/to/model --view blocks --format markdown
```

`blocks` 视图用纯字符盒图展示主干数据流和每个 Decoder Block 内部结构块，例如：

```text
TOKEN EMBEDDING
        │
        ▼
DECODER BLOCK x 28
├─ RMSNorm
├─ Attention
│  ├─ q_proj
│  ├─ k_proj
│  ├─ v_proj
│  └─ o_proj
├─ Residual Add
├─ RMSNorm
├─ MLP
│  ├─ gate_proj
│  ├─ up_proj
│  └─ down_proj
└─ Residual Add -> next layer
```

对 MoE 模型，MLP 区域会显示 router、expert 数和 active expert 数。

## 4. 对比两个模型

基础对比：

```bash
mad diff /path/to/model_a /path/to/model_b --view all --format markdown -o diff.md
```

训练态和部署态对账时开启 fuzzy match：

```bash
mad diff /train/model /deploy/model --view all --fuzzy-match -o train-vs-deploy.md
```

`--fuzzy-match` 会识别：

- `q_proj + k_proj + v_proj` 到 `qkv_proj` 的融合
- `gate_proj + up_proj` 到 `gate_up_proj` 的融合
- `lm_head.weight` 和 embedding tied 的等价关系
- 量化产生的 `scales`、`zeros`、`g_idx` 等辅助 tensor

CI 中可以用 JSON 输出和失败码：

```bash
mad diff baseline.snapshot.json /path/to/current \
  --view all \
  --format json \
  --fail-on-change \
  -o diff.json
```

当存在非 exact 差异时，`--fail-on-change` 返回退出码 `2`。

## 5. 快照

导出快照：

```bash
mad snapshot /path/to/model -o model.snapshot.json
```

快照包含：

- 归一化后的模型 profile
- config 关键信息
- tensor key、shape、dtype、kind
- 解析 warning

快照适合提交到 CI 基线，也适合在无法访问原始模型目录时做离线 diff。

## 6. 显存估算

训练侧估算只统计权重相关 bucket：

```bash
mad memory /path/to/model --mode train --format markdown
```

部署侧估算会额外加入 KV cache：

```bash
mad memory /path/to/model \
  --mode deploy \
  --seq-len 40960 \
  --batch-size 1 \
  --format markdown
```

当前显存估算是静态近似值，适合对账权重规模和 KV cache 规模，不等同于真实推理峰值显存。

## 7. Qwen3-0.6B vs Qwen3-1.7B 示例

下载 metadata 和必要权重：

```bash
modelscope download Qwen/Qwen3-0.6B \
  config.json model.safetensors \
  --local_dir ~/Documents/project/Qwen3-0.6B

modelscope download Qwen/Qwen3-1.7B \
  --include config.json '*.safetensors.index.json' \
  --local_dir ~/Documents/project/Qwen3-1.7B
```

结构级对比：

```bash
mad diff \
  ~/Documents/project/Qwen3-0.6B/config.json \
  ~/Documents/project/Qwen3-1.7B/config.json \
  --view overview,memory,tree \
  --format markdown \
  -o ~/Documents/project/qwen3-0.6b-vs-1.7b.md
```

真实 key 集合检查：

```bash
mad diff \
  ~/Documents/project/Qwen3-0.6B \
  ~/Documents/project/Qwen3-1.7B/model.safetensors.index.json \
  --view mapping \
  --format markdown
```

查看两侧 safetensors key 折叠图：

```bash
mad diff \
  ~/Documents/project/Qwen3-0.6B \
  ~/Documents/project/Qwen3-1.7B/model.safetensors.index.json \
  --view patterns \
  --format markdown
```

查看两侧字符结构图：

```bash
mad diff \
  ~/Documents/project/Qwen3-0.6B/config.json \
  ~/Documents/project/Qwen3-1.7B/config.json \
  --view blocks \
  --format markdown
```

## 8. 输出解读

| 状态 | 含义 |
|---|---|
| `exact` | key、shape、dtype 完全一致 |
| `equivalent` | 语义等价但存在 dtype、fuse、tied 等差异 |
| `different` | shape 不一致 |
| `left_only` | 只存在于左侧模型 |
| `right_only` | 只存在于右侧模型 |
| `auxiliary` | 量化辅助 tensor |

key 折叠图字段：

| 字段 | 含义 |
|---|---|
| `Safetensor Key Folding [311 keys -> 13 patterns]` | 原始 key 数和折叠后的模式数 |
| `model.layers.{0..27}.self_attn.q_proj.weight` | 数字位置折叠后的 key 模式 |
| `x28` | 该模式覆盖的真实 key 数 |
| `[2048,1024] BF16` | 该组 tensor 的 shape 和 dtype |

字符结构图字段：

| 字段 | 含义 |
|---|---|
| `TOKEN EMBEDDING` | 词表 embedding 入口 |
| `DECODER BLOCK x 28` | 重复的 decoder 层数 |
| `Attention` | Q/K/V/O 投影与 head 配置 |
| `MLP` | gate/up/down 或 MoE experts |
| `FINAL NORM` | 输出前归一化 |
| `LM HEAD` | 输出头，包含 tied embedding 标记 |

热力图符号：

| 符号 | 含义 |
|---|---|
| `░` | 完全一致 |
| `▓` | 等价但有差异 |
| `█` | 真实差异 |
| `!` | 左侧独有 |
| `+` | 右侧独有 |

## 9. 常见问题

### 为什么只传 `config.json` 也能对比？

工具会从 config 推导标准 Transformer 结构，合成用于展示和参数估算的 tensor 列表。这样速度最快，但不能代表真实 checkpoint 是否包含某个 key。

### 为什么 index 不显示 shape？

safetensors index 通常只包含 `weight_map`，不包含每个 tensor 的 shape 和 dtype。要读取真实 shape/dtype，需要本地有 `.safetensors` shard。

### 为什么 config 写了 tied embedding，但 checkpoint 里还有 `lm_head.weight`？

这类模型可能在逻辑架构上 tied，但 checkpoint 仍单独存储 `lm_head.weight`。结构级参数量和实际 checkpoint 文件大小可能因此不同。
