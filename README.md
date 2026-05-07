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
| 多视图渲染 | Overview（Mermaid）/ Heatmap / Layer Detail / Key Mapping / Memory / Raw Tree 共 6 种视图 |
| 多维度 Diff | 结构 / 命名 / 参数量 / 异构层 / 显存 五个维度对比 |
| 量化感知 | 自动识别 GPTQ / AWQ / fp8 / LoRA 的辅助 tensor，归属到主 weight |
| MoE / 混合注意力 | 专家组折叠、State Cache 估算、`[A]/[L]` 层标记 |
| 跨格式输出 | term / markdown / mermaid / svg / html / json |
| 显存联动 | 复用 `training-resource-estimator` 估算训练 / 推理显存 |

## CLI 草案

```bash
mad show <model> [--view all|overview|tree|heatmap|memory|...]
mad diff <model_a> <model_b> [--view ...] [--fuzzy-match]
mad snapshot <model> -o snap.json
mad memory <model> [--mode train|deploy]
```

## 快速开始

```bash
# 仓库内直接运行
./mad show tests/fixtures/model_a --view overview --format mermaid

# 或安装为 mad 命令
python3 -m pip install -e .

# 展示单模型结构：支持本地模型目录、config.json、safetensors/index、snapshot JSON
mad show /path/to/model --view overview,tree,memory --format markdown

# 对比训练态与部署态；fuzzy-match 会识别 qkv/gate_up fuse、量化辅助 tensor、tied lm_head
mad diff /train/model /deploy/model --view all --fuzzy-match -o train-vs-deploy.md

# 导出离线快照给 CI 或后续回归
mad snapshot /path/to/model -o model.snapshot.json

# CI 卡点：出现非 exact diff 时返回退出码 2
mad diff baseline.json /path/to/model --format json --fail-on-change
```

当前实现是离线优先的轻量 CLI：

| 能力 | 当前状态 |
|---|---|
| 本地 `config.json` | 已支持；无权重 metadata 时按配置合成结构 tensor |
| `.safetensors` / index | 已支持读取 header，不加载权重内容 |
| 快照 JSON | 已支持 `mad snapshot` 导出与 `mad diff` 输入 |
| HF / ModelScope ID | 可选支持；安装 `model-viewer[hub]` 后拉取轻量 metadata |
| 结构 Diff | exact / dtype / shape / left-only / right-only |
| Fuzzy 对账 | qkv fuse、gate/up fuse、tied embedding、量化辅助 tensor |
| 输出格式 | term / markdown / mermaid / draw.io XML / html / json |

## 关联项目

- **训练资源估算**：`/Users/cgs/Documents/project/training-resource-estimator/`
  - 借用其 config 解析、显存估算公式
- **闭源模型上架协作**：`dashscope/dashscope-finetune/docs/closed-source-model-onboarding-collaboration.md`
  - 一线协作场景源头
