# Anime LoRA

使用 Stable Diffusion 1.5 与 LoRA 进行动漫风格适配，比较逐图详细描述和固定风格描述两种训练条件。项目包含数据预处理、双 GPU 训练、文生图推理和 CLIP 配对评价。

## 实验设置

| 组别 | 训练条件 | 提示词均衡 CLIP 余弦 |
| --- | --- | --- |
| E0 | 原始 SD 1.5，不微调 | 0.3812 |
| E1 | LoRA，风格前缀与逐图 BLIP 描述 | 0.3267 |
| E2 | LoRA，固定风格描述 | 0.3597 |

数据集为 Naruto BLIP Captions，清洗后 1,220 张，训练集 976 张。E1、E2 均完成 2,000 次更新，LoRA rank=16、alpha=16、学习率 1e-4、有效批量 4。三组使用相同的 24 条提示和 4 个种子，各生成 96 张 512×512 图像。

上表使用三组共同未过滤的 77 组配对样本，先按提示词汇总再等权平均。CLIP 衡量图文匹配。已有结果见 [指标汇总](results/analysis/metrics.json) 和 [逐图分数](results/analysis/clip_per_image.csv)。

## 目录

```text
configs/         模型版本、数据版本和训练配置
requirements/    训练与 CPU 评价依赖
src/             数据准备、LoRA 训练与预算控制
scripts/         推理、评价和代码检查工具
tests/           预算、运行初始化和代码导出测试
results/         已有评价指标
```

## 安装

使用 Python 3.11。训练和推理需要 CUDA；先安装与本机 CUDA 匹配的 PyTorch 和 torchvision，再安装训练依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements/train.txt
```

完整对照训练流程需要两张可见的 CUDA GPU，分别训练 E1 与 E2；单图推理只需要一张。两组是独立单卡实验，不采用 DDP。

## 训练

在仓库根目录执行：

```bash
python src/experiment.py orchestrate
```

脚本读取 `configs/experiment.json`，完成数据下载、检查与划分，缓存冻结 VAE 和文本编码器输出，训练两组 LoRA，再统一生成对比图片。每组先运行 50 次更新测速，按预算选取共同更新数，上限为 2,000；每 250 步验证和保存。

默认结果写入 `outputs/experiment/`，模型与输入缓存写入 `.cache/experiment/`。可通过环境变量指定新实验目录：

```bash
EXPERIMENT_ROOT=outputs/run_01 \
EXPERIMENT_CACHE=.cache/run_01 \
python src/experiment.py orchestrate
```

每次实验保存配置快照。同一结果目录中的重启保留原截止时间和恢复状态；独立实验应使用新的结果目录。

## 推理

```bash
python scripts/infer.py \
  --adapter outputs/experiment/E2/final \
  --prompt "an elderly woman in a purple shawl drinking tea on a wooden porch" \
  --seed 101 \
  --output outputs/sample.png
```

省略 `--adapter` 使用原模型；`--scale` 控制 LoRA 强度，默认 1.0。采样步数为 25，CFG 为 7，输出 512×512 图像。

## 评价

CPU 评价建议使用独立环境：

```bash
python3 -m venv .venv-eval
source .venv-eval/bin/activate
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements/eval.txt
python scripts/evaluate_results.py \
  --source outputs/experiment \
  --output outputs/analysis
```

评价器针对完整的本次实验设置，要求每组 96 张生成图、生成清单、2,000 步训练日志和最终适配器，输出逐图 CLIP 余弦与配对统计。

```bash
make check
```


## 来源

- 训练接口参考 [Diffusers v0.35.1 LoRA 示例](https://github.com/huggingface/diffusers/blob/v0.35.1/examples/text_to_image/train_text_to_image_lora.py)。
- 基座模型为 [Stable Diffusion 1.5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)。
- 数据集为 [Naruto BLIP Captions](https://huggingface.co/datasets/lambda/naruto-blip-captions)。
- 评价使用 [CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32)。

