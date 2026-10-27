# 评价结果

仓库保留三个文本结果文件：

- [metrics.json](analysis/metrics.json)：训练统计、CLIP 汇总和配对差值区间。
- [clip_per_image.csv](analysis/clip_per_image.csv)：288 张输出的逐图分数与过滤标记。
- [clip_model.json](analysis/clip_model.json)：冻结评价模型的版本。

历史训练日志、全部生成图像和适配器保留在本地 `results/anime_lora/`，由 Git 忽略。新实验默认输出到 `outputs/experiment/`。评价方法与运行命令见[项目说明](../README.md)。
