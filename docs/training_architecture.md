# VocalRender Training Architecture

本文档描述训练代码的组织方式。`scripts/train_vocalrender_svs.py`
只负责读取 YAML / CLI 覆盖项并调用包内 runner。

## 入口层

- `scripts/train_vocalrender_svs.py`

脚本不维护训练细节，只做：

1. 解析 `--config_path`
2. 解析 `--set key=value` 覆盖项
3. 调用 `vocalrender.training.runners.svs.run(config)`

## 配置层

`src/vocalrender/training/config.py` 提供：

- `SVSTrainConfig`（typed dataclass）
- 散平 YAML 到分组配置的 normalize 兼容层

统一分组如下：

- `model`
- `data`
- `train`
- `eval`
- `runtime`
- `dist`

YAML 兼容两种写法：字段可以散落在顶层（自动归组），也可以直接写成
`model:` / `data:` / `train:` 等分组块。

## Runner 层

`src/vocalrender/training/runners/` 是训练编排层。

- `svs.py`
  - `build_runtime`
  - `build_model_and_tokenizer`
  - `build_datasets_and_loaders`
  - `build_eval_context`
  - `build_training_state`
  - `train_one_step`
  - `run_validation_and_checkpoint`

## 公共基础设施

`src/vocalrender/training/runtime.py` 统一封装：

- 运行目录与 TensorBoard 初始化
- optimizer / scheduler 构建
- checkpoint / resume 装配
- runtime state 收集

`src/vocalrender/training/loop_schedule.py` 统一封装：

- 验证触发条件
- 音频评估触发条件
- checkpoint 保存条件

## 数据与验证

- `svs_loading.py`：Arrow 数据加载与 train/val 切分
- `svs_data.py`：SVS dataset wrapper / dataloader（score masking、
  prompt-audio 拼接、动态 batch）
- `validation.py`：验证流程控制（loss eval + 音频 eval 调度）
- `val_audio.py`：多 GPU 验证音频生成、SingMOS/AES 指标与 TensorBoard 记录

## 兼容策略

- 散平与分组两种 YAML 键写法均可用
- checkpoint 目录结构和训练日志 key 与内部实验版本保持一致，旧
  checkpoint（纯 state_dict）可直接加载
