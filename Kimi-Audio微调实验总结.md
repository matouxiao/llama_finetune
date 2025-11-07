# Kimi-Audio 模型在 LLaMAFactory 中的微调实验总结

## 实验目标

使用 LLaMAFactory 对 Kimi-Audio-7B 模型进行 LoRA 微调，实现语音识别（ASR）任务。数据集包含 19 个音频文件。

## 实验环境

- **模型**: Kimi-Audio-7B
- **框架**: LLaMAFactory
- **微调方法**: LoRA (Low-Rank Adaptation)
- **数据集大小**: 19 个音频文件
- **训练设备**: 8 卡 GPU（分布式训练）

## 一、初始配置

### 1.1 添加 Kimi-Audio 自定义模板

在 `llama/LLaMA-Factory/src/llamafactory/data/template.py` 中添加了 `kimi_audio` 模板：

```python
register_template(
    name="kimi_audio",
    format_user=StringFormatter(
        slots=["<|im_user_msg_start|>{{content}}<|im_msg_end|><|im_assistant_msg_start|>"]
    ),
    format_assistant=StringFormatter(slots=["{{content}}<|im_msg_end|>"]),
    format_system=StringFormatter(slots=["<|im_user_msg_start|>{{content}}<|im_msg_end|>"]),
    default_system="You are a helpful assistant provided by Moonshot-AI.",
    stop_words=["<|im_msg_end|>"],
    replace_eos=True,
    mm_plugin=get_mm_plugin(
        name="kimi_audio",
        audio_token="<|im_media_begin|>",
        audio_bos_token="<|im_media_begin|>",
        audio_eos_token="<|im_media_end|>",
    ),
)
```

### 1.2 创建多模态插件

在 `llama/LLaMA-Factory/src/llamafactory/data/mm_plugin.py` 中创建了 `KimiAudioPlugin` 类：

- 跳过了 `feature_extractor` 验证（Kimi-Audio 使用自定义音频处理）
- 实现了 `process_messages` 方法处理音频占位符
- 实现了 `_get_mm_inputs` 方法（不传递音频特征，由模型内部处理）

### 1.3 数据格式转换

创建了转换脚本 `kimi-audio/Kimi-Audio/finetune_codes/demo_data/audio_understanding/convert_to_llamafactory.py`，将 Kimi-Audio 的 JSONL 格式转换为 LLaMAFactory 的 `sharegpt` 格式：

- **输入格式**: JSONL，包含 `task_type` 和 `conversation`
- **输出格式**: JSON，包含 `messages` 和 `audios` 字段
- **音频占位符**: 使用 `<audio>` 标记

### 1.4 数据集注册

在 `llama/LLaMA-Factory/data/dataset_info.json` 中注册数据集：

```json
{
  "kimi_audio_data": {
    "file_name": "kimi_audio_data.json",
    "formatting": "sharegpt",
    "columns": {
      "messages": "messages",
      "audios": "audios"
    },
    "tags": {
      "role_tag": "role",
      "content_tag": "content",
      "user_tag": "user",
      "assistant_tag": "assistant"
    }
  }
}
```

### 1.5 LoRA 训练参数建议（小数据集）

- **num_train_epochs**: 10-20（小数据集需要更多轮次）
- **learning_rate**: 1e-4 到 5e-4
- **gradient_accumulation_steps**: 4-8（增加有效 batch size）
- **save_steps**: 10（频繁保存检查点）
- **lora_rank**: 8-16
- **lora_alpha**: 16-32
- **lora_dropout**: 0.05-0.1

## 二、遇到的错误及解决方法

### 2.1 Git 克隆错误

**错误信息**:
```
fatal: unable to access 'https://github.com/...': HTTP/2 stream 1 was not closed cleanly
```

**解决方法**:
```bash
git config --global http.version HTTP/1.1
```

### 2.2 多模态插件缺失

**错误信息**:
```
ValueError: Multimodal plugin kimi_audio not found.
```

**解决方法**: 在 `mm_plugin.py` 中创建 `KimiAudioPlugin` 类并注册到 `PLUGINS` 字典。

### 2.3 Tokenizer 配置问题

**错误信息**:
```
ValueError: Unrecognized configuration class KimiAudioConfig to build an AutoTokenizer.
```

**解决方法**: 从 HuggingFace (`moonshotai/Kimi-Audio-7B-Instruct`) 下载 tokenizer 文件到本地模型目录。

### 2.4 数据集路径问题

**错误信息**:
```
ValueError: File data/kimi_audio_data.json not found.
```

**解决方法**: 
- 确保数据文件在 `LLaMA-Factory/data/` 目录下
- 在 Web UI 中设置 `Dataset Dir` 为 `data`，`Dataset` 为 `kimi_audio_data`

### 2.5 Feature Extractor 验证失败

**错误信息**:
```
ValueError: Audio feature extractor was not found, please check and update your model file.
```

**解决方法**: 在 `KimiAudioPlugin._validate_input` 中跳过 `feature_extractor` 检查，因为 Kimi-Audio 使用自定义音频处理（Whisper + GLM4 tokenizer）。

### 2.6 Flash Attention 缺失

**错误信息**:
```
RuntimeError: flash attention must be installed
```

**解决方法**: 
1. 安装 flash-attn: `pip install flash-attn`
2. 或者修改 `modeling_moonshot_kimia.py`，使 Flash Attention 可选：

```python
if is_flash_attn_available():
    USE_FLASH_ATTN = True
else:
    USE_FLASH_ATTN = False
    logger.warning("Flash attention is not available, using standard attention instead.")
```

### 2.7 trust_remote_code 警告

**错误信息**:
```
trust_remote_code is not supported anymore.
```

**解决方法**: 在 `loader.py` 中，只为远程数据集传递 `trust_remote_code`，本地文件不传递。

### 2.8 num_proc 参数错误

**错误信息**:
```
ValueError: num_proc must be an integer > 0.
```

**解决方法**: 在 `converter.py` 和 `loader.py` 中，只有当 `preprocessing_num_workers > 0` 时才传递 `num_proc` 参数。

### 2.9 generation_config 为 None

**错误信息**:
```
AttributeError: 'NoneType' object has no attribute 'do_sample'
```

**解决方法**: 在 `patcher.py` 中添加 None 检查：

```python
gen_config = model.generation_config
if gen_config is not None and not gen_config.do_sample and (...):
    gen_config.do_sample = True
```

### 2.10 generate 方法缺失

**错误信息**:
```
AttributeError: 'MoonshotKimiaForCausalLM' object has no attribute 'generate'
```

**解决方法**: 在 `patcher.py` 中为模型添加 `generate` 方法：

```python
if not hasattr(model, "generate"):
    model.generate = MethodType(GenerationMixin.generate, model)
```

### 2.11 prepare_inputs_for_generation 缺失

**错误信息**:
```
AttributeError: 'MoonshotKimiaForCausalLM' object has no attribute 'prepare_inputs_for_generation'
```

**解决方法**: 在 `patcher.py` 中为模型添加该方法（PEFT 需要）：

```python
if not hasattr(model, "prepare_inputs_for_generation"):
    model.prepare_inputs_for_generation = MethodType(GenerationMixin.prepare_inputs_for_generation, model)
```

### 2.12 LoRA Target Modules 匹配问题

**错误信息**:
```
ValueError: Target module MoonshotDecoderLayer(...) is not supported.
```

**解决方法**: 修改 `patch_target_modules` 函数，确保只匹配 Linear 层并返回完整模块路径：

```python
# 只匹配 Linear 或 Conv1D 层
if "Linear" in module.__class__.__name__ or "Conv1D" in module.__class__.__name__:
    module_last_part = name.split(".")[-1]
    if any(target_module == module_last_part for target_module in target_modules):
        module_names.append(name)  # 返回完整路径
```

### 2.13 audio_paths 参数错误

**错误信息**:
```
TypeError: MoonshotKimiaForCausalLM.forward() got an unexpected keyword argument 'audio_paths'
```

**解决方法**: 在 `KimiAudioPlugin._get_mm_inputs` 中不返回 `audio_paths`，因为 Kimi-Audio 的音频处理在模型内部完成。

### 2.14 text_input_ids 为 None

**错误信息**:
```
AttributeError: 'NoneType' object has no attribute 'to'
```

**解决方法**: 在 `modeling_moonshot_kimia.py` 中添加 None 检查：

```python
if text_input_ids is not None:
    text_input_ids = text_input_ids.to(torch.cuda.current_device())
```

### 2.15 Loss 计算缺失

**错误信息**:
```
ValueError: The model did not return a loss from the inputs, only the following keys: logits.
```

**解决方法**: 在 `modeling_moonshot_kimia.py` 的 `forward` 方法中添加 loss 计算：

```python
loss = None
if labels is not None:
    # 使用 text_logits 计算 loss（ASR 任务）
    shift_logits = text_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
```

### 2.16 DDP 未使用参数错误

**错误信息**:
```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one.
Parameter indices which did not receive grad for rank 3: 308 309 310 ...
```

**解决方法**: 在 `parser.py` 中检测 Kimi-Audio 模型并启用 `find_unused_parameters=True`：

```python
# 检测 Kimi-Audio 模型
is_kimi_audio = False
if model_args.model_name_or_path:
    config_path = os.path.join(model_args.model_name_or_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = json.load(f)
            architectures = config_data.get("architectures", [])
            if "KimiAudioModel" in architectures:
                is_kimi_audio = True

if is_kimi_audio:
    training_args.ddp_find_unused_parameters = True
```

### 2.17 日志文件处理错误

**错误信息**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'saves/.../running_log.txt'
AttributeError: 'LoggerHandler' object has no attribute 'thread_pool'
```

**解决方法**: 在 `logging.py` 中：
- 处理 `output_dir` 为 None 的情况
- 添加异常处理，避免文件不存在时出错
- 在 `close` 方法中检查 `thread_pool` 是否存在

## 三、关键代码修改总结

### 3.1 模板和插件 (`template.py`, `mm_plugin.py`)

- 添加 `kimi_audio` 模板注册
- 创建 `KimiAudioPlugin` 类
- 实现自定义音频处理逻辑

### 3.2 模型补丁 (`patcher.py`)

- 添加 `generate` 方法
- 添加 `prepare_inputs_for_generation` 方法
- 处理 `generation_config` 为 None 的情况

### 3.3 模型代码 (`modeling_moonshot_kimia.py`)

- 使 Flash Attention 可选
- 处理 `text_input_ids` 为 None
- 添加 loss 计算逻辑

### 3.4 目标模块匹配 (`visual.py`)

- 修改 `patch_target_modules` 只匹配 Linear 层
- 返回完整模块路径而非简单名称

### 3.5 训练参数 (`parser.py`)

- 为 Kimi-Audio 模型启用 `find_unused_parameters=True`

### 3.6 数据处理 (`loader.py`, `converter.py`)

- 修复 `num_proc` 参数传递
- 修复 `trust_remote_code` 使用

### 3.7 日志处理 (`logging.py`)

- 处理文件不存在的情况
- 修复线程池关闭逻辑

## 四、训练配置建议

### 4.1 Web UI 配置

- **Model Name**: 模型路径（绝对路径）
- **Template**: `kimi_audio`
- **Dataset Dir**: `data`
- **Dataset**: `kimi_audio_data`
- **Finetuning Type**: `lora`
- **LoRA Rank**: 8-16
- **LoRA Alpha**: 16-32
- **LoRA Dropout**: 0.05-0.1
- **Learning Rate**: 1e-4 到 5e-4
- **Num Train Epochs**: 10-20
- **Gradient Accumulation Steps**: 4-8
- **Save Steps**: 10
- **Preprocessing Num Workers**: 0（避免 num_proc 错误）

### 4.2 命令行配置

如果使用命令行，关键参数：

```yaml
model_name_or_path: /path/to/kimi-audio/model
template: kimi_audio
dataset: kimi_audio_data
dataset_dir: data
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.1
learning_rate: 2e-4
num_train_epochs: 15
gradient_accumulation_steps: 4
save_steps: 10
preprocessing_num_workers: 0
```

## 五、经验总结

### 5.1 自定义模型集成要点

1. **模板注册**: 需要正确配置特殊 token 和格式
2. **多模态插件**: 需要实现 `process_messages` 和 `_get_mm_inputs`
3. **模型补丁**: 确保模型有 `generate` 和 `prepare_inputs_for_generation` 方法
4. **Loss 计算**: 自定义模型需要实现 loss 计算逻辑
5. **DDP 配置**: 对于有未使用参数的模型，需要启用 `find_unused_parameters`

### 5.2 常见问题预防

1. **版本兼容性**: 注意 transformers、peft 等库的版本要求
2. **路径问题**: 使用绝对路径避免相对路径问题
3. **设备一致性**: 确保所有张量在同一设备上
4. **参数验证**: 添加 None 检查避免 AttributeError
5. **错误处理**: 添加异常处理提高代码健壮性

### 5.3 调试技巧

1. **逐步测试**: 先测试模型加载，再测试数据处理，最后测试训练
2. **日志查看**: 查看完整的错误堆栈信息
3. **参数检查**: 使用 `print` 或日志输出关键参数的值
4. **简化测试**: 使用小数据集和少量 epoch 快速验证

## 六、最终成功配置

经过所有修复后，训练成功启动。关键配置：

- ✅ 模板: `kimi_audio`
- ✅ 数据集: `kimi_audio_data` (19 个样本)
- ✅ 微调方法: LoRA
- ✅ DDP: `find_unused_parameters=True`
- ✅ Loss 计算: 使用 `text_logits` 计算交叉熵损失
- ✅ 所有必需方法已添加

## 七、后续优化建议

1. **数据增强**: 对于小数据集，可以考虑数据增强技术
2. **学习率调度**: 使用学习率调度器提高训练稳定性
3. **验证集**: 添加验证集监控过拟合
4. **早停机制**: 实现早停避免过拟合
5. **混合精度**: 使用 bf16/fp16 加速训练

---

**实验日期**: 2025-11-07  
**模型版本**: Kimi-Audio-7B-Instruct  
**LLaMAFactory 版本**: 最新版本

