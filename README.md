# 基层慧眼（高级版）

多模态基层医疗影像辅助诊断系统：支持 **X光图像 + 病历文本** 输入，输出结构化报告草稿，并提供医生在线编辑、病例管理、API 部署、训练蒸馏与性能评估链路。

## 1. 已实现能力

- 产品闭环：上传影像 -> 生成草稿 -> 医生编辑 -> 保存病例 -> 病例回看
- 报告结构化：固定 6 个模块（结论/征象/风险/检查建议/处理建议/科普）
- 基础安全：支持页面账号密码（`APP_AUTH_USER` / `APP_AUTH_PASS`）
- 日志与稳定性：推理重试、事件日志、错误提示
- 工程部署：Web（Gradio）+ API（FastAPI）
- 训练链路：IU 数据预处理、Teacher LoRA、Student Distillation、辅助分类训练
- 评估与压测：ROUGE/BLEU、AUC/F1、时延/吞吐/显存

## 2. 快速启动（Web）

```bash
cd your-project-directory
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt

# 可选：页面登录保护
export APP_AUTH_USER=doctor
export APP_AUTH_PASS=123456

# 可选：设置模型缓存路径（推荐使用本地大盘）
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/path/to/your/model/cache

python src/app_gradio.py
```

## 3. API 启动

```bash
source .venv/bin/activate
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

核心接口：
- `GET /health`
- `POST /infer`
- `POST /cases`
- `GET /cases`
- `GET /cases/{case_id}`

## 4. IU 数据训练链路

### 4.1 数据预处理（Kaggle IU 压缩包解压后）

```bash
python src/train/prepare_iu_for_training.py \
  --root path/to/iu_dataset \
  --out-dir data/iu_train
```

生成：
- `data/iu_train/train.jsonl`
- `data/iu_train/val.jsonl`
- `data/iu_train/test.jsonl`
- `data/iu_train/stats.json`

### 4.2 Teacher LoRA 训练（7B）

```bash
python src/train/run_with_config.py \
  --script src/train/train_teacher_lora.py \
  --config configs/train_teacher_lora.yaml
```

### 4.3 Student 蒸馏（3B）

```bash
python src/train/run_with_config.py \
  --script src/train/distill_student.py \
  --config configs/distill_student.yaml
```

### 4.4 辅助任务：normal/abnormal 分类

```bash
python src/train/train_aux_classifier.py \
  --train data/iu_train/train.jsonl \
  --val data/iu_train/val.jsonl \
  --out output/aux_classifier.pt
```

## 5. 评估与性能对比

### 报告质量

```bash
python src/eval/evaluate_reports.py \
  --pred output/demo_predictions.jsonl \
  --out output/report_metrics.json
```

### 分类指标

```bash
python src/eval/evaluate_classifier.py \
  --pred output/cls_predictions.jsonl \
  --out output/classifier_metrics.json
```

### 推理性能（P50/P95/吞吐/显存）

```bash
python src/bench/benchmark_inference.py \
  --image examples/chest_xray_sample.jpg \
  --note examples/sample_note.txt \
  --model-id Qwen/Qwen3-VL-4B-Instruct
```

### 多版本对比（Teacher / Student / Quantized）

```bash
python src/bench/benchmark_variants.py \
  --config configs/benchmark_variants.yaml \
  --image examples/chest_xray_sample.jpg \
  --note examples/sample_note.txt
```

## 6. 一键串行流程（示例）

```bash
bash scripts_run.sh
```

## 7. 项目结构

- `src/app_gradio.py`：医生交互页面 + 病例管理
- `src/api_server.py`：服务化 API
- `src/inference.py`：多模态推理核心（重试 + 结构化）
- `src/core/`：报告解析、病例存储
- `src/train/`：数据预处理、Teacher LoRA、Student 蒸馏、辅助分类
- `src/eval/`：报告与分类评估
- `src/bench/`：性能压测与多版本对比
- `configs/`：训练、蒸馏、性能对比配置

## 8. 合规说明

- 本项目仅用于课程/研究，不可替代医生诊断。
- 医疗数据使用需遵守数据源协议与机构合规要求。
