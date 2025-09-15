# 综述撰写AI代理训练

本项目旨在训练一个用于自动撰写学术综述的AI代理。整个训练流程分为三个主要阶段：

1.  **大纲微调 (Outline Finetuning)**：训练模型根据主题生成结构合理、逻辑清晰的综述大纲。
2.  **内容微调 (Content Finetuning)**：在给定大纲的条件下，训练模型生成详细、流畅的综述内容。
3.  **强化学习 (Reinforcement Learning)**：使用GRPO等算法对模型进行对齐，优化生成结果的质量，使其更符合人类偏好。

## 项目结构

```
/
├───ART_Langgraph_content/  # 内容生成模型训练
│   ├───train.py
│   └───requirements.txt
│
├───ART_Langgraph_outline/  # 大纲生成模型训练
│   ├───train.py
│   └───requirements.txt
│
├───Unsloth/                # SFT、强化学习训练与推理
│   ├───train_sft.py
│   ├───train_grpo.py
│   ├───inference_sft.py
│   ├───inference_grpo.py
│   └───requirements.txt
│
└───README.md               # 本文档
```

## 环境设置

每个训练阶段的目录（`ART_Langgraph_content`, `ART_Langgraph_outline`, `Unsloth`）都包含独立的 `requirements.txt` 文件。请在各自目录下安装所需依赖。

例如，为 `Unsloth` 目录设置环境：
```bash
cd Unsloth
pip install -r requirements.txt
```

同样地，为 `ART_Langgraph_content` 和 `ART_Langgraph_outline` 目录执行相应操作。

**环境变量**:
部分脚本可能需要配置环境变量（如API密钥、模型路径等）。请检查每个目录下的 `.env` 或 `env_template` 文件，并根据需要创建或修改 `.env` 文件。

## 训练流程

请按照以下步骤顺序执行训练。

### 第一步：大纲微调

此阶段训练模型生成综述大纲。

1.  进入大纲训练目录：
    ```bash
    cd ART_Langgraph_outline
    ```
2.  开始训练：
    ```bash
    python train.py
    ```
3.  训练完成后，会生成一个用于生成大纲的微调模型。

### 第二步：内容微调

此阶段基于第一步生成的大纲，训练模型填充详细内容。

1.  进入内容训练目录：
    ```bash
    cd ART_Langgraph_content
    ```
2.  开始训练：
    ```bash
    python train.py
    ```
3.  训练完成后，会生成一个用于生成内容的微调模型。

### 第三步：SFT 与强化学习

此阶段在 `Unsloth` 目录中进行，使用更高效的训练方法对模型进行最终优化。

1.  进入 `Unsloth` 目录：
    ```bash
    cd Unsloth
    ```

2.  **SFT (Supervised Finetuning)** (可选):
    如果需要在此阶段进行额外的监督微调，请运行：
    ```bash
    python train_sft.py
    ```

3.  **GRPO (Ghost Reward Preference Optimization) 强化学习**:
    这是优化的关键步骤，通过强化学习使模型的输出更符合人类偏好。
    ```bash
    python train_grpo.py
    ```

## 模型推理

训练完成后，您可以使用 `Unsloth` 目录中的推理脚本来测试模型效果。

-   **测试SFT模型**:
    ```bash
    python inference_sft.py
    ```

-   **测试GRPO模型**:
    ```bash
    python inference_grpo.py
    ```