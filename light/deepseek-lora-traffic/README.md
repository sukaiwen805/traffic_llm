# deepseek-lora-medical
利用简单的代码完成deepseek基于自定义数据集的lora微调
# 交通模型微调项目

本项目使用LoRA技术对DeepSeek-R1-Distill-Qwen-7B模型进行微调，以提升其在交通问题上的回答能力。通过参数高效微调技术，使模型能够根据用户提出的问题提供详细的分析和结果。

## 模型简介

基础模型：[DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)

该模型是DeepSeek团队基于Qwen架构开发的轻量级大语言模型，经过蒸馏优化，在保持良好性能的同时大幅减小了模型体积，适合在有限计算资源下进行微调和部署。

## 数据集

使用了自定义的交通问答数据集`data.json`，包含交通调整问题、复杂思维链和标准回答。数据格式如下：
json
{
"Question": "用户提出的交通问题",
"Complex_CoT": "详细的思维分析过程",
"Response": "结果的规范回答"
}

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- peft 0.4+
- datasets
- matplotlib
- CUDA环境（GPU必须）

## 安装指南
pip install torch torchvision torchaudio
pip install transformers peft datasets matplotlib
## 使用方法

1. 准备数据集文件`data.json`
2. 调整配置参数（如需要）
3. 运行训练脚本

## 训练参数说明

- **LoRA配置**:
  - r=16：低秩矩阵的秩
  - lora_alpha=32：LoRA缩放参数
  - target_modules=["q_proj", "v_proj"]：目标更新的模块
  - lora_dropout=0.05：防止过拟合的dropout率

- **训练配置**:
  - 批处理大小：2（使用梯度累积实现等效批大小为8）
  - 训练轮次：3
  - 学习率：3e-3
  - 使用FP16混合精度训练

## 微调结果

训练完成后，模型会保存在`models`目录下，同时生成训练过程的损失曲线图`loss_curve.png`，用于评估模型训练的稳定性和有效性。

## 模型评估与应用

微调后的模型可用于：
1. 交通场景优化调整

## 项目优势

1. 参数高效训练：使用LoRA技术，只训练很小一部分参数
2. 资源友好：针对显存进行了优化设计
3. 可视化监控：提供训练过程的损失曲线
4. 交通领域专精：针对交通场景优化


