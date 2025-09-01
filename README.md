# 中文情感分类 Demo

## 项目简介
- 使用中文 RoBERTa 微调情感分类任务
- 部署 Gradio Web Demo，可在线输入文本预测情感

## 快速运行
1. 确保模型文件夹 chinese_sentiment_model 存在
2. 安装依赖：
   pip install -r requirements.txt
3. 运行：
   python app.py
4. 部署到 Hugging Face Spaces：上传整个仓库即可

## 技术栈
- Transformers (hfl/chinese-roberta-wwm-ext)
- PyTorch
- Gradio
