import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import time
import json

class ChineseSentimentClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.labels = ['负面', '中性', '正面']  # 三分类：对应label2id = {"negative": 0, "neutral": 1, "positive": 2}
        self.loaded = False
        
    def load_model(self):
        """加载模型"""
        try:
            # 从本地文件夹加载模型
            self.tokenizer = AutoTokenizer.from_pretrained('./chinese_sentiment_model')
            self.model = AutoModelForSequenceClassification.from_pretrained('./chinese_sentiment_model')
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            return "✅ 模型加载成功！"
        except Exception as e:
            return f"❌ 模型加载失败：{str(e)}"
    
    def predict_sentiment(self, text):
        """情感预测"""
        if not self.loaded:
            return "❌ 模型未加载，请先加载模型"
        
        if not text.strip():
            return "❌ 请输入有效文本"
        
        try:
            # 文本预处理
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
                confidence = predictions[0][predicted_class].item()
            
            # 获取所有类别的概率
            probs = predictions[0].cpu().numpy()
            
            result = {
                'predicted_label': self.labels[predicted_class],
                'confidence': confidence,
                'all_scores': {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
            }
            
            return result
            
        except Exception as e:
            return f"❌ 预测失败：{str(e)}"

# 初始化分类器
classifier = ChineseSentimentClassifier()

def format_prediction_result(result):
    """格式化预测结果"""
    if isinstance(result, str):
        return result
    
    if isinstance(result, dict) and 'predicted_label' in result:
        # 构建结果显示
        pred_label = result['predicted_label']
        confidence = result['confidence']
        all_scores = result['all_scores']
        
        # 情感emoji
        emoji_map = {"正面": "😊", "中性": "😐", "负面": "😞"}
        emoji = emoji_map.get(pred_label, "🤔")
        
        output = f"""
## {emoji} 预测结果

**情感类别**: {pred_label}  
**置信度**: {confidence:.4f} ({confidence*100:.2f}%)

### 📊 详细分数：
"""
        
        for label, score in all_scores.items():
            bar_length = int(score * 20)  # 简单的文本进度条
            bar = "█" * bar_length + "░" * (20 - bar_length)
            output += f"- **{label}**: {score:.4f} `{bar}` {score*100:.2f}%\n"
        
        # 置信度解释
        if confidence > 0.8:
            confidence_desc = "🔥 高置信度"
        elif confidence > 0.6:
            confidence_desc = "✅ 中等置信度"
        else:
            confidence_desc = "⚠️ 低置信度"
        
        output += f"\n**置信度评估**: {confidence_desc}"
        
        return output
    
    return str(result)

def predict_interface(text):
    """预测接口"""
    if not text.strip():
        return "请输入要分析的文本内容"
    
    start_time = time.time()
    result = classifier.predict_sentiment(text)
    end_time = time.time()
    
    formatted_result = format_prediction_result(result)
    
    # 添加性能信息
    if isinstance(result, dict):
        inference_time = (end_time - start_time) * 1000
        formatted_result += f"\n\n⚡ **推理时间**: {inference_time:.2f}ms"
    
    return formatted_result

def batch_predict_interface(texts):
    """批量预测接口"""
    if not texts.strip():
        return "请输入要分析的文本内容（每行一个）"
    
    lines = [line.strip() for line in texts.split('\n') if line.strip()]
    if not lines:
        return "请输入有效的文本内容"
    
    results = []
    start_time = time.time()
    
    for i, text in enumerate(lines[:10]):  # 限制最多10条
        result = classifier.predict_sentiment(text)
        if isinstance(result, dict):
            label = result['predicted_label']
            confidence = result['confidence']
            emoji_map = {"正面": "😊", "中性": "😐", "负面": "😞"}
            emoji = emoji_map.get(label, "🤔")
            results.append(f"{i+1}. {emoji} {text[:50]}... → **{label}** ({confidence:.3f})")
        else:
            results.append(f"{i+1}. ❌ {text[:50]}... → 预测失败")
    
    end_time = time.time()
    total_time = (end_time - start_time) * 1000
    
    output = "## 📋 批量预测结果\n\n" + "\n".join(results)
    output += f"\n\n⚡ **总处理时间**: {total_time:.2f}ms | **平均时间**: {total_time/len(lines):.2f}ms/条"
    
    return output

def load_model_interface():
    """加载模型接口"""
    return classifier.load_model()

# 示例文本 - 对应三分类
EXAMPLE_TEXTS = {
    "正面评价": "这部电影真的很好看，演员表演很棒，剧情吸引人！",
    "负面评价": "剧情很差，浪费时间，难看透顶，浪费钱。",
    "中性评价": "一般般，不太喜欢，还行，没有特别惊喜。",
    "复杂情感": "虽然有些地方不太满意，但总体来说还可以接受。"
}

# 创建Gradio界面
with gr.Blocks(
    title="🎭 中文情感分类系统",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {max-width: 1400px; margin: auto;}
    .example-box {background: #f0f0f0; padding: 10px; border-radius: 5px; margin: 5px 0;}
    """
) as demo:
    
    gr.Markdown("""
    # 🎭 中文情感分类系统演示
    
    **基于BERT的中文情感分析模型** - 展示预训练模型微调和部署能力
    
    🔥 **技术特点**：BERT微调 + 高精度分类 + 实时推理 + 批量处理
    """)
    
    # 模型加载区域
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🚀 模型管理")
            load_btn = gr.Button("📥 加载情感分类模型", variant="primary", size="lg")
            load_status = gr.Textbox(label="📊 加载状态", value="模型未加载", interactive=False)
        
        with gr.Column(scale=1):
            gr.Markdown("### 📈 模型信息")
            gr.Markdown("""
            - **基础模型**: BERT-base-chinese
            - **任务类型**: 二分类情感分析  
            - **训练数据**: 中文评论数据集
            - **输出类别**: 正面 / 负面
            """)
    
    gr.Markdown("---")
    
    # 主要功能区域
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 单文本情感分析")
            
            text_input = gr.Textbox(
                label="📝 输入文本",
                placeholder="输入要分析情感的中文文本...",
                lines=4,
                max_lines=10
            )
            
            predict_btn = gr.Button("🎯 分析情感", variant="primary")
            
            # 示例文本按钮
            gr.Markdown("**📚 快速测试样例**：")
            with gr.Row():
                for label, text in EXAMPLE_TEXTS.items():
                    example_btn = gr.Button(label, variant="secondary", size="sm")
                    example_btn.click(
                        lambda t=text: t,
                        outputs=[text_input]
                    )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📊 预测结果")
            result_output = gr.Markdown(
                label="分析结果",
                value="等待输入文本进行情感分析..."
            )
    
    gr.Markdown("---")
    
    # 批量处理区域
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📋 批量情感分析")
            
            batch_input = gr.Textbox(
                label="📝 批量输入（每行一个文本）",
                placeholder="今天天气真好\n这部电影太无聊了\n产品质量不错\n服务态度很差",
                lines=6
            )
            
            batch_btn = gr.Button("🔄 批量分析", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 📈 批量结果")
            batch_output = gr.Markdown(
                value="等待批量输入..."
            )
    
    # 技术详情折叠面板
    with gr.Accordion("🔧 技术实现详情", open=False):
        gr.Markdown("""
        ### 📋 技术栈详情
        
        **模型架构**：
        - 基础模型：chinese-roberta-wwm-ext (哈工大版本)
        - 微调任务：三分类情感分析 (正面/中性/负面)
        - 优化器：AdamW + 学习率调度
        - 损失函数：CrossEntropyLoss
        
        **数据处理**：
        - 文本长度：最大512 tokens
        - 预处理：分词 + 填充/截断
        - 后处理：Softmax概率输出
        
        **性能优化**：
        - 推理加速：torch.no_grad()
        - 批量处理：支持批量推理
        - 设备适配：自动GPU/CPU切换
        
        ### 🎯 项目亮点
        
        1. **完整的微调流程**：从数据预处理到模型训练部署
        2. **工程化实现**：错误处理、性能监控、用户友好界面
        3. **可扩展设计**：支持多分类、批量处理、模型热更新
        4. **部署优化**：适配Hugging Face Spaces环境限制
        """)
    
    # 事件绑定
    load_btn.click(
        fn=load_model_interface,
        outputs=[load_status]
    )
    
    predict_btn.click(
        fn=predict_interface,
        inputs=[text_input],
        outputs=[result_output]
    )
    
    text_input.submit(
        fn=predict_interface,
        inputs=[text_input],
        outputs=[result_output]
    )
    
    batch_btn.click(
        fn=batch_predict_interface,
        inputs=[batch_input],
        outputs=[batch_output]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch()