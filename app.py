import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_path = "chinese_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=False)

def predict(text):
    return classifier(text)[0]['label']

demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="中文情感分类 Demo",
    description="输入中文文本，预测情感标签（正向 / 中性 / 负向）"
)

demo.launch()
