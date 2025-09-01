import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# 加载模型
model_path = "./chinese_sentiment_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

id2label = {0: "negative", 1: "neutral", 2: "positive"}

def predict(text):
    res = classifier(text)
    label_idx = int(res[0]['label'].split("_")[-1])
    return {"label": id2label[label_idx], "score": float(res[0]['score'])}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="输入中文文本"),
    outputs=gr.Label(num_top_classes=1)
)

iface.launch()

