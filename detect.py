#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
社交媒体虚假信息检测系统 - 命令行检测脚本
使用方法: python detect.py --text "待检测的文本内容"
"""

import re
import argparse
import warnings
warnings.filterwarnings('ignore')  # 忽略警告

# 禁用 transformers 的特定警告
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def load_models():
    """加载模型文件"""
    global xgb_model, tokenizer, bert_model
    
    try:
        import joblib
        import torch
        from transformers import BertTokenizer, BertModel
        import xgboost as xgb
    except ImportError as e:
        print(f"错误：缺少必要的Python库，请运行: pip install joblib torch transformers xgboost")
        print(f"缺失的库: {e}")
        return False
    
    try:
        # 加载XGBoost模型
        xgb_model = joblib.load('models/fake_news_model.pkl')
        print("✓ XGBoost模型加载成功")
    except FileNotFoundError:
        print("错误：未找到 models/fake_news_model.pkl")
        print("请先在 Jupyter Notebook 中运行 experiment.ipynb 生成模型文件")
        return False
    except Exception as e:
        print(f"加载XGBoost模型失败: {e}")
        return False
    
    try:
        # 加载BERT模型（静默加载，不输出警告）
        import sys
        from io import StringIO
        
        # 临时重定向stdout，屏蔽BERT加载时的输出
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        model_path = r'C:\Users\86187\bert-base-chinese'
        tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=True)
        bert_model = BertModel.from_pretrained(model_path, local_files_only=True)
        bert_model.eval()
        
        # 恢复输出
        sys.stdout = old_stdout
        print("✓ BERT模型加载成功")
    except FileNotFoundError:
        print("警告：未找到本地BERT模型，将使用在线模式（需要网络连接）")
        try:
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            bert_model = BertModel.from_pretrained('bert-base-chinese')
            bert_model.eval()
            print("✓ BERT模型在线加载成功")
        except Exception as e:
            print(f"加载BERT模型失败: {e}")
            return False
    except Exception as e:
        print(f"加载BERT模型失败: {e}")
        return False
    
    return True

def clean_text(text):
    """文本清洗"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    return text.strip()

def predict(text, category_index=0):
    """
    预测单条文本
    category_index: 0-9 科技、军事、政治、社会生活、文体娱乐、教育考试、财经商业、医药健康、灾难事故、无法确定
    """
    import numpy as np
    import torch
    
    # 清洗文本
    clean = clean_text(text)
    print(f"清洗后文本: {clean[:80]}...")
    
    # 提取BERT特征
    inputs = tokenizer(clean, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        bert_feat = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    # 生成类别特征（独热编码）
    cat_feat = np.zeros(10)
    cat_feat[category_index] = 1
    
    # 拼接特征
    features = np.concatenate([bert_feat, cat_feat]).reshape(1, -1)
    
    # 预测
    prob = xgb_model.predict_proba(features)[0]
    pred = 1 if prob[1] > 0.5 else 0
    
    return {
        'prediction': '虚假' if pred == 1 else '真实',
        'confidence': float(prob[1]),
        'probability_fake': float(prob[1]),
        'probability_real': float(prob[0])
    }

def main():
    parser = argparse.ArgumentParser(
        description='社交媒体虚假信息检测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python detect.py --text "今天天气真好"
  python detect.py --text "震惊！发生大事了！" --category 0
        '''
    )
    parser.add_argument('--text', '-t', type=str, required=True, help='待检测的文本内容')
    parser.add_argument('--category', '-c', type=int, default=0,
                        help='类别索引: 0科技,1军事,2政治,3社会生活,4文体娱乐,5教育考试,6财经商业,7医药健康,8灾难事故,9无法确定 (默认:0)')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("社交媒体虚假信息检测系统")
    print("=" * 50)
    print(f"输入文本: {args.text[:100]}")
    print(f"类别索引: {args.category}")
    print("-" * 50)
    
    # 加载模型
    if not load_models():
        print("\n模型加载失败，请检查配置")
        return
    
    # 预测
    print("\n正在检测...")
    result = predict(args.text, args.category)
    
    # 输出结果
    print("\n" + "=" * 50)
    print("检测结果")
    print("=" * 50)
    print(f"判定结果: {result['prediction']}")
    print(f"虚假概率: {result['probability_fake']:.4f}")
    print(f"真实概率: {result['probability_real']:.4f}")
    print("=" * 50)

if __name__ == '__main__':
    main()
