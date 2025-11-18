# -*- coding: utf-8 -*-
import sys
import io
import os

# Thiết lập encoding UTF-8 cho output
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from flask import Flask, render_template, request, jsonify
from data_processor import DataProcessor
from spam_classifier import SpamClassifier
from topic_classifier import TopicClassifier
import re

app = Flask(__name__)

# Tải mô hình khi khởi động
print("Đang tải mô hình...")
spam_classifier = SpamClassifier()
topic_classifier = TopicClassifier()
data_processor = DataProcessor('spam.csv')

try:
    spam_classifier.load_model()
    topic_classifier.load_model()
    data_processor.load_spam_vectorizer()
    data_processor.load_topic_vectorizer()
    print("✓ Đã tải tất cả mô hình thành công!")
except Exception as e:
    print(f"✗ Lỗi khi tải mô hình: {e}")


def preprocess_email(email):
    """Tiền xử lý văn bản email"""
    try:
        email = email.lower()
        email = email.replace('\n', ' ')
        email = re.sub(r'[^a-z0-9\s.,!?@#$%^&*()_+\-=\[\]{};\'":\\|<>/~`]', ' ', email)
        email = ' '.join(email.split())
        return email
    except Exception as e:
        return email


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        email_text = data.get('email', '')

        if not email_text.strip():
            return jsonify({'error': 'Email không được để trống'}), 400

        # Tiền xử lý email
        #processed_email = preprocess_email(email_text)

        # Phân loại spam
        X_spam = data_processor.transform_new_data([email_text], vectorizer_type='spam')
        spam_prediction = spam_classifier.predict(X_spam)[0]
        spam_probability = spam_classifier.predict_proba(X_spam)[0]

        result = {
            'is_spam': int(spam_prediction),
            'spam_confidence': float(spam_probability[1] * 100),
            'classification': 'Spam' if spam_prediction == 1 else 'Không phải Spam'
        }

        # Nếu không phải spam, phân loại chủ đề
        if spam_prediction == 0:
            try:
                X_topic = data_processor.transform_new_data([processed_email], vectorizer_type='topic')
                topic_prediction = topic_classifier.predict(X_topic)[0]
                topic_probability = topic_classifier.predict_proba(X_topic)[0]

                topic_names = {
                    0: 'Công nghệ',
                    1: 'Kinh doanh',
                    2: 'Giải trí',
                    3: 'Thể thao',
                    4: 'Chính trị'
                }

                result['topic'] = topic_names.get(topic_prediction, 'Không xác định')
                result['topic_confidence'] = float(topic_probability[topic_prediction] * 100)
                result['all_topics'] = {
                    topic_names[i]: float(topic_probability[i] * 100)
                    for i in range(len(topic_probability))
                }
            except Exception as e:
                result['topic_error'] = str(e)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Ứng dụng Web Phân loại Email")
    print("=" * 60)
    print("📧 Truy cập: http://localhost:5000")
    print("⏹  Dừng: Nhấn Ctrl+C")
    print("=" * 60 + "\n")
    app.run(debug=False, host='0.0.0.0', port=5000)

