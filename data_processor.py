import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import os
import re
import unicodedata
from googletrans import Translator

class DataProcessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.spam_vectorizer = CountVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b\w+\b'  # Cho phép các ký tự Unicode
        )
        self.topic_vectorizer = CountVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b\w+\b'
        )
        self.translator = Translator()
        
    def translate_to_english(self, text):
        """Dịch văn bản sang tiếng Anh"""
        try:
            # Kiểm tra xem văn bản có phải tiếng Việt không
            if any('\u00C0' <= char <= '\u1EF9' for char in text):
                # Dịch sang tiếng Anh
                translated = self.translator.translate(text, dest='en')
                return translated.text
            return text
        except Exception as e:
            print(f"Lỗi khi dịch: {str(e)}")
            return text
        
    def normalize_text(self, text):
        """Chuẩn hóa văn bản tiếng Việt"""
        # Dịch sang tiếng Anh nếu là tiếng Việt
        text = self.translate_to_english(text)
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Chuẩn hóa Unicode
        text = unicodedata.normalize('NFC', text)
        
        # Loại bỏ URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Loại bỏ email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Loại bỏ số
        text = re.sub(r'\d+', '', text)
        
        # Loại bỏ ký tự đặc biệt nhưng giữ lại dấu tiếng Việt
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
        
        # Loại bỏ khoảng trắng thừa
        text = ' '.join(text.split())
        
        return text
        
    def load_spam_data(self):
        """Tải và tiền xử lý dữ liệu spam"""
        if not self.data_path:
            raise ValueError("Đường dẫn dữ liệu không được thiết lập")
            
        df = pd.read_csv(self.data_path)
        # Chuyển đổi thành nhị phân (spam = 1, ham = 0)
        df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
        
        # Tiền xử lý văn bản
        print("Đang tiền xử lý dữ liệu spam...")
        df['processed_message'] = df['Message'].apply(self.normalize_text)
        
        return df
    
    def load_topic_data(self, data_path='bbc-text.csv'):
        """Tải và tiền xử lý dữ liệu BBC News"""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Không tìm thấy dữ liệu BBC News tại {data_path}")
            
        # Tải dữ liệu BBC News
        df = pd.read_csv(data_path)
        
        # Ánh xạ các danh mục thành nhãn số
        category_map = {
            'tech': 0,
            'business': 1,
            'entertainment': 2,
            'sport': 3,
            'politics': 4
        }
        
        # Chuyển đổi danh mục thành nhãn số
        df['category_id'] = df['category'].map(category_map)
        
        # Tiền xử lý văn bản
        print("Đang tiền xử lý dữ liệu BBC News...")
        df['processed_text'] = df['text'].apply(self.normalize_text)
        
        return df['processed_text'].values, df['category_id'].values
    
    def prepare_spam_data(self, test_size=0.2):
        """Chuẩn bị dữ liệu spam cho huấn luyện và kiểm thử"""
        df = self.load_spam_data()
        
        # Chia dữ liệu thành tập huấn luyện và kiểm thử
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_message'], 
            df['spam'], 
            test_size=test_size, 
            random_state=42,
            stratify=df['spam']
        )
        
        # Chuyển đổi dữ liệu văn bản thành đặc trưng số
        X_train_cv = self.spam_vectorizer.fit_transform(X_train)
        X_test_cv = self.spam_vectorizer.transform(X_test)
        
        # Lưu vectorizer
        self.save_spam_vectorizer()
        
        return X_train_cv, X_test_cv, y_train, y_test
    
    def prepare_topic_data(self, test_size=0.2):
        """Chuẩn bị dữ liệu BBC News cho huấn luyện và kiểm thử"""
        texts, labels = self.load_topic_data()
        
        if len(texts) == 0:
            raise ValueError("Không tìm thấy dữ liệu chủ đề!")
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Chuyển đổi dữ liệu văn bản
        X_train_cv = self.topic_vectorizer.fit_transform(X_train)
        X_test_cv = self.topic_vectorizer.transform(X_test)
        
        # Lưu vectorizer
        self.save_topic_vectorizer()
        
        return X_train_cv, X_test_cv, y_train, y_test
    
    def transform_new_data(self, new_emails, vectorizer_type='spam'):
        """Chuyển đổi email mới cho việc dự đoán"""
        vectorizer = self.spam_vectorizer if vectorizer_type == 'spam' else self.topic_vectorizer
        
        # Tải vectorizer nếu chưa được huấn luyện
        if not hasattr(vectorizer, 'vocabulary_'):
            if vectorizer_type == 'spam':
                self.load_spam_vectorizer()
            else:
                self.load_topic_vectorizer()
            
        # Tiền xử lý văn bản mới
        processed_emails = [self.normalize_text(email) for email in new_emails]
        return vectorizer.transform(processed_emails)
        
    def save_spam_vectorizer(self, vectorizer_path='models'):
        """Lưu vectorizer spam đã huấn luyện"""
        if not os.path.exists(vectorizer_path):
            os.makedirs(vectorizer_path)
        vectorizer_file = os.path.join(vectorizer_path, 'vectorizer.joblib')
        joblib.dump(self.spam_vectorizer, vectorizer_file)
        
    def save_topic_vectorizer(self, vectorizer_path='models'):
        """Lưu vectorizer chủ đề đã huấn luyện"""
        if not os.path.exists(vectorizer_path):
            os.makedirs(vectorizer_path)
        vectorizer_file = os.path.join(vectorizer_path, 'topic_vectorizer.joblib')
        joblib.dump(self.topic_vectorizer, vectorizer_file)
        
    def load_spam_vectorizer(self, vectorizer_path='models/vectorizer.joblib'):
        """Tải vectorizer spam đã huấn luyện"""
        if os.path.exists(vectorizer_path):
            self.spam_vectorizer = joblib.load(vectorizer_path)
        else:
            raise FileNotFoundError(f"Không tìm thấy vectorizer spam tại {vectorizer_path}")
            
    def load_topic_vectorizer(self, vectorizer_path='models/topic_vectorizer.joblib'):
        """Tải vectorizer chủ đề đã huấn luyện"""
        if os.path.exists(vectorizer_path):
            self.topic_vectorizer = joblib.load(vectorizer_path)
        else:
            raise FileNotFoundError(f"Không tìm thấy vectorizer chủ đề tại {vectorizer_path}") 