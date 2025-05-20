from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os

class SpamClassifier:
    def __init__(self):
        self.model = MultinomialNB()
        
    def train(self, X_train, y_train):
        """Huấn luyện mô hình"""
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        """Đánh giá hiệu suất mô hình"""
        y_pred = self.model.predict(X_test)
        return classification_report(y_test, y_pred)
    
    def predict(self, X):
        """Dự đoán trên dữ liệu mới"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Lấy xác suất dự đoán cho mỗi lớp"""
        return self.model.predict_proba(X)
        
    def save_model(self, model_path='models'):
        """Lưu mô hình đã huấn luyện vào ổ đĩa"""
        # Tạo thư mục models nếu chưa tồn tại
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # Lưu mô hình
        model_file = os.path.join(model_path, 'spam_classifier.joblib')
        joblib.dump(self.model, model_file)
        print(f"Đã lưu mô hình vào {model_file}")
        
    def load_model(self, model_path='models/spam_classifier.joblib'):
        """Tải mô hình đã huấn luyện từ ổ đĩa"""
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Đã tải mô hình từ {model_path}")
        else:
            raise FileNotFoundError(f"Không tìm thấy mô hình tại {model_path}") 