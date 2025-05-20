from data_processor import DataProcessor
from spam_classifier import SpamClassifier

def train_model():
    """Huấn luyện mô hình phân loại spam và lưu lại"""
    print("Bắt đầu quá trình huấn luyện mô hình...")
    
    # Khởi tạo bộ xử lý dữ liệu
    data_processor = DataProcessor('spam.csv')
    
    # Chuẩn bị dữ liệu
    print("Đang chuẩn bị dữ liệu...")
    X_train, X_test, y_train, y_test = data_processor.prepare_spam_data()
    
    # Khởi tạo và huấn luyện bộ phân loại
    print("Đang huấn luyện mô hình...")
    classifier = SpamClassifier()
    classifier.train(X_train, y_train)
    
    # Đánh giá mô hình
    print("\nĐánh giá mô hình:")
    print(classifier.evaluate(X_test, y_test))
    
    # Lưu mô hình
    classifier.save_model()
    print("\nHuấn luyện mô hình hoàn tất và đã lưu thành công!")

if __name__ == "__main__":
    train_model() 