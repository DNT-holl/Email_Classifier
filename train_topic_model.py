import pandas as pd
from sklearn.model_selection import train_test_split
from topic_classifier import TopicClassifier
from data_processor import DataProcessor

def train_topic_model():
    """Huấn luyện mô hình phân loại chủ đề và lưu lại"""
    print("Bắt đầu quá trình huấn luyện mô hình chủ đề...")
    
    # Khởi tạo bộ xử lý dữ liệu
    data_processor = DataProcessor()
    
    try:
        # Chuẩn bị dữ liệu
        print("Đang chuẩn bị dữ liệu BBC News...")
        X_train, X_test, y_train, y_test = data_processor.prepare_topic_data()
        
        # Khởi tạo và huấn luyện bộ phân loại
        print("Đang huấn luyện mô hình chủ đề...")
        classifier = TopicClassifier()
        classifier.train(X_train, y_train)
        
        # Đánh giá mô hình
        print("\nĐánh giá mô hình:")
        print(classifier.evaluate(X_test, y_test))
        
        # Lưu mô hình
        classifier.save_model()
        print("\nHuấn luyện mô hình chủ đề hoàn tất và đã lưu thành công!")
        
    except ValueError as e:
        print(f"Lỗi: {str(e)}")
    except Exception as e:
        print(f"Lỗi không mong muốn: {str(e)}")

if __name__ == "__main__":
    train_topic_model() 