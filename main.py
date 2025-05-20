from data_processor import DataProcessor
from spam_classifier import SpamClassifier
from topic_classifier import TopicClassifier
import os
import re
import joblib

def load_models():
    """Tải cả hai mô hình phân loại spam và chủ đề"""
    # Tải bộ phân loại spam
    spam_classifier = SpamClassifier()
    try:
        spam_classifier.load_model()
        print("Đã tải mô hình spam thành công!")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy mô hình spam!")
        print("Vui lòng chạy 'python train_model.py' trước.")
        exit(1)
        
    # Tải bộ phân loại chủ đề
    topic_classifier = TopicClassifier()
    try:
        topic_classifier.load_model()
        print("Đã tải mô hình chủ đề thành công!")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy mô hình chủ đề!")
        print("Vui lòng chạy 'python train_topic_model.py' trước.")
        exit(1)
        
    return spam_classifier, topic_classifier

def preprocess_email(email):
    """Tiền xử lý văn bản email để xử lý ký tự đặc biệt và văn bản dài"""
    try:
        # Chuyển thành chữ thường
        email = email.lower()
        
        # Thay thế dòng mới bằng khoảng trắng
        email = email.replace('\n', ' ')
        
        # Loại bỏ ký tự đặc biệt nhưng giữ lại dấu câu và ký hiệu phổ biến
        email = re.sub(r'[^a-z0-9\s.,!?@#$%^&*()_+\-=\[\]{};\'":\\|<>/~`]', ' ', email)
        
        # Loại bỏ khoảng trắng thừa
        email = ' '.join(email.split())
        
        return email
    except Exception as e:
        print(f"Lỗi khi xử lý văn bản: {str(e)}")
        return email

def check_email(spam_classifier, topic_classifier, data_processor, email):
    """Kiểm tra xem email có phải là spam không và phân loại chủ đề nếu không phải spam"""
    try:
        # Tiền xử lý email
        processed_email = preprocess_email(email)
        
        # Chuyển đổi cho phân loại spam
        X_spam = data_processor.transform_new_data([processed_email], vectorizer_type='spam')
        
        # Kiểm tra nếu là spam
        spam_prediction = spam_classifier.predict(X_spam)[0]
        spam_probability = spam_classifier.predict_proba(X_spam)[0]
        
        # Hiển thị kết quả
        print("\nKết quả kiểm tra:")
        print("="*80)
        print("Nội dung email:")
        print("-"*80)
        print(email)
        print("-"*80)
        print(f"Phân loại: {'Spam' if spam_prediction == 1 else 'Không phải Spam'}")
        print(f"Độ tin cậy: {spam_probability[1]*100:.2f}% (spam)")
        
        # Nếu không phải spam, phân loại chủ đề
        if spam_prediction == 0:
            try:
                # Chuyển đổi cho phân loại chủ đề
                X_topic = data_processor.transform_new_data([processed_email], vectorizer_type='topic')
                
                # Lấy dự đoán chủ đề
                topic_prediction = topic_classifier.predict(X_topic)[0]
                topic_probability = topic_classifier.predict_proba(X_topic)[0]
                
                print(f"\nChủ đề: {topic_classifier.get_topic_name(topic_prediction)}")
                print(f"Độ tin cậy: {topic_probability[topic_prediction]*100:.2f}%")
            except Exception as e:
                print(f"\nLỗi khi phân loại chủ đề: {str(e)}")
                print("Vui lòng chạy lại 'python train_topic_model.py' để train model chủ đề.")
        
        print("="*80)
        
    except Exception as e:
        print(f"\nLỗi khi xử lý email: {str(e)}")
        print("Vui lòng thử lại với email khác.")

def get_multiline_input():
    """Nhận đầu vào nhiều dòng từ người dùng"""
    print("\nNhập nội dung email cần kiểm tra (nhấn Enter 2 lần để kết thúc):")
    lines = []
    while True:
        try:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nĐã hủy nhập. Vui lòng thử lại.")
            return None
    
    if not lines:
        return None
    
    # Loại bỏ dòng trống cuối cùng
    if lines[-1] == "":
        lines.pop()
    
    return "\n".join(lines)

def main():
    # Tải cả hai mô hình
    spam_classifier, topic_classifier = load_models()
    
    # Khởi tạo bộ xử lý dữ liệu cho dự đoán
    data_processor = DataProcessor('spam.csv')
    
    # Đảm bảo vectorizer được tải
    try:
        data_processor.load_spam_vectorizer()
        data_processor.load_topic_vectorizer()
    except FileNotFoundError as e:
        print(f"Lỗi: {str(e)}")
        print("Vui lòng chạy cả 'python train_model.py' và 'python train_topic_model.py' trước.")
        exit(1)
    
    try:
        while True:
            print("\n=== Kiểm tra Email ===")
            print("1. Kiểm tra email")
            print("2. Thoát")
            
            choice = input("\nChọn chức năng (1-2): ")
            
            if choice == '1':
                email = get_multiline_input()
                
                if email and email.strip():
                    check_email(spam_classifier, topic_classifier, data_processor, email)
                else:
                    print("Email không được để trống!")
                    
            elif choice == '2':
                print("\nCảm ơn bạn đã sử dụng chương trình!")
                break
                
            else:
                print("\nLựa chọn không hợp lệ. Vui lòng chọn lại!")
                
    except Exception as e:
        print(f"\nLỗi không mong muốn: {str(e)}")
        print("Vui lòng thử lại sau.")

if __name__ == "__main__":
    main() 