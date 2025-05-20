from data_processor import DataProcessor
from spam_classifier import SpamClassifier
from topic_classifier import TopicClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support,
    roc_curve,
    auc,
    classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import time
import os

def evaluate_spam_model():
    """Đánh giá chi tiết mô hình phân loại spam"""
    print("\n" + "="*50)
    print("ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI SPAM")
    print("="*50)
    
    # Tải dữ liệu
    print("Đang tải dữ liệu spam...")
    data_processor = DataProcessor('spam.csv')
    X_train, X_test, y_train, y_test = data_processor.prepare_spam_data()
    
    # Tải model đã huấn luyện
    print("Đang tải mô hình spam...")
    classifier = SpamClassifier()
    try:
        classifier.load_model()
    except FileNotFoundError:
        print("Mô hình spam chưa được huấn luyện. Đang huấn luyện mô hình mới...")
        classifier.train(X_train, y_train)
        classifier.save_model()
    
    # Đo thời gian dự đoán
    print("\nĐánh giá hiệu suất:")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    end_time = time.time()
    
    # Lấy xác suất cho đường cong ROC
    y_prob = classifier.predict_proba(X_test)[:, 1]
    
    # Tính các metric
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    
    print(f"Thời gian dự đoán: {end_time - start_time:.4f} giây cho {X_test.shape[0]} mẫu")
    
    # Kích thước model
    model_path = 'models/spam_classifier.joblib'
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"Kích thước mô hình: {model_size:.2f} MB")
    
    # In các metric
    print("\nCác chỉ số đánh giá:")
    print(f"Độ chính xác (Accuracy): {acc:.4f}")
    print(f"Độ tin cậy (Precision): {precision:.4f}")
    print(f"Độ bao phủ (Recall): {recall:.4f}")
    print(f"Điểm F1 (F1-score): {f1:.4f}")
    
    # In báo cáo phân loại chi tiết
    print("\nBáo cáo phân loại chi tiết:")
    report = classification_report(y_test, y_pred, target_names=['Không spam', 'Spam'])
    print(report)
    
    # Vẽ ma trận nhầm lẫn
    print("\nĐang vẽ ma trận nhầm lẫn...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Không spam', 'Spam'],
               yticklabels=['Không spam', 'Spam'])
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn - Mô hình Spam')
    plt.savefig('spam_confusion_matrix.png')
    plt.close()
    print("Đã lưu ma trận nhầm lẫn vào spam_confusion_matrix.png")
    
    # Vẽ đường cong ROC
    print("\nĐang vẽ đường cong ROC...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tỉ lệ dương tính giả (False Positive Rate)')
    plt.ylabel('Tỉ lệ dương tính thật (True Positive Rate)')
    plt.title('Đường cong ROC - Mô hình Spam')
    plt.legend(loc="lower right")
    plt.savefig('spam_roc_curve.png')
    plt.close()
    print("Đã lưu đường cong ROC vào spam_roc_curve.png")
    
    # So sánh với các thuật toán khác
    print("\nSo sánh với các thuật toán khác...")
    compare_algorithms(X_train, X_test, y_train, y_test, "Phân loại Spam")
    
    # Cross-validation
    print("\nĐánh giá bằng 5-fold cross-validation...")
    cross_validate(data_processor.spam_vectorizer, 'spam.csv', 'spam', 'Spam')
    
    return acc, precision, recall, f1

def evaluate_topic_model():
    """Đánh giá chi tiết mô hình phân loại chủ đề"""
    print("\n" + "="*50)
    print("ĐÁNH GIÁ MÔ HÌNH PHÂN LOẠI CHỦ ĐỀ")
    print("="*50)
    
    # Tải dữ liệu
    print("Đang tải dữ liệu chủ đề...")
    data_processor = DataProcessor()
    X_train, X_test, y_train, y_test = data_processor.prepare_topic_data()
    
    # Tải model đã huấn luyện
    print("Đang tải mô hình chủ đề...")
    classifier = TopicClassifier()
    try:
        classifier.load_model()
    except FileNotFoundError:
        print("Mô hình chủ đề chưa được huấn luyện. Đang huấn luyện mô hình mới...")
        classifier.train(X_train, y_train)
        classifier.save_model()
    
    # Đo thời gian dự đoán
    print("\nĐánh giá hiệu suất:")
    start_time = time.time()
    y_pred = classifier.predict(X_test)
    end_time = time.time()
    
    # Tính các metric
    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"Thời gian dự đoán: {end_time - start_time:.4f} giây cho {X_test.shape[0]} mẫu")
    
    # Kích thước model
    model_path = 'models/topic_classifier.joblib'
    if os.path.exists(model_path):
        model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        print(f"Kích thước mô hình: {model_size:.2f} MB")
    
    # In các metric
    print("\nCác chỉ số đánh giá:")
    print(f"Độ chính xác (Accuracy): {acc:.4f}")
    print(f"Độ tin cậy (Precision weighted): {precision:.4f}")
    print(f"Độ bao phủ (Recall weighted): {recall:.4f}")
    print(f"Điểm F1 (F1-score weighted): {f1:.4f}")
    
    # In báo cáo phân loại chi tiết
    print("\nBáo cáo phân loại chi tiết:")
    report = classification_report(y_test, y_pred, target_names=classifier.topics)
    print(report)
    
    # Vẽ ma trận nhầm lẫn
    print("\nĐang vẽ ma trận nhầm lẫn...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=classifier.topics,
               yticklabels=classifier.topics)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn - Mô hình Chủ đề')
    plt.tight_layout()
    plt.savefig('topic_confusion_matrix.png')
    plt.close()
    print("Đã lưu ma trận nhầm lẫn vào topic_confusion_matrix.png")
    
    # So sánh với các thuật toán khác
    print("\nSo sánh với các thuật toán khác...")
    compare_algorithms(X_train, X_test, y_train, y_test, "Phân loại Chủ đề")
    
    # Cross-validation
    print("\nĐánh giá bằng 5-fold cross-validation...")
    cross_validate(data_processor.topic_vectorizer, 'bbc-text.csv', 'category', 'Chủ đề', is_topic=True)
    
    return acc, precision, recall, f1

def compare_algorithms(X_train, X_test, y_train, y_test, task_name):
    """So sánh hiệu suất của các thuật toán khác nhau"""
    results = {}
    
    # Naive Bayes (hiện tại)
    print("Đánh giá với Naive Bayes...")
    nb = MultinomialNB()
    start_time = time.time()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    end_time = time.time()
    results['Naive Bayes'] = {
        'accuracy': accuracy_score(y_test, y_pred_nb),
        'time': end_time - start_time
    }
    
    # Logistic Regression
    print("Đánh giá với Logistic Regression...")
    lr = LogisticRegression(max_iter=1000)
    start_time = time.time()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    end_time = time.time()
    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'time': end_time - start_time
    }
    
    # Random Forest
    print("Đánh giá với Random Forest...")
    rf = RandomForestClassifier(n_estimators=100)
    start_time = time.time()
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    end_time = time.time()
    results['Random Forest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'time': end_time - start_time
    }
    
    # Linear SVM
    print("Đánh giá với Linear SVM...")
    svm = LinearSVC(max_iter=1000)
    start_time = time.time()
    svm.fit(X_train, y_train)
    y_pred_svm = svm.predict(X_test)
    end_time = time.time()
    results['Linear SVM'] = {
        'accuracy': accuracy_score(y_test, y_pred_svm),
        'time': end_time - start_time
    }
    
    # Hiển thị kết quả
    print("\nKết quả so sánh các thuật toán:")
    for name, metrics in results.items():
        print(f"{name}: độ chính xác = {metrics['accuracy']:.4f}, thời gian = {metrics['time']:.4f}s")
    
    # Vẽ biểu đồ so sánh
    algorithms = list(results.keys())
    accuracies = [results[algo]['accuracy'] for algo in algorithms]
    times = [results[algo]['time'] for algo in algorithms]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Biểu đồ độ chính xác
    ax1.bar(algorithms, accuracies, color='skyblue')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Độ chính xác')
    ax1.set_title(f'So sánh độ chính xác các thuật toán cho {task_name}')
    
    # Biểu đồ thời gian
    ax2.bar(algorithms, times, color='salmon')
    ax2.set_ylabel('Thời gian huấn luyện và dự đoán (giây)')
    ax2.set_title(f'So sánh thời gian các thuật toán cho {task_name}')
    
    plt.tight_layout()
    plt.savefig(f'{task_name.lower().replace(" ", "_")}_comparison.png')
    plt.close()
    print(f"Đã lưu biểu đồ so sánh vào {task_name.lower().replace(' ', '_')}_comparison.png")
    
    return results

def cross_validate(vectorizer, data_file, label_column, task_name, is_topic=False):
    """Đánh giá mô hình bằng cross-validation"""
    try:
        if is_topic:
            # Tải dữ liệu chủ đề
            df = pd.read_csv(data_file)
            # Ánh xạ các danh mục thành nhãn số
            category_map = {
                'tech': 0,
                'business': 1,
                'entertainment': 2,
                'sport': 3,
                'politics': 4
            }
            df['category_id'] = df['category'].map(category_map)
            texts = df['text'].values
            labels = df['category_id'].values
        else:
            # Tải dữ liệu spam
            df = pd.read_csv(data_file)
            df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
            texts = df['Message'].values
            labels = df['spam'].values
        
        # Chuyển đổi văn bản
        X = vectorizer.transform(texts)
        
        # K-fold cross validation
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = MultinomialNB()
        
        # Đánh giá bằng cross-validation
        cv_scores = cross_val_score(model, X, labels, cv=kf, scoring='accuracy')
        
        print(f"Kết quả 5-fold Cross Validation cho mô hình {task_name}:")
        print(f"Các điểm số: {[f'{score:.4f}' for score in cv_scores]}")
        print(f"Điểm trung bình: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Vẽ biểu đồ
        plt.figure(figsize=(8, 6))
        plt.boxplot(cv_scores, vert=False, patch_artist=True)
        plt.axvline(x=cv_scores.mean(), color='red', linestyle='--', label=f'Trung bình ({cv_scores.mean():.4f})')
        plt.xlabel('Độ chính xác')
        plt.title(f'5-fold Cross Validation cho mô hình {task_name}')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(f'{task_name.lower().replace(" ", "_")}_cross_validation.png')
        plt.close()
        print(f"Đã lưu biểu đồ cross-validation vào {task_name.lower().replace(' ', '_')}_cross_validation.png")
        
    except Exception as e:
        print(f"Lỗi khi thực hiện cross-validation: {str(e)}")

def evaluate_with_examples():
    """Đánh giá mô hình với một số ví dụ cụ thể"""
    print("\n" + "="*50)
    print("ĐÁNH GIÁ BẰNG CÁC VÍ DỤ CỤ THỂ")
    print("="*50)
    
    # Tải models
    data_processor = DataProcessor('spam.csv')
    spam_classifier = SpamClassifier()
    topic_classifier = TopicClassifier()
    
    try:
        spam_classifier.load_model()
        topic_classifier.load_model()
        data_processor.load_spam_vectorizer()
        data_processor.load_topic_vectorizer()
    except FileNotFoundError as e:
        print(f"Lỗi: {str(e)}")
        print("Vui lòng huấn luyện models trước khi đánh giá.")
        return
    
    # Các ví dụ email
    examples = [
        {
            "text": "Chúc mừng! Bạn đã trúng thưởng 10 triệu đồng. Gửi thông tin cá nhân ngay để nhận giải.",
            "expected": {"is_spam": True, "topic": None},
            "description": "Email lừa đảo trúng thưởng"
        },
        {
            "text": "Cuộc họp báo cáo quý 2 sẽ diễn ra vào 14h chiều nay tại phòng họp A. Mong các bạn tham dự đầy đủ.",
            "expected": {"is_spam": False, "topic": "business"},
            "description": "Email công việc về cuộc họp"
        },
        {
            "text": "Arsenal đã giành chiến thắng 3-0 trước Chelsea trong trận derby London tối qua. Hãy xem highlight trên trang web của chúng tôi.",
            "expected": {"is_spam": False, "topic": "sport"},
            "description": "Tin tức thể thao về bóng đá"
        },
        {
            "text": "Apple vừa ra mắt iPhone 15 với nhiều tính năng đột phá. Sản phẩm sẽ được bán ra từ tháng 9.",
            "expected": {"is_spam": False, "topic": "tech"},
            "description": "Tin tức công nghệ về iPhone"
        },
        {
            "text": "Thủ tướng đã có cuộc họp với các bộ trưởng về chính sách kinh tế mới. Dự kiến sẽ có những thay đổi lớn.",
            "expected": {"is_spam": False, "topic": "politics"},
            "description": "Tin tức chính trị về chính phủ"
        },
        {
            "text": "Ca sĩ Taylor Swift sẽ tổ chức concert tại Việt Nam vào tháng 12 năm nay. Vé sẽ được mở bán vào tuần sau.",
            "expected": {"is_spam": False, "topic": "entertainment"},
            "description": "Tin tức giải trí về concert"
        }
    ]
    
    results = []
    
    # Đánh giá từng ví dụ
    for i, example in enumerate(examples):
        print(f"\nVí dụ {i+1}: {example['description']}")
        print(f"Nội dung: {example['text']}")
        
        # Dự đoán spam
        X_spam = data_processor.transform_new_data([example['text']], vectorizer_type='spam')
        spam_pred = spam_classifier.predict(X_spam)[0]
        spam_prob = spam_classifier.predict_proba(X_spam)[0]
        
        is_spam = spam_pred == 1
        print(f"Kết quả phân loại spam: {'Spam' if is_spam else 'Không phải Spam'} (Xác suất: {spam_prob[1]:.4f})")
        
        # Nếu không phải spam, dự đoán chủ đề
        topic_pred = None
        topic_name = None
        topic_prob = None
        
        if not is_spam:
            X_topic = data_processor.transform_new_data([example['text']], vectorizer_type='topic')
            topic_pred = topic_classifier.predict(X_topic)[0]
            topic_prob = topic_classifier.predict_proba(X_topic)[0][topic_pred]
            topic_name = topic_classifier.get_topic_name(topic_pred)
            print(f"Kết quả phân loại chủ đề: {topic_name} (Xác suất: {topic_prob:.4f})")
        
        # Kiểm tra với kết quả mong đợi
        spam_correct = is_spam == example['expected']['is_spam']
        
        topic_correct = None
        if not is_spam and example['expected']['is_spam'] == False:
            expected_topic = example['expected']['topic']
            topic_correct = topic_name == expected_topic
        
        print(f"Kết luận: Phân loại spam {'đúng' if spam_correct else 'sai'}")
        if topic_correct is not None:
            print(f"Kết luận: Phân loại chủ đề {'đúng' if topic_correct else 'sai'}")
        
        # Lưu kết quả
        results.append({
            'text': example['text'],
            'description': example['description'],
            'expected_spam': example['expected']['is_spam'],
            'predicted_spam': is_spam,
            'spam_correct': spam_correct,
            'spam_confidence': spam_prob[1] if is_spam else 1 - spam_prob[1],
            'expected_topic': example['expected']['topic'],
            'predicted_topic': topic_name,
            'topic_correct': topic_correct,
            'topic_confidence': topic_prob if topic_prob else None
        })
    
    # Tính tỉ lệ đúng
    spam_accuracy = sum(1 for r in results if r['spam_correct']) / len(results)
    topic_results = [r for r in results if r['topic_correct'] is not None]
    topic_accuracy = sum(1 for r in topic_results if r['topic_correct']) / len(topic_results) if topic_results else 0
    
    print("\nKết quả tổng hợp:")
    print(f"Độ chính xác phân loại spam trên ví dụ: {spam_accuracy:.4f}")
    print(f"Độ chính xác phân loại chủ đề trên ví dụ: {topic_accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    # Đảm bảo thư mục models tồn tại
    os.makedirs('models', exist_ok=True)
    
    # Đánh giá mô hình spam
    spam_metrics = evaluate_spam_model()
    
    # Đánh giá mô hình chủ đề
    topic_metrics = evaluate_topic_model()
    
    # Đánh giá với các ví dụ cụ thể
    example_results = evaluate_with_examples()
    
    print("\n" + "="*50)
    print("HOÀN TẤT ĐÁNH GIÁ")
    print("="*50)
    print("Đã lưu các biểu đồ đánh giá vào thư mục hiện tại.") 