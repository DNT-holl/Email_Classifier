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
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import MultinomialNB
import time
import os

def plot_learning_curve(estimator, X_train, y_train, X_test, y_test, title):
    """Vẽ đường cong học tập để phát hiện overfitting hoặc underfitting"""
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator, X_train, y_train, train_sizes=train_sizes, cv=5, scoring='accuracy')
    
    # Tính điểm trung bình và độ lệch chuẩn
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    validation_mean = np.mean(validation_scores, axis=1)
    validation_std = np.std(validation_scores, axis=1)
    
    # Vẽ đường cong học tập
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Số lượng mẫu huấn luyện")
    plt.ylabel("Độ chính xác")
    plt.grid()
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, validation_mean - validation_std, validation_mean + validation_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Điểm huấn luyện")
    plt.plot(train_sizes, validation_mean, 'o-', color="g", label="Điểm xác thực chéo")
    
    # Vẽ điểm cho tập test nếu có
    if X_test is not None and y_test is not None:
        # Huấn luyện lại mô hình trên toàn bộ tập train
        estimator.fit(X_train, y_train)
        test_score = estimator.score(X_test, y_test)
        plt.axhline(y=test_score, color='b', linestyle='--', 
                    label=f'Điểm trên tập test ({test_score:.4f})')
    
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}_learning_curve.png')
    plt.close()
    print(f"Đã lưu đường cong học tập vào {title.lower().replace(' ', '_')}_learning_curve.png")
    
    # Phân tích dấu hiệu overfitting/underfitting
    gap = train_mean[-1] - validation_mean[-1]
    
    if train_mean[-1] < 0.6:
        status = "UNDERFITTING: Mô hình có vẻ quá đơn giản, không học được từ dữ liệu."
    elif gap > 0.1:
        status = f"OVERFITTING: Mô hình hoạt động tốt trên tập train nhưng kém hơn trên tập validation (chênh lệch {gap:.4f})."
    else:
        status = f"PHÙ HỢP: Mô hình có vẻ cân bằng tốt (chênh lệch {gap:.4f})."
    
    print(f"Phân tích đường cong học tập: {status}")
    
    return {
        'train_score': train_mean[-1],
        'validation_score': validation_mean[-1],
        'gap': gap,
        'status': status
    }

def check_overfitting(model, X_train, y_train, X_test, y_test, title):
    """Kiểm tra overfitting bằng cách so sánh hiệu suất trên tập train và test"""
    # Huấn luyện mô hình
    if not hasattr(model, 'fit'):
        # Nếu model là đối tượng classifier (SpamClassifier/TopicClassifier)
        model.model.fit(X_train, y_train)
        train_accuracy = model.model.score(X_train, y_train)
        test_accuracy = model.model.score(X_test, y_test)
    else:
        # Nếu model là scikit-learn model trực tiếp
        model.fit(X_train, y_train)
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
    
    # Tính chênh lệch
    diff = train_accuracy - test_accuracy
    
    print(f"\nKiểm tra Overfitting cho {title}:")
    print(f"Độ chính xác trên tập train: {train_accuracy:.4f}")
    print(f"Độ chính xác trên tập test: {test_accuracy:.4f}")
    print(f"Chênh lệch: {diff:.4f}")
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Tập train', 'Tập test'], [train_accuracy, test_accuracy], color=['blue', 'orange'])
    
    # Thêm giá trị lên đỉnh cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.4f}', 
                ha='center', va='bottom', fontsize=12)
    
    plt.ylim(0, 1.1)
    plt.title(f'So sánh hiệu suất trên tập train và test - {title}')
    plt.ylabel('Độ chính xác')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f'{title.lower().replace(" ", "_")}_train_vs_test.png')
    plt.close()
    print(f"Đã lưu biểu đồ so sánh vào {title.lower().replace(' ', '_')}_train_vs_test.png")
    
    # Đánh giá overfitting
    if diff > 0.1:
        status = f"OVERFITTING: Chênh lệch lớn ({diff:.4f}) giữa hiệu suất trên tập train và test."
    elif diff < 0:
        status = f"BẤT THƯỜNG: Hiệu suất trên tập test tốt hơn tập train ({diff:.4f})."
    else:
        status = f"PHÙ HỢP: Mô hình có hiệu suất tốt và cân bằng trên cả hai tập (chênh lệch {diff:.4f})."
    
    print(f"Kết luận: {status}")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'diff': diff,
        'status': status
    }

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
    
    # Kiểm tra overfitting
    print("\nKiểm tra overfitting và underfitting...")
    check_overfitting(classifier, X_train, y_train, X_test, y_test, "Mô hình Spam")
    
    # Vẽ đường cong học tập
    print("\nVẽ đường cong học tập...")
    plot_learning_curve(MultinomialNB(), X_train, y_train, X_test, y_test, "Mô hình Spam")
    
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
    
    # Kiểm tra overfitting
    print("\nKiểm tra overfitting và underfitting...")
    check_overfitting(classifier, X_train, y_train, X_test, y_test, "Mô hình Chủ đề")
    
    # Vẽ đường cong học tập
    print("\nVẽ đường cong học tập...")
    plot_learning_curve(MultinomialNB(), X_train, y_train, X_test, y_test, "Mô hình Chủ đề")
    
    return acc, precision, recall, f1

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