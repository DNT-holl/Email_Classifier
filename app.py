import streamlit as st
from data_processor import DataProcessor
from spam_classifier import SpamClassifier
from topic_classifier import TopicClassifier
import pandas as pd
import plotly.express as px

def load_models():
    """Tải cả hai mô hình phân loại spam và chủ đề"""
    # Tải bộ phân loại spam
    spam_classifier = SpamClassifier()
    try:
        spam_classifier.load_model()
        st.success("Đã tải mô hình spam thành công!")
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy mô hình spam! Vui lòng chạy 'python train_model.py' trước.")
        st.stop()
        
    # Tải bộ phân loại chủ đề
    topic_classifier = TopicClassifier()
    try:
        topic_classifier.load_model()
        st.success("Đã tải mô hình chủ đề thành công!")
    except FileNotFoundError:
        st.error("Lỗi: Không tìm thấy mô hình chủ đề! Vui lòng chạy 'python train_topic_model.py' trước.")
        st.stop()
        
    return spam_classifier, topic_classifier

def main():
    st.set_page_config(
        page_title="Hệ thống Phân loại Email",
        page_icon="📧",
        layout="wide"
    )
    
    st.title("📧 Hệ thống Phân loại Email")
    st.markdown("""
    Hệ thống này có thể:
    1. Phát hiện email spam hoặc không spam
    2. Phân loại email không spam thành các chủ đề (công nghệ, kinh doanh, giải trí, thể thao, chính trị)
    """)
    
    # Tải mô hình
    spam_classifier, topic_classifier = load_models()
    
    # Khởi tạo bộ xử lý dữ liệu
    data_processor = DataProcessor('spam.csv')
    
    # Đảm bảo vectorizer được tải
    try:
        data_processor.load_spam_vectorizer()
        data_processor.load_topic_vectorizer()
    except FileNotFoundError as e:
        st.error(f"Lỗi: {str(e)}")
        st.error("Vui lòng chạy cả 'python train_model.py' và 'python train_topic_model.py' trước.")
        st.stop()
    
    # Tạo hai cột cho đầu vào và kết quả
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Nhập Email")
        email_text = st.text_area(
            "Nhập nội dung email của bạn tại đây:",
            height=300,
            placeholder="Nhập hoặc dán nội dung email của bạn tại đây..."
        )
        
        if st.button("Phân tích Email", type="primary"):
            if not email_text.strip():
                st.warning("Vui lòng nhập nội dung email!")
            else:
                try:
                    # Xử lý email
                    X_spam = data_processor.transform_new_data([email_text], vectorizer_type='spam')
                    
                    # Lấy dự đoán spam
                    spam_prediction = spam_classifier.predict(X_spam)[0]
                    spam_probability = spam_classifier.predict_proba(X_spam)[0]
                    
                    # Hiển thị kết quả trong cột thứ hai
                    with col2:
                        st.subheader("Kết quả Phân tích")
                        
                        # Tạo container cho kết quả
                        results_container = st.container()
                        
                        with results_container:
                            # Hiển thị kết quả spam
                            spam_result = "Spam" if spam_prediction == 1 else "Không phải Spam"
                            spam_color = "red" if spam_prediction == 1 else "green"
                            st.markdown(f"### Phát hiện Spam: <span style='color:{spam_color}'>{spam_result}</span>", unsafe_allow_html=True)
                            st.progress(spam_probability[1])
                            st.markdown(f"Độ tin cậy: {spam_probability[1]*100:.2f}% (spam)")
                            
                            # Nếu không phải spam, phân loại chủ đề
                            if spam_prediction == 0:
                                try:
                                    # Lấy dự đoán chủ đề
                                    X_topic = data_processor.transform_new_data([email_text], vectorizer_type='topic')
                                    topic_prediction = topic_classifier.predict(X_topic)[0]
                                    topic_probability = topic_classifier.predict_proba(X_topic)[0]
                                    
                                    # Hiển thị kết quả chủ đề
                                    st.markdown("### Phân loại Chủ đề")
                                    
                                    # Tạo biểu đồ cột cho xác suất chủ đề
                                    topic_names = topic_classifier.topics
                                    topic_probs = topic_probability
                                    
                                    # Tạo DataFrame cho biểu đồ
                                    df = pd.DataFrame({
                                        'Chủ đề': topic_names,
                                        'Xác suất': topic_probs
                                    })
                                    
                                    # Tạo biểu đồ cột
                                    fig = px.bar(
                                        df,
                                        x='Chủ đề',
                                        y='Xác suất',
                                        color='Xác suất',
                                        color_continuous_scale='Viridis',
                                        title='Xác suất Phân loại Chủ đề'
                                    )
                                    
                                    # Cập nhật layout
                                    fig.update_layout(
                                        xaxis_title="Chủ đề",
                                        yaxis_title="Xác suất",
                                        showlegend=False
                                    )
                                    
                                    # Hiển thị biểu đồ
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Hiển thị chủ đề được dự đoán
                                    st.markdown(f"**Chủ đề dự đoán:** {topic_classifier.get_topic_name(topic_prediction)}")
                                    st.markdown(f"**Độ tin cậy:** {topic_probability[topic_prediction]*100:.2f}%")
                                    
                                except Exception as e:
                                    st.error(f"Lỗi khi phân loại chủ đề: {str(e)}")
                                    st.error("Vui lòng chạy 'python train_topic_model.py' để huấn luyện mô hình chủ đề.")
                
                except Exception as e:
                    st.error(f"Lỗi khi xử lý email: {str(e)}")
    
    # Thêm thông tin về hệ thống
    with st.sidebar:
        st.header("Giới thiệu")
        st.markdown("""
        Hệ thống này sử dụng học máy để:
        1. Phát hiện email spam sử dụng phân loại Naive Bayes
        2. Phân loại email không spam thành 5 chủ đề:
           - Công nghệ
           - Kinh doanh
           - Giải trí
           - Thể thao
           - Chính trị
        
        Hệ thống được huấn luyện trên:
        - Bộ dữ liệu spam cho việc phát hiện spam
        - Bộ dữ liệu BBC News cho việc phân loại chủ đề
        """)
        
        st.header("Cách sử dụng")
        st.markdown("""
        1. Nhập hoặc dán nội dung email vào ô văn bản
        2. Nhấn nút "Phân tích Email"
        3. Xem kết quả:
           - Kết quả phát hiện spam
           - Phân loại chủ đề (nếu không phải spam)
        """)

if __name__ == "__main__":
    main() 