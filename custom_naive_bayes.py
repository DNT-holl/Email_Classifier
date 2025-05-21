import numpy as np
from scipy.sparse import issparse
import joblib
import warnings


class CustomMultinomialNB:
    """
    Triển khai Multinomial Naive Bayes tự xây
    Tương thích với MultinomialNB của scikit-learn
    """
    def __init__(self, alpha=1.0):
        """
        Khởi tạo Multinomial Naive Bayes
        
        Tham số:
        alpha : float, mặc định=1.0
            Tham số Laplace/Lidstone smoothing. 0 nghĩa là không smoothing.
        """
        self.alpha = alpha
        self.class_count_ = None  # Số lượng văn bản trong mỗi lớp
        self.class_log_prior_ = None  # Log của xác suất tiên nghiệm P(y)
        self.classes_ = None  # Các lớp
        self.n_classes_ = 0  # Số lượng lớp
        self.feature_count_ = None  # Số lượng đặc trưng trong mỗi lớp
        self.feature_log_prob_ = None  # Log của xác suất có điều kiện P(x_i|y)
        self.n_features_ = None  # Số lượng đặc trưng
    
    def _check_X(self, X):
        """Kiểm tra dữ liệu đầu vào X"""
        if issparse(X):
            # Chuyển đổi ma trận thưa thành mảng lưu trữ CSR
            X = X.tocsr()
        else:
            # Chuyển đổi thành numpy array
            X = np.asarray(X)
        
        if X.dtype != 'int' and X.dtype != 'float':
            warnings.warn("X có kiểu dữ liệu không phải số, có thể ảnh hưởng đến kết quả")
        
        return X
    
    def _count_features(self, X, y):
        """Đếm số lượng đặc trưng trong mỗi lớp"""
        self.feature_count_ = np.zeros((self.n_classes_, self.n_features_), dtype=np.float64)
        
        for i, c in enumerate(self.classes_):
            if issparse(X):
                # Xử lý ma trận thưa
                X_c = X[y == c]
                if X_c.shape[0] > 0:
                    self.feature_count_[i] = np.bincount(
                        X_c.indices, weights=X_c.data, minlength=self.n_features_
                    )
            else:
                # Xử lý ma trận đầy đủ
                X_c = X[y == c]
                if X_c.shape[0] > 0:
                    self.feature_count_[i] = np.sum(X_c, axis=0)
    
    def _compute_log_priors(self):
        """Tính log của xác suất tiên nghiệm P(y)"""
        self.class_log_prior_ = np.log(self.class_count_) - np.log(np.sum(self.class_count_))
    
    def _compute_feature_log_prob(self):
        """Tính log của xác suất có điều kiện P(x_i|y)"""
        # Áp dụng Laplace smoothing
        smoothed_fc = self.feature_count_ + self.alpha
        # Tính tổng số đặc trưng trong mỗi lớp (sau khi smoothing)
        sum_fc = np.sum(smoothed_fc, axis=1).reshape(-1, 1)
        # Tính log của xác suất có điều kiện
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(sum_fc)
    
    def fit(self, X, y):
        """
        Huấn luyện mô hình Multinomial Naive Bayes
        
        Tham số:
        X : array-like hoặc sparse matrix, shape (n_samples, n_features)
            Các đặc trưng đã số hóa
        y : array-like, shape (n_samples,)
            Nhãn mục tiêu
            
        Trả về:
        self : đối tượng
        """
        X = self._check_X(X)
        if issparse(X):
            self.n_features_ = X.shape[1]
        else:
            self.n_features_ = X.shape[1]
        
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Đếm số lượng văn bản trong mỗi lớp
        self.class_count_ = np.zeros(self.n_classes_, dtype=np.float64)
        for i, c in enumerate(self.classes_):
            self.class_count_[i] = np.sum(y == c)
        
        # Đếm số lượng đặc trưng trong mỗi lớp
        self._count_features(X, y)
        
        # Tính toán các xác suất
        self._compute_log_priors()
        self._compute_feature_log_prob()
        
        return self
    
    def predict_log_proba(self, X):
        """
        Trả về log của xác suất dự đoán cho mỗi lớp
        
        Tham số:
        X : array-like hoặc sparse matrix, shape (n_samples, n_features)
            Dữ liệu vector đặc trưng
            
        Trả về:
        log_proba : array, shape (n_samples, n_classes)
            Log của xác suất của mỗi mẫu cho mỗi lớp
        """
        if not hasattr(self, 'classes_'):
            raise AttributeError("Mô hình chưa được huấn luyện")
        
        X = self._check_X(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        if n_features != self.n_features_:
            raise ValueError(f"Số lượng đặc trưng của X ({n_features}) không khớp với mô hình ({self.n_features_})")
        
        # Tính log của xác suất hậu nghiệm
        joint_log_likelihood = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            if issparse(X):
                # Tính toán cho ma trận thưa
                joint_log_likelihood[:, i] = self.class_log_prior_[i]
                
                # Xử lý ma trận thưa bằng cách tính tổng trên indices
                for r in range(n_samples):
                    indices = X[r].indices
                    data = X[r].data
                    for j, d in zip(indices, data):
                        if j < self.n_features_:
                            joint_log_likelihood[r, i] += self.feature_log_prob_[i, j] * d
            else:
                # Tính toán cho ma trận đầy đủ
                joint_log_likelihood[:, i] = safe_sparse_dot(X, self.feature_log_prob_[i])
                joint_log_likelihood[:, i] += self.class_log_prior_[i]
        
        # Chuẩn hóa để tránh underflow
        log_prob_x = logsumexp(joint_log_likelihood, axis=1).reshape(-1, 1)
        log_proba = joint_log_likelihood - log_prob_x
        
        return log_proba
    
    def predict_proba(self, X):
        """
        Trả về xác suất dự đoán cho mỗi lớp
        
        Tham số:
        X : array-like hoặc sparse matrix, shape (n_samples, n_features)
            Dữ liệu vector đặc trưng
            
        Trả về:
        proba : array, shape (n_samples, n_classes)
            Xác suất của mỗi mẫu cho mỗi lớp
        """
        log_proba = self.predict_log_proba(X)
        return np.exp(log_proba)
    
    def predict(self, X):
        """
        Dự đoán lớp cho X
        
        Tham số:
        X : array-like hoặc sparse matrix, shape (n_samples, n_features)
            Dữ liệu vector đặc trưng
            
        Trả về:
        y : array, shape (n_samples,)
            Nhãn dự đoán
        """
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]
    
    def _joint_log_likelihood(self, X):
        """Tính log của xác suất đồng thời P(x, y)"""
        X = self._check_X(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        
        if n_features != self.n_features_:
            raise ValueError(f"Số lượng đặc trưng của X ({n_features}) không khớp với mô hình ({self.n_features_})")
        
        joint_log_likelihood = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            if issparse(X):
                # Tính toán cho ma trận thưa
                joint_log_likelihood[:, i] = self.class_log_prior_[i]
                
                # Xử lý ma trận thưa bằng cách tính tổng trên indices
                for r in range(n_samples):
                    indices = X[r].indices
                    data = X[r].data
                    for j, d in zip(indices, data):
                        if j < self.n_features_:
                            joint_log_likelihood[r, i] += self.feature_log_prob_[i, j] * d
            else:
                # Tính toán cho ma trận đầy đủ
                joint_log_likelihood[:, i] = self.class_log_prior_[i]
                joint_log_likelihood[:, i] += np.dot(X, self.feature_log_prob_[i])
        
        return joint_log_likelihood
    
    def save_model(self, filename):
        """Lưu mô hình vào file"""
        joblib.dump(self, filename)
    
    @classmethod
    def load_model(cls, filename):
        """Tải mô hình từ file"""
        return joblib.load(filename)


# Các hàm tiện ích
def safe_sparse_dot(a, b):
    """Nhân ma trận an toàn cho cả ma trận thưa và đầy đủ"""
    if issparse(a) or issparse(b):
        return np.array(a.dot(b))
    else:
        return np.dot(a, b)


def logsumexp(arr, axis=None):
    """Tính log(sum(exp(arr))) một cách ổn định số học"""
    arr_max = np.max(arr, axis=axis, keepdims=True)
    arr_max_broadcast = np.broadcast_to(arr_max, arr.shape)
    
    # Đối với trường hợp đơn giản khi axis=None
    if axis is None:
        return np.log(np.sum(np.exp(arr - arr_max_broadcast))) + arr_max[0]
    
    # Đối với trường hợp có axis
    return np.log(np.sum(np.exp(arr - arr_max_broadcast), axis=axis)) + np.squeeze(arr_max, axis=axis) 