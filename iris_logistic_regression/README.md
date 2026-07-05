<div align="center">
  <h1>🌸 Iris Flower Classification</h1>
  <p><i>Phân loại giống hoa Iris sử dụng Logistic Regression (Machine Learning Lab)</i></p>
</div>

---

## 📖 Giới thiệu
Đây là project bài lab Machine Learning kinh điển về phân loại hoa Iris. Bài toán tập trung vào việc áp dụng thuật toán **Logistic Regression** (Hồi quy Logistic) từ đầu đến cuối quy trình, giúp người học hiểu rõ cách thức hoạt động của mô hình phân loại tuyến tính trên dữ liệu dạng bảng.

## 🎯 Mục tiêu bài toán
Mục tiêu chính là phân loại chính xác 3 loài hoa Iris: 
1. **Iris-setosa**
2. **Iris-versicolor**
3. **Iris-virginica**

Dựa trên 4 đặc trưng (features) sinh học đo được:
- Chiều dài đài hoa (`sepal length`)
- Chiều rộng đài hoa (`sepal width`)
- Chiều dài cánh hoa (`petal length`)
- Chiều rộng cánh hoa (`petal width`)

## 📊 Dataset sử dụng
- **Vị trí file dữ liệu**: `data/iris.csv`
- **Nguồn dữ liệu**: Lấy từ nguồn cung cấp uy tín bên ngoài (ví dụ: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris)).
- **Lưu ý quan trọng**: Dự án này yêu cầu kỹ năng Data Pipeline cơ bản, do đó tuyệt đối **KHÔNG** sử dụng hàm `load_iris()` có sẵn của `scikit-learn` hay bất kỳ thư viện mock data nào.

## ⚙️ Mô hình sử dụng
- **Mô hình duy nhất**: Logistic Regression (từ `sklearn.linear_model`)
- **Tối ưu hóa**: Thử nghiệm các tham số `C`, `penalty`, `solver`, và `max_iter`.

## 🛠 Các bước triển khai (Project Workflow)

Quá trình phân tích và huấn luyện trong file Jupyter Notebook sẽ đi qua 8 bước chính sau:

1. **Load dataset**: Đọc và nạp dữ liệu từ file csv bằng `pandas`.
2. **Data exploration (EDA)**: Khám phá dữ liệu, kiểm tra các đại lượng thống kê, vẽ biểu đồ phân bố đặc trưng và nhãn để thấy rõ ranh giới các loài.
3. **Data preprocessing**: Xử lý dữ liệu trống (nếu có) và thực hiện chuẩn hóa dữ liệu (Feature Scaling) phù hợp với Logistic Regression.
4. **Train/test split**: Phân chia tập dữ liệu thành tập huấn luyện (train set) và tập kiểm thử (test set) một cách khách quan.
5. **Train Logistic Regression model**: Khởi tạo và huấn luyện mô hình Logistic Regression cơ bản.
6. **Evaluate model**: Đánh giá chất lượng mô hình trên tập kiểm thử thông qua các độ đo (metrics) như Accuracy, Confusion Matrix, và Classification Report.
7. **Hyperparameter tuning**: Tiến hành tinh chỉnh (Fine-tuning) các tham số của thuật toán để tìm ra mô hình tối ưu nhất.
8. **Final prediction and conclusion**: Chạy dự đoán cuối cùng trên mẫu ngẫu nhiên và đưa ra kết luận về ưu/nhược điểm của phương pháp.

---
*Dự án này là bài tập thực hành được xây dựng dựa trên sự hợp tác chuyên nghiệp giữa Human và AI Agent.*
