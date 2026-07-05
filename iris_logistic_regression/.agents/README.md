# 🌸 Iris Flower Classification - Agent Working Rules

## 🎯 Mục tiêu Project
Phân loại chính xác 3 loài hoa Iris (Setosa, Versicolor, Virginica) bằng thuật toán **Logistic Regression**. Dự án này phục vụ mục đích giảng dạy/thực hành, do đó mọi bước triển khai cần phải cực kỳ chi tiết, dễ hiểu và mang tính sư phạm cao.

## 📜 Quy tắc làm việc cốt lõi của Agent (Mandatory Rules)

Tất cả các hành động của AI Agent/Subagent trong project này **BẮT BUỘC** phải tuân thủ các quy tắc sau:

### 1. Quản lý Dữ liệu (Strict Data Policy)
- ❌ **KHÔNG ĐƯỢC** tự chế dữ liệu giả (No mock data).
- ❌ **KHÔNG ĐƯỢC** sử dụng hàm `load_iris()` có sẵn từ thư viện `sklearn.datasets`.
- ✅ **BẮT BUỘC** phải tải/đọc dữ liệu thực tế từ file CSV được đặt tại đường dẫn: `data/iris.csv`.
- Nguồn tải dữ liệu ưu tiên: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) hoặc Kaggle.

### 2. Tiêu chuẩn Coding & Document (Pedagogical Focus)
- **Chất lượng Code:** Python code phải được viết cực kỳ rõ ràng, tuân thủ PEP8, cấu trúc rành mạch và dễ hiểu đối với sinh viên.
- **Giải thích (Markdown):** Mọi block code trong Jupyter Notebook phải được đi kèm với ít nhất một Markdown cell giải thích chi tiết **TẠI SAO** lại thực hiện bước đó bằng **Tiếng Việt**.
- **Chú thích (Comments):** Comment trực tiếp bên trong code (inline comments / docstrings) ưu tiên dùng **Tiếng Anh đơn giản** để sinh viên làm quen với môi trường quốc tế.

### 3. Ràng buộc về Thuật toán (Algorithm Constraints)
- Chỉ tập trung triển khai, đánh giá và tối ưu mô hình **Logistic Regression**.
- ❌ Tuyệt đối **KHÔNG** sử dụng các Deep Learning frameworks (như TensorFlow, Keras, PyTorch).
- ❌ **KHÔNG** tuning/sử dụng các tham số thuộc về Decision Tree (ví dụ: `max_depth`, `min_samples_split`, `min_samples_leaf`).
- ✅ Khi thực hiện tuning cho Logistic Regression, chỉ tinh chỉnh các tham số hợp lệ của mô hình này: `C`, `penalty`, `solver`, `max_iter`.

### 4. Kế thừa Working Rules
- AI Agent luôn phải tham chiếu và tuân thủ các quy định giao tiếp, báo cáo và làm việc chung được định nghĩa ở thư mục gốc: `../working_rule.md`.
