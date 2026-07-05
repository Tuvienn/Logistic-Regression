# Trạng thái Project: Iris Flower Classification

## Trạng thái project hiện tại
Mới khởi tạo cấu trúc thư mục và các file quản lý cơ bản.

## Danh sách việc cần làm
- [ ] Tải dataset `iris.csv` từ nguồn ngoài (UCI hoặc Kaggle) và lưu vào thư mục `data/`.
- [ ] Viết notebook `iris_logistic_regression.ipynb` theo các bước:
  - Load dataset.
  - Data exploration.
  - Data preprocessing.
  - Train/test split.
  - Train Logistic Regression model (thử nghiệm tham số C, penalty, solver, max_iter).
  - Evaluate model.
  - Hyperparameter tuning.
  - Final prediction and conclusion.

## Việc đã hoàn thành
- [x] Tạo cấu trúc project `iris_logistic_regression/`.
- [x] Tạo các file quy tắc `.agents/README.md` và `.agents/TASK_STATUS.md`.
- [x] Tạo `README.md` và `requirements.txt`.

## Việc chưa hoàn thành
- Tải dataset.
- Implement code phân tích và mô hình hóa trong notebook.

## Ghi chú cho lần làm tiếp theo
- Tìm nguồn tải `iris.csv` chính xác (ví dụ: UCI ML Repository).
- Đảm bảo không sử dụng parameter của Decision Tree (max_depth, min_samples_split, v.v.).
- Bám sát yêu cầu chỉ sử dụng Logistic Regression và không dùng TensorFlow, Keras, PyTorch.
