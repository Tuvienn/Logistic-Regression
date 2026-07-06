import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

with open('iris_logistic_regression.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

phase3_idx = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and cell.source.startswith('## Phase 3:'):
        phase3_idx = i
        break

if phase3_idx != -1:
    nb.cells = nb.cells[:phase3_idx]

cells_to_add = [
    new_markdown_cell("## Phase 3: Train/Test Split & Preprocessing\n\nTrong bước này, chúng ta sẽ chia dữ liệu thành tập huấn luyện (train set) và tập kiểm thử (test set) với tỷ lệ 80/20. Đồng thời, chúng ta sẽ áp dụng `StandardScaler` để chuẩn hóa các đặc trưng (features). Việc chuẩn hóa giúp cho thuật toán Logistic Regression hội tụ nhanh và ổn định hơn."),
    new_code_cell("from phases.phase_03_preprocessing import run_preprocessing\n\npreprocessed = run_preprocessing(iris_dataset, export_artifacts=True)\nprint(f\"X_train shape: {preprocessed.X_train_scaled.shape}\")\nprint(f\"X_test shape: {preprocessed.X_test_scaled.shape}\")"),
    new_code_cell("preprocessed.X_train_scaled.head()"),
    
    new_markdown_cell("## Phase 4: Baseline Models\n\nTrước khi huấn luyện mô hình học máy thực thụ, chúng ta thiết lập một Baseline Model (Mô hình cơ sở) sử dụng `DummyClassifier`. Mô hình này luôn dự đoán nhãn xuất hiện nhiều nhất trong tập huấn luyện. Mục đích là để có một điểm chuẩn so sánh xem liệu mô hình Logistic Regression sau này có thực sự học được điều gì từ dữ liệu hay không."),
    new_code_cell("from phases.phase_04_baseline import run_baseline\n\nbaseline_report = run_baseline(preprocessed)\nprint(f\"Baseline strategy: {baseline_report.strategy}\")\nprint(f\"Baseline accuracy: {baseline_report.accuracy:.4f}\")"),
    
    new_markdown_cell("## Phase 5: Model Tuning\n\nTại đây, chúng ta sử dụng `GridSearchCV` để tìm kiếm siêu tham số tốt nhất (Hyperparameter tuning) cho mô hình `LogisticRegression`. Các tham số được thử nghiệm bao gồm hằng số điều chuẩn `C` và thuật toán tối ưu `solver`."),
    new_code_cell("from phases.phase_05_model_tuning import run_tuning\n\ntuned_model_report = run_tuning(preprocessed, export_artifacts=True)\nprint(f\"Best Parameters: {tuned_model_report.best_params}\")\nprint(f\"Best CV Score (Accuracy): {tuned_model_report.best_cv_score:.4f}\")"),
    
    new_markdown_cell("## Phase 6: Final Evaluation\n\nSử dụng mô hình tốt nhất từ Phase 5, chúng ta dự đoán trên tập kiểm thử (test set) và đánh giá chi tiết thông qua các chỉ số như Accuracy, Precision, Recall và F1-score. Ma trận nhầm lẫn (Confusion Matrix) cũng được trực quan hóa để xem chi tiết dự đoán cho từng nhãn."),
    new_code_cell("from phases.phase_06_evaluation import run_evaluation\n\nevaluation_report = run_evaluation(preprocessed, tuned_model_report, export_artifacts=True)\nprint(f\"Test Accuracy: {evaluation_report.accuracy:.4f}\")\nprint(\"\\nClassification Report:\")\nprint(evaluation_report.classification_report)"),
    new_code_cell("display(Image(filename='outputs/evaluation/confusion_matrix.png'))"),
    
    new_markdown_cell("## Phase 7: Prediction Analysis\n\nPhân tích các mẫu bị phân loại sai (Misclassified samples). Việc xem xét các dự đoán sai giúp ta hiểu rõ hơn điểm yếu của mô hình và ranh giới phân định giữa các lớp khó phân biệt (thường là Versicolor và Virginica)."),
    new_code_cell("from phases.phase_07_prediction_analysis import run_prediction_analysis\n\nprediction_report = run_prediction_analysis(preprocessed, tuned_model_report, export_artifacts=True)\nprint(f\"Misclassified samples count: {len(prediction_report.misclassified_df)}\")\nprediction_report.misclassified_df"),
    
    new_markdown_cell("## Phase 8: Model Interpretation\n\nCuối cùng, chúng ta trích xuất các trọng số (coefficients) của mô hình Logistic Regression để hiểu mức độ ảnh hưởng của từng đặc trưng lên quyết định phân loại cho từng nhãn loài hoa."),
    new_code_cell("from phases.phase_08_interpretation import run_interpretation\n\ninterpretation_report = run_interpretation(preprocessed, tuned_model_report, export_artifacts=True)\ninterpretation_report.coefficients_df"),
    new_code_cell("display(Image(filename='outputs/interpretation/coefficients_plot.png'))"),
    
    new_markdown_cell("## Phase 9: Conclusion\n\nQua quá trình phân tích và huấn luyện, mô hình Logistic Regression đã chứng minh khả năng phân loại rất tốt đối với tập dữ liệu Iris. Việc chuẩn hóa dữ liệu, sử dụng baseline model và thực hiện tinh chỉnh tham số (hyperparameter tuning) đã giúp tìm ra mô hình tối ưu nhất một cách rõ ràng và khoa học. Các trường hợp phân loại sai (nếu có) phản ánh tính chất thực tế của ranh giới giữa một số nhãn trong bộ dữ liệu.")
]

nb.cells.extend(cells_to_add)

with open('iris_logistic_regression.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
