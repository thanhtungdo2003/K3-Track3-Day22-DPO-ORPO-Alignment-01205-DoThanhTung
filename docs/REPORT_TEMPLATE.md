# Preference Alignment Experiment Report

Student: Đỗ Thanh Tùng
Student ID: 2A202601205

## 1. Phân tích và Làm sạch Dữ liệu

### Data Loading Summary

* **Tổng số mẫu được nạp**: 24 cặp preference có gắn nhãn.
* **Lỗi dữ liệu ban đầu**: Dòng 1 chứa dấu ngoặc kép `"` chưa được escape xung quanh `self-attention`, khiến file JSONL ban đầu không thể parse tại cột 36. Sau khi escape các dấu ngoặc kép, bản ghi đã được sửa hợp lệ.
* **Kiểm tra và chuẩn hóa**: Loader báo cáo file và số dòng khi xảy ra lỗi JSON hoặc schema, đồng thời từ chối các prompt trùng lặp sau khi chuẩn hóa. Các trường text bắt buộc được `strip` và không được để trống. Kiểm tra trùng lặp sử dụng Unicode NFKC, `casefold` và chuẩn hóa khoảng trắng bằng cách tách rồi ghép lại, do đó các biến thể chỉ khác Unicode, chữ hoa/chữ thường hoặc khoảng trắng không thể vượt qua validation. Các trường không xác định ở cấp example cũng bị từ chối.

### Split Strategy

* **Tỷ lệ train/validation**: Cách chia mặc định 80/20 tạo ra 19 nhóm prompt cho train và 5 nhóm prompt cho validation. Không có tập test.
* **Tránh leakage và khả năng tái lập**: Dữ liệu được nhóm theo prompt đã chuẩn hóa, sau đó các key chuẩn hóa được sắp xếp và shuffle bằng local seeded RNG với seed mặc định là 42. Việc sort trước khi shuffle loại bỏ sự phụ thuộc trước đây vào thứ tự không xác định của `set`. Toàn bộ một nhóm prompt luôn nằm trong cùng một partition, mỗi input row được giữ lại đúng một lần và thứ tự row trong mỗi partition được bảo toàn.

## 2. Triển khai DPO và ORPO

### Objective Selection

DPO là objective chính, sử dụng reference model cố định làm cơ sở so sánh. DPO ưu tiên log-ratio giữa response được chọn và response bị từ chối của policy so với log-ratio tương ứng của reference model.

ORPO cũng được triển khai dưới dạng kết hợp giữa SFT mean loss và preference penalty dựa trên odds ratio có trọng số.

* `beta`: mặc định 0.1, phải là số hữu hạn và lớn hơn 0.
* `lambda_orpo`: mặc định 0.1, phải là số hữu hạn và lớn hơn hoặc bằng 0.

### Input Contracts

* **DPO**: Chuyển các giá trị số thực sang `float64` và từ chối giá trị complex trước khi cast. Cả bốn mảng log-probability phải không rỗng, hữu hạn, có cùng shape chính xác và có giá trị không lớn hơn 0. Việc kiểm tra exact shape được thực hiện có chủ đích để ngăn NumPy tự động broadcasting các mảng không tương thích. Scaled margin cũng phải là giá trị hữu hạn.

* **ORPO**: Yêu cầu các mảng chosen và rejected log-probability phải không rỗng, hữu hạn, có cùng shape và có giá trị nhỏ hơn 0. Giá trị 0 bị loại vì `log(1 - exp(logp))`, và do đó log-odds, trở nên không xác định tại `logp = 0`. Giá trị SFT NLL phải không rỗng, hữu hạn và không âm. Mean của SFT NLL được tính độc lập với preference-batch mean. Preference weight có thể bằng 0.

### Numerical Stability

Cả hai objective đều tính log-sigmoid bằng:

`-np.logaddexp(0, -x)`

ORPO không clipping các log-probability hợp lệ. Các loss term lớn không âm được scale trước khi tính mean nhằm tránh overflow khi nhiều giá trị hữu hạn tiến gần giới hạn của `float64`.

ORPO tính `log(1 - exp(logp))` theo từng trường hợp. Với các giá trị lớn hơn `-log(2)`, hệ thống sử dụng `log(-expm1(logp))` để đảm bảo độ chính xác khi giá trị tiến gần 0. Với các giá trị còn lại, sử dụng `log1p(-exp(logp))` để đảm bảo tính ổn định ở negative tail.

Một regression với chosen/rejected log-probabilities lần lượt là `-100` và `-50` cho penalty xấp xỉ `50`. Khi đảo lại theo thứ tự ưu tiên, penalty giảm xuống khoảng `1.9287498479639178e-22`. Giá trị âm gần 0 nhất có thể biểu diễn vẫn cho kết quả hữu hạn.

Các closed-form fixtures cho kết quả:

* **DPO loss**: `0.663597`
* **ORPO loss**: `1.017086`

### Configuration and Reproducible Environment

Configuration được parse thành các nested typed Pydantic models. Unknown fields bị cấm ở mọi cấp độ cấu hình. Các training value được ràng buộc theo các điều kiện hợp lệ. Method chỉ được phép là `dpo`, `orpo` hoặc `mock`. Các relative path tới data, output và regression được resolve dựa trên thư mục chứa configuration file.

Native workflow là:

`uv sync --locked --extra dev`

sau đó:

`uv run ...`

CI sử dụng cùng lockfile-based workflow.

Python 3.11 được pin trong `.python-version` và là môi trường đã được kiểm thử. Project metadata khai báo hỗ trợ Python `>=3.10`, tuy nhiên báo cáo này không khẳng định rằng mọi phiên bản Python hợp lệ đều đã được kiểm thử.

Thay đổi duy nhất đối với `trainers.py` ngoài student TODO là formatting blank line theo yêu cầu của Ruff và không ảnh hưởng đến behavior.

## 3. Evaluation Results

### Exact Metrics Artifact

```json
{
  "evaluation_scope": "full_sample_lexical_baseline",
  "losses": 7,
  "mean_score_margin": 0.015940656565656557,
  "num_examples": 24,
  "pairwise_accuracy": 0.5208333333333334,
  "scorer": "lexical_overlap_v1",
  "tie_rate": 0.375,
  "ties": 9,
  "wins": 8
}
```

`lexical_overlap_v1` đo tỷ lệ các unique prompt word tokens xuất hiện trong mỗi response.

Đối với một labeled pair:

* Chosen response thắng nhận 1 điểm.
* Tie nhận 0.5 điểm.
* Rejected response thắng nhận 0 điểm.

Do đó, accuracy được tính:

`(8 + 0.5 * 9) / 24 = 0.5208333333333334`

Các metrics này bao phủ toàn bộ 24 labeled pairs trong `data/sample_preferences.jsonl`.

Đây **không phải** held-out evaluation, trained-model evaluation, before/after training comparison hoặc regression-prompt evaluation.

Không có training run nào được thực hiện. Các closed-form losses ở trên là unit-test fixtures, không phải final training losses.

### Qualitative Failure Example

* **Prompt**: What is the purpose of a confusion matrix?
* **Chosen response**: A confusion matrix provides a detailed breakdown of classification performance, showing true positives, true negatives, false positives, and false negatives.
* **Rejected response**: A confusion matrix is used to visualize the distribution of the target variable in the dataset.
* **Scorer preference**: Không chính xác. Rejected response đạt điểm 0.75 so với 0.50 vì nó lặp lại nhiều từ trong prompt hơn, mặc dù nội dung thực tế là sai.

## 4. Discussion and Failure Modes

* **What worked**: Các lỗi loading và configuration được báo cáo ngắn gọn, có đầy đủ context; schema và metric boundaries được validation; split có tính deterministic và tránh leakage; DPO/ORPO losses khớp với closed-form và extreme-value tests mà không cần clipping.

* **Observed and fixed defects**: JSON ban đầu bị malformed và đã được sửa. Audit cũng phát hiện baseline row-based split không group các equivalent prompts, whitespace-only text có thể vượt qua validation, NumPy có thể silently broadcast các loss array không cùng shape, và cả DPO lẫn ORPO đều chưa được implement. Các vấn đề này được xử lý bằng normalized prompt grouping với sorted seeded keys, strip-before-length validation kết hợp canonical normalization, exact-shape checks và stable log-domain objectives.

* **Remaining correctness limitation**: Lexical overlap đo việc tái sử dụng từ thay vì factual quality. Phương pháp này thua ở 7 labeled pairs và hòa ở 9 pairs. Ví dụ confusion matrix cho thấy một response sai về mặt thực tế vẫn có thể thắng chỉ vì lặp lại nhiều từ trong prompt. Response dài cũng có nhiều cơ hội tạo lexical overlap hơn. Vì vậy, score chỉ phù hợp làm deterministic smoke-test baseline.

* **Safety gap**: `PreferenceTrainer.train()` vẫn chưa được implement và chưa có generative model nào được chạy. Do đó, bốn regression prompts chưa thể được đánh dấu pass hoặc fail. Một evaluation thực tế cần kiểm tra riêng khả năng redirection đối với high-risk medical requests, tuân thủ strict summary limits, uncertainty calibration và xử lý các troubleshooting requests thiếu context.
