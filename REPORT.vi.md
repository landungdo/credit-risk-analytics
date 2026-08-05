# Báo cáo Kỹ thuật — Mô hình Rủi ro Tín dụng & Phân tích Danh mục

**Dự án:** Mô hình xác suất vỡ nợ (PD) có khả năng giải thích, trên dữ liệu Lending Club
**Phạm vi:** pipeline đầu-cuối — thu thập dữ liệu, phương pháp luận, mô hình hóa, khả năng giải thích, kiểm định công bằng, rủi ro danh mục, giám sát trôi phân phối, và API phục vụ kèm CI.

> 🌐 English version: see [REPORT.md](REPORT.md).

---

## 1. Phát biểu bài toán

Mục tiêu là ước lượng xác suất một khoản vay tiêu dùng bị vỡ nợ, theo cách đủ vững
để dùng trong rủi ro tín dụng thực tế — không phải chỉ chạy theo điểm số trên bảng
xếp hạng. Điều đó đòi hỏi ba thứ ngoài độ chính xác thô:

1. Cách đánh giá phải phản ánh đúng việc mô hình khi triển khai sẽ đối mặt với tương lai.
2. Xác suất dự đoán phải dùng được trực tiếp trong tính toán tổn thất và vốn.
3. Mọi quyết định cá nhân phải giải thích được và kiểm định được về mặt công bằng.

Ba yêu cầu này định hình mọi quyết định phía dưới.

---

## 2. Dữ liệu

**Nguồn.** Bộ dữ liệu khoản vay được duyệt của Lending Club (2007–2018), công khai
trên Kaggle. File gốc chứa ~2,2 triệu khoản vay và ~150 cột.

**Chọn cột.** Giữ lại 22 cột: kết quả (`loan_status`), ngày giải ngân (`issue_d`),
và các đặc điểm người vay/khoản vay có sẵn tại thời điểm nộp đơn (số tiền vay, kỳ
hạn, lãi suất, grade, thu nhập, DTI, số lần trễ hạn, số hạn mức tín dụng, tỷ lệ sử
dụng hạn mức quay vòng, thâm niên việc làm, tình trạng sở hữu nhà, mục đích vay,
bang, v.v.).

**Tránh rò rỉ ngay từ khâu chọn cột.** Các trường chỉ tồn tại *sau khi* khoản vay
đã chạy (ví dụ `total_pymnt`, `recoveries`) bị loại có chủ đích — chúng sẽ làm rò rỉ
kết quả vào đặc trưng.

**Mẫu làm việc.** Dùng mẫu ngẫu nhiên 20.000 dòng cho quá trình phát triển để
pipeline nhanh và dễ chia sẻ. Mẫu giữ nguyên toàn bộ khoảng thời gian, nên mọi phân
tích theo thời gian vẫn hợp lệ.

---

## 3. Định nghĩa biến mục tiêu

Nhãn được suy ra từ `loan_status`:

- **1 (vỡ nợ):** Charged Off (và biến thể theo ngoại lệ chính sách tín dụng)
- **0 (trả xong):** Fully Paid (và biến thể tương ứng)
- **Loại trừ:** Current, Late, In Grace Period — các khoản này chưa có kết quả cuối
  cùng, nên gán nhãn cho chúng đồng nghĩa với việc đưa phỏng đoán vào biến mục tiêu.

### Phát hiện quan trọng #1 — right-censoring theo vintage (kỳ giải ngân)

Tỷ lệ khoản vay đã có kết quả cuối cùng giảm mạnh ở các năm giải ngân gần đây:

| Năm giải ngân | % đã hoàn tất |
|---|---|
| 2014 | ~95% |
| 2015 | ~90% |
| 2016 | ~67% |
| 2017 | ~40% |
| 2018 | ~11% |

Các vintage gần đây phần lớn vẫn là khoản đang chạy, và những khoản *đã* hoàn tất
lại thiên lệch về nhóm kết thúc nhanh (kỳ hạn ngắn). Dùng nguyên trạng sẽ làm mô
hình bị méo. **Hệ quả:** loại hoàn toàn vintage 2017–2018 khỏi quá trình mô hình hóa.

---

## 4. Phương pháp luận

### 4.1 Chia tách theo thời gian (out-of-time, không phải ngẫu nhiên)

Chia ngẫu nhiên cho phép mô hình nhìn thấy khoản vay tương lai và quá khứ cùng lúc,
che giấu sự suy giảm mà một mô hình thật gặp phải theo thời gian. Thay vào đó, chia
theo trình tự thời gian:

- **Train:** issue_d < 2015
- **Validation:** 2015 (dùng cho early stopping và hiệu chỉnh)
- **Test:** 2016 (một vintage sau, thực sự giữ riêng)

Tỷ lệ vỡ nợ tăng dần qua các tập (≈16,9% → 20,9% → 23,7%), một hiệu ứng vintage
thật mà thiết kế out-of-time làm lộ ra thay vì che giấu.

### 4.2 Đặc trưng

Biến phân loại được đưa vào mô hình bằng kiểu dữ liệu `category` của pandas thay vì
one-hot (tránh nổ ~50 cột từ `addr_state`). Giá trị thiếu được giữ nguyên NaN có chủ
đích — mô hình cây xử lý được và thường học được split hữu ích từ "thiếu". Một đặc
trưng được tạo thêm, `credit_history_months` (ngày giải ngân trừ ngày mở hạn mức tín
dụng sớm nhất), là biến tín dụng có tín hiệu mạnh theo chuẩn ngành.

### 4.3 Mô hình

Cây tăng cường gradient (XGBoost) với early stopping trên vintage validation 2015,
độ sâu và regularization vừa phải để hạn chế overfit, và cố định random seed để tái
lập được.

### 4.4 Hiệu chỉnh (calibration)

Một mô hình xếp hạng tốt (AUC cao) chưa chắc đã *được hiệu chỉnh*. Vì PD được đưa
vào tính tổn thất và vốn, một bộ hiệu chỉnh isotonic được khớp trên vintage
validation và đánh giá trên vintage test, đo bằng Brier score và biểu đồ độ tin cậy.

---

## 5. Kết quả

### 5.1 Khả năng phân biệt (test out-of-time, 2016)

| Chỉ số | Train | Validation (2015) | Test (2016, OOT) |
|---|---|---|---|
| AUC | 0,79 | 0,72 | 0,68 |
| KS  | 0,44 | 0,32 | 0,28 |

Khoảng cách train–test hiện rõ và được *đo lường* — bức tranh trung thực mà một
phép chia out-of-time đưa ra, so với con số bị thổi phồng nếu chia ngẫu nhiên.

### 5.2 Hiệu chỉnh

Hiệu chỉnh isotonic làm giảm Brier score trên vintage test (≈0,173 → 0,170) và kéo
xác suất dự đoán về sát đường chéo lý tưởng trên biểu đồ độ tin cậy, giúp PD dùng
được như một xác suất thật:

![Biểu đồ độ tin cậy](reports/reliability_diagram.png)

### 5.3 Nghiên cứu loại bỏ (ablation) — kiểm tra rò rỉ và mô hình tham chiếu

`int_rate`, `grade`, và `sub_grade` do chính quy trình đánh giá rủi ro của Lending
Club gán, nên câu hỏi tự nhiên là liệu mô hình có chỉ học lại grade đã chấm sẵn hay
không. Ba mô hình trên cùng một phép chia:

| Mô hình | AUC | KS |
|---|---|---|
| FULL — XGBoost, mọi đặc trưng | 0,687 | 0,286 |
| NO_PRICING — bỏ int_rate/grade/sub_grade | 0,649 | 0,227 |
| BASELINE — hồi quy logistic, mọi đặc trưng | 0,676 | 0,260 |

### Phát hiện quan trọng #2 — điểm số không chủ yếu do rò rỉ, và tín hiệu phần lớn tuyến tính

- Bỏ toàn bộ biến định giá chỉ làm giảm ~0,04 AUC (0,687 → 0,649): mô hình vẫn giữ
  phần lớn khả năng phân biệt chỉ từ đặc điểm người vay, nên không phải chỉ lặp lại
  một grade đã chấm sẵn.
- XGBoost hơn baseline hồi quy logistic chỉ ~0,01 AUC: quan hệ phần lớn là tuyến
  tính/đơn điệu, nên một mô hình logistic kiểu scorecard sẽ là lựa chọn triển khai
  hợp lý và dễ giải thích hơn. Gradient boosting được giữ lại chủ yếu để phục vụ
  phần giải thích dựa trên SHAP.

---

## 6. Khả năng giải thích

Mỗi dự đoán được quy về các yếu tố tác động bằng SHAP, làm nổi các yếu tố chính đẩy
dự đoán về phía vỡ nợ hay trả nợ. Sau đó, một lý do từ chối bằng ngôn ngữ tự nhiên
được sinh ra từ các yếu tố đó.

### Phát hiện quan trọng #3 — cơ chế "grounding" chống bịa lý do

Một LLM khi được yêu cầu "giải thích một quyết định từ chối" có thể bịa ra những lý
do nghe hợp lý nhưng không có căn cứ — trong cho vay, đây là rủi ro tuân thủ. Một
bước kiểm tra sẽ đối chiếu văn bản sinh ra với các yếu tố SHAP được phép, và loại bỏ
bất kỳ lời giải thích nào đưa vào yếu tố mà SHAP không hề nêu. Hành vi này được kiểm
thử tự động: một lời giải thích "sạch" thì đạt, còn lời giải thích bịa thêm "lịch sử
việc làm" hay "tình trạng sở hữu nhà" thì bị bắt.

---

## 7. Công bằng

Dữ liệu không có thuộc tính được bảo vệ trực tiếp, nên phần kiểm định disparate
impact dùng biến đại diện (proxy) theo nhóm thu nhập và vùng. Tại một ngưỡng duyệt
minh họa, quy tắc bốn-phần-năm (four-fifths rule) đánh cờ nhóm thu nhập thấp. Điều
quan trọng là tỷ lệ vỡ nợ thực tế trong nhóm được duyệt cũng được báo cáo theo từng
nhóm: tỷ lệ bị từ chối cao hơn của nhóm bị đánh cờ đi kèm với tỷ lệ vỡ nợ thực tế
cao hơn, minh họa đúng căng thẳng cốt lõi của cho vay công bằng — giữa độ chính xác
thống kê và tác động chênh lệch. Một sự chênh lệch không tự động là bằng chứng của
thiên vị vô lý, nhưng vẫn cần được xem xét.

---

## 8. Rủi ro danh mục

Các PD đã hiệu chỉnh được tổng hợp thành các chỉ số danh mục mà bộ phận rủi ro báo cáo:

- **Tổn thất kỳ vọng (Expected Loss)** = PD × LGD × EAD (LGD 45%, EAD ≈ số tiền vay)
  — ≈10% tổng dư nợ trên danh mục test.
- **Vốn (Capital)** theo công thức Basel IRB đơn giản hóa — ≈18% tổng dư nợ.

Điều này khép lại vòng từ điểm số ML đến các con số tài chính mà điểm số hướng tới,
và là lý do vì sao phần hiệu chỉnh (Mục 5.2) lại quan trọng.

---

## 9. Giám sát trôi phân phối (drift)

Chỉ số Population Stability Index (PSI) được tính tổng thể và theo từng nhóm nhỏ.

### Phát hiện quan trọng #4 — trôi phân phối ở nhóm nhỏ bị ẩn sau một chỉ số tổng thể ổn định

PSI tổng thể rất thấp (~0,02, "ổn định"), nhưng vùng Northeast lại chạm ngưỡng
"moderate" (~0,10). Một hệ giám sát chỉ nhìn tổng thể sẽ bỏ lỡ điều này. Phản ứng
đúng mực là đưa nhóm này vào danh sách theo dõi, chứ không phải hiệu chỉnh lại mô
hình chỉ vì một chỉ số sát ngưỡng, trên mẫu nhỏ — đây chính là loại phán đoán mà một
hệ giám sát trôi tồn tại để hỗ trợ.

---

## 10. Kỹ thuật (Engineering)

- **Phục vụ (serving):** một service FastAPI cung cấp `/predict`, `/explain`, và
  `/portfolio/summary`, dựa trên các artifact mô hình đã lưu và nạp lúc khởi động.
- **Đóng gói container:** một Dockerfile huấn luyện và phục vụ mô hình.
- **Kiểm thử & CI:** bộ kiểm thử pytest (17 test) bao phủ metrics, tính toàn vẹn
  của phép chia, tính toán danh mục, tính chất PSI, và cơ chế grounding của phần
  giải thích; GitHub Actions chạy bộ test này mỗi lần push.

---

## 11. Giới hạn

Nêu rõ ràng để kết quả được đọc đúng bối cảnh:

- Các biến định giá (`int_rate`/`grade`/`sub_grade`) mang tính nội sinh một phần; mô
  hình NO_PRICING là ước lượng thận trọng cho rủi ro nội tại của người vay.
- Mất cân bằng lớp (~20%) được xử lý ngầm qua các chỉ số độc lập với ngưỡng, thay vì
  lấy mẫu lại.
- Ngưỡng duyệt trong phần công bằng và các giả định LGD/EAD chỉ mang tính minh họa,
  chưa được tinh chỉnh theo ma trận chi phí kinh doanh hay được mô hình hóa.
- Nhóm công bằng là biến đại diện, chưa phải phân tích cho vay công bằng ở mức tuân thủ.
- Các chỉ số đến từ một mẫu đại diện; giá trị chính xác thay đổi nhẹ theo mẫu/seed.

---

## 12. Kết luận

Dự án đưa ra một mô hình PD mà giá trị nằm ở phương pháp luận, không phải ở một con
số nổi bật: đánh giá out-of-time đo lường thay vì che giấu sự suy giảm, một xác suất
đã hiệu chỉnh dùng được cho tính toán tổn thất và vốn thật, các lời giải thích được
bảo vệ khỏi việc bịa đặt, một phần kiểm định công bằng phân biệt được chênh lệch với
thiên vị, và giám sát trôi bắt được điều mà chỉ số tổng thể bỏ lỡ. Nghiên cứu ablation
cho thấy kết quả không phải rò rỉ thuần túy cũng không phụ thuộc vào độ phức tạp của
mô hình — một lập trường trung thực và có thể bảo vệ. Những giới hạn còn lại được
giới hạn phạm vi có chủ đích và ghi rõ, và bản thân điều đó là một phần của mục tiêu
trình diễn: biết mô hình *không* chứng minh được điều gì cũng quan trọng như các chỉ
số của nó.
