# 🚚 Dự Án Này Tối Ưu Cái Gì? — Giải Thích Cho Người Không Chuyên

> Tài liệu này dành cho người **không cần biết lập trình hay toán học** vẫn hiểu được:
> dự án giải quyết vấn đề gì, tối ưu ở đâu, làm sao biết nó thật sự tốt, tốt hơn bao nhiêu,
> đã thêm những "bộ phận" mới nào, và tại sao lại thêm.

---

## 1. Bài toán thực tế: "Sắp xe đi giao hàng sao cho rẻ nhất"

Hãy tưởng tượng bạn là chủ một công ty giao hàng. Sáng nay bạn có:

- **1 kho hàng** (gọi là *depot*) — nơi mọi xe xuất phát và quay về.
- **100 khách hàng** rải rác khắp thành phố, mỗi người đặt một lượng hàng khác nhau.
- **Một đội xe tải** giống nhau, mỗi xe chở được tối đa (ví dụ) 2 tấn.
- Và điểm khó nhất: **mỗi khách chỉ nhận hàng trong một khung giờ nhất định** — ví dụ nhà hàng A chỉ nhận từ 8h–9h sáng, cửa hàng B chỉ nhận từ 14h–15h chiều. Đây gọi là **"cửa sổ thời gian" (time window)**. Đến sớm thì xe phải **chờ**; đến muộn thì **coi như hỏng**, không được phép.

Câu hỏi đặt ra: **Nên dùng bao nhiêu xe, và mỗi xe đi theo thứ tự nào**, để:

1. **Dùng ít xe nhất có thể** (mỗi xe thêm là thêm một tài xế, thêm chi phí cố định).
2. **Tổng quãng đường chạy ngắn nhất có thể** (tiết kiệm xăng, thời gian).
3. Mà **không vi phạm** bất kỳ khung giờ nào và không xe nào chở quá tải.

Đây chính là bài toán **VRPTW** (*Vehicle Routing Problem with Time Windows* — Bài toán định tuyến xe có cửa sổ thời gian). Nghe thì đơn giản, nhưng...

### Tại sao nó khó đến mức cần cả một dự án nghiên cứu?

Với chỉ 100 khách, số cách sắp xếp lộ trình khả dĩ **lớn hơn số nguyên tử trong vũ trụ**. Không máy tính nào trên đời "thử hết mọi cách" được. Vì vậy mục tiêu không phải tìm ra lời giải *hoàn hảo tuyệt đối* (bất khả thi), mà tìm ra lời giải **rất gần mức tốt nhất** trong thời gian hợp lý (vài giây đến vài phút). Đây là loại bài toán mà ngành logistics thực tế (Giao Hàng Nhanh, Grab, Amazon...) phải giải mỗi ngày.

---

## 2. Tối ưu **cái gì** và **ở đâu**?

Dự án đo "độ tốt" của một lời giải bằng **đúng 2 con số**, theo thứ tự ưu tiên:

| Ký hiệu | Tên | Ý nghĩa đời thường | Ưu tiên |
| :--- | :--- | :--- | :---: |
| **NV** | *Number of Vehicles* — Số xe | Cần bao nhiêu tài xế/xe tải | **Ưu tiên 1** (giảm trước) |
| **TD** | *Total Distance* — Tổng quãng đường | Tổng số km cả đội xe chạy | Ưu tiên 2 (giảm sau) |

**Vì sao giảm số xe trước, quãng đường sau?** Vì trong thực tế, cắt được **một chiếc xe** (một tài xế, một khoản bảo hiểm, một chi phí cố định cả ngày) đáng giá hơn nhiều so với tiết kiệm vài km xăng. Nguyên tắc "ưu tiên cái quan trọng trước" này trong ngành gọi là **tối ưu theo thứ tự từ điển** (lexicographic).

**Tối ưu ở đâu?** Ở khâu **lập kế hoạch trước khi xe lăn bánh**. Phần mềm nhận vào danh sách khách + khung giờ + tải trọng, rồi trả ra: "Xe 1 đi A→C→F, xe 2 đi B→D→E..." sao cho tổng chi phí thấp nhất.

---

## 3. Quy trình tối ưu diễn ra **ra sao**?

Cách làm cốt lõi giống hệt cách một người thợ khéo léo cải thiện dần một bản nháp. Có thể hình dung qua **5 bước**:

### Bước 1 — Vẽ một bản nháp ban đầu (nhanh, xấu cũng được)
Máy dựng một lời giải thô ban đầu bằng cách "cứ ai gần thì nhét vào xe cho tiện" (gọi là *greedy* — tham lam). Bản nháp này thường thừa xe và đường vòng vèo, nhưng có còn hơn không — ta cần một điểm xuất phát.

### Bước 2 — "Đập đi xây lại" một phần (vòng lặp cải tiến)
Đây là trái tim của thuật toán, tên là **ALNS** (*Adaptive Large Neighborhood Search* — Tìm kiếm lân cận lớn thích ứng). Mỗi vòng lặp máy làm 2 việc:

- **Phá (Destroy):** rút một nhóm khách ra khỏi lộ trình hiện tại — ví dụ "gỡ 15 khách khó chịu nhất ra".
- **Sửa (Repair):** chèn lại nhóm khách đó vào những vị trí mới, tốt hơn.

Cứ phá–sửa hàng nghìn lần. Nếu bản mới rẻ hơn thì giữ; nếu tệ hơn thì **thỉnh thoảng vẫn giữ** để tránh bị "kẹt trong một lối mòn" (giống như đôi khi phải đi lùi một bước để tiến ba bước).

> 💡 **Ví von:** Giống chơi xếp hình Rubik. Bạn không giải một phát ra ngay, mà xoay đi xoay lại, chấp nhận đôi lúc trông có vẻ rối hơn, để cuối cùng ra kết quả gọn.

### Bước 3 — Một "bộ não" học cách ra quyết định thông minh hơn
Điểm đặc biệt của dự án: thay vì phá–sửa một cách **ngẫu nhiên máy móc**, có một **bộ não trí tuệ nhân tạo (AI)** ngồi quan sát và quyết định:

- *"Lúc này nên phá kiểu gì, sửa kiểu gì thì hiệu quả nhất?"*
- *"Đang bị kẹt (không cải thiện được nữa) rồi — có nên đổi chiến thuật không?"*

Bộ não này học từ kinh nghiệm bằng kỹ thuật **Học tăng cường sâu** (giống cách AI học chơi cờ vua: thử, thấy nước nào dẫn tới thắng thì lần sau ưu tiên nước đó). Tên kỹ thuật là **DDQN**. Nhờ vậy cả hệ thống được gọi là **Hybrid (lai) DDQN–ALNS**: *"cơ bắp" ALNS làm việc nặng, "bộ não" DDQN chỉ đạo chiến thuật.*

### Bước 4 — Ghép nối những mảnh tốt nhất
Trong lúc chạy, mỗi khi tìm được một **tuyến đường lẻ đặc biệt đẹp**, máy cất nó vào một "kho tuyến tốt" (*Route Pool*). Định kỳ, máy giải một bài toán nhỏ để **ghép các mảnh tuyến đẹp** lại thành một lời giải tổng thể còn tốt hơn — như ghép những mảnh ghép hoàn hảo từ nhiều bản nháp khác nhau.

### Bước 5 — Đánh bóng lần cuối (Polish)
Trước khi chốt, máy làm một loạt thao tác tinh chỉnh: thử **bỏ nốt một chiếc xe còn dư**, và **duỗi thẳng** các đoạn đường đi vòng. Đây là công đoạn "lau chùi" để vắt kiệt vài phần trăm cuối cùng.

---

## 4. Làm sao **biết** nó thật sự tối ưu? (Đây là câu hỏi quan trọng nhất)

Một tuyên bố "phần mềm của tôi tốt" là vô nghĩa nếu không chứng minh được. Dự án chứng minh bằng **3 lớp bằng chứng**, rất chặt chẽ:

### Bằng chứng 1 — So với "đáp án chuẩn của thế giới" (BKS)
Giới nghiên cứu toàn cầu có những **bộ đề thi chuẩn** dùng chung suốt hàng chục năm (bộ **Solomon** với 100 khách, bộ **Gehring–Homberger** với 200–1000 khách). Mỗi đề này đã có **lời giải tốt nhất từng biết** — gọi là **BKS** (*Best-Known Solution*), do hàng nghìn nhà khoa học trên thế giới cùng đua nhau cải thiện.

→ Ta chỉ cần so kết quả của mình với BKS. Càng gần BKS (khoảng cách gap càng nhỏ, 0% là **chạm đúng kỷ lục thế giới**) thì càng giỏi. Đây là thước đo khách quan, không thể "tự chấm điểm cho mình".

### Bằng chứng 2 — So với chính mình và với đối thủ mạnh
Dự án luôn chạy song song nhiều phiên bản để so đo công bằng:

- **ALNS-Base**: bản không có "bộ não" AI (chỉ có cơ bắp) — để chứng minh AI *có* đóng góp thật.
- **Google OR-Tools**: bộ giải công nghiệp miễn phí nổi tiếng của Google — đối thủ đáng gờm.
- **Hybrid-DDQN**: bản đầy đủ của dự án.

### Bằng chứng 3 — Kiểm định thống kê (để loại trừ "ăn may")
Đây là lớp bằng chứng tinh vi nhất. Vì thuật toán có yếu tố ngẫu nhiên, một lần thắng có thể chỉ là **may mắn**. Nên mỗi cấu hình được chạy **lặp lại nhiều lần**, rồi dùng một phép kiểm định toán học tên là **Wilcoxon** để trả lời: *"Xác suất kết quả tốt hơn này chỉ do ăn may là bao nhiêu?"*

- Nếu con số đó (gọi là **p-value**) **nhỏ hơn 5% (0.05)** → kết luận "thắng thật, không phải ăn may".
- Dự án đạt các mức như *p = 0.0000000593* — tức xác suất ăn may gần như **bằng không**.

> 🔬 **Tinh thần trung thực:** Dự án cũng thẳng thắn ghi nhận **chỗ chưa thắng chắc**. Ví dụ trên một bài toán 400 khách, phần thắng chỉ đạt p = 0.375 (tức *có thể là ăn may*) → nhóm ghi rõ "chưa có ý nghĩa thống kê" thay vì thổi phồng. Chính sự trung thực này làm các con số còn lại đáng tin.

---

## 5. Vậy rốt cuộc tối ưu được **bao nhiêu**?

Đây là những con số nổi bật (so với bản không có AI — ALNS-Base):

### Trên bộ đề 100 khách (Solomon)
- **Giảm ~47% lượng xe dư thừa.** Cụ thể: trung bình bản cũ thừa 0.27 xe/bài, bản mới chỉ còn thừa 0.14 xe/bài. So với Google OR-Tools thì cắt được **~76% xe dư**.
- **Xóa ~83% quãng đường lãng phí.** Khi so công bằng (cùng số xe), phần đường đi dư so với kỷ lục thế giới giảm từ +1.08% xuống chỉ còn **+0.19%** — tức gần chạm kỷ lục thế giới.

### Trên các bài toán siêu lớn (200 – 1000 khách)
Càng đông khách, lợi thế của AI càng rõ:

| Quy mô | Lợi ích so với bản không AI |
| :--- | :--- |
| 200 khách | Đạt số xe tối thiểu **ổn định 100% số lần chạy** (bản cũ chỉ 30–70%) |
| 800 khách | Tiết kiệm trung bình **1.44 xe/bài toán** |
| **1000 khách** | Tiết kiệm **~2.22 xe/bài toán** và giảm **~9.4% tổng quãng đường** |

> 🚀 Ở quy mô 1000 khách, dự án **thắng cả Google OR-Tools** (dùng ít xe hơn, và OR-Tools mất tới ~6.454 giây ≈ 1 giờ 47 phút cho một bài).

### Cái giá phải trả (nói thẳng)
Chất lượng cao hơn **đổi bằng thời gian tính toán**. Bản có AI chạy chậm hơn bản thường khoảng **1.5 đến 4 lần** (trên bài nhỏ) và có thể tới hàng chục lần (trên bài rất lớn). Với logistics lập kế hoạch trước cả đêm thì đây là đánh đổi hoàn toàn xứng đáng — chậm vài phút để cắt được cả một chiếc xe cả ngày.

---

## 6. Đã thêm những "bộ phận core" gì so với bản gốc? Và **tại sao**?

"Bản gốc" ở đây là một bộ giải VRPTW thông thường (chỉ có ALNS thuần). Dự án đã lắp thêm nhiều "bộ phận" mới. Dưới đây là từng cái, kèm **lý do thêm** và **ưu/nhược điểm** — trình bày để người không chuyên vẫn nắm được.

### 🧠 Bộ phận 1 — "Bộ não" AI hai tầng (DDQN)
- **Là gì:** Một AI chỉ huy gồm 2 cấp. *Cấp cao* quyết định chiến lược lớn ("đang bế tắc, đổi chiến thuật đi!"). *Cấp thấp* quyết định chiến thuật chi tiết ("vòng này phá kiểu A, sửa kiểu B").
- **Tại sao thêm:** Bản gốc chọn chiến thuật kiểu "quay xổ số" — dò dẫm. Bộ não học được từ kinh nghiệm nên chọn khôn hơn, thoát bế tắc nhanh hơn.
- **Ưu điểm:** Kết quả **ổn định và tốt hơn**, đặc biệt trên bài lớn.
- **Nhược điểm:** Phức tạp hơn, chạy chậm hơn, và cần công sức để "huấn luyện" AI.

### 🎓 Bộ phận 2 — "Quy tắc chấp nhận học được" (LAC) thay cho công thức cứng
- **Là gì:** Quyết định "có nên tạm giữ một bản nháp tệ hơn để hy vọng về sau tốt hơn không". Bản gốc dùng một công thức vật lý cứng nhắc (Simulated Annealing — mô phỏng quá trình tôi kim loại nguội dần). Dự án thay bằng một AI nhỏ **tự học** khi nào nên chấp nhận.
- **Tại sao thêm:** Công thức cứng không biết thích nghi với từng bài; AI biết.
- **Ưu/nhược:** Linh hoạt hơn ✅ / nhưng lại là một mô hình nữa cần huấn luyện ❌.

### 🗺️ Bộ phận 3 — "Mắt nhìn bản đồ" bằng mạng đồ thị (GNN / GAT)
- **Là gì:** Một mạng nơ-ron **nhìn toàn cảnh bản đồ** khách hàng và **dự đoán trước** những cặp điểm "nhiều khả năng nên nối thẳng với nhau" (như người có kinh nghiệm liếc bản đồ là đoán được tuyến hợp lý).
- **Tại sao thêm:** Giúp máy **khởi đầu thông minh hơn** và **bỏ qua nhanh** những khả năng vô vọng → tiết kiệm thời gian dò tìm.
- **Kết quả đo được:** Trên một số bài, tăng tốc **7% đến 86%**, thậm chí giúp cắt thêm 1 xe.
- **Nhược điểm:** Cần dữ liệu để huấn luyện mạng này; lợi ích không đồng đều giữa các bài.

### 🧩 Bộ phận 4 — "Kho tuyến tốt" + ghép nối tối ưu (Route Pool + MILP)
- **Là gì:** Cất giữ mọi tuyến đường đẹp gặp trong lúc chạy, rồi định kỳ **ghép các mảnh đẹp nhất** lại bằng một bộ giải chính xác.
- **Tại sao thêm:** Đây chính là "vũ khí" giúp **cắt giảm số xe** — mục tiêu ưu tiên số 1.

### ⚖️ Bộ phận 5 — Cân bằng công việc cho tài xế (Đa mục tiêu / Pareto)
- **Là gì:** Bản gốc chỉ quan tâm xe + quãng đường, nên có thể xảy ra cảnh **một tài xế chạy 10 tiếng, người khác chỉ 2 tiếng**. Bộ phận mới đo thêm độ *chênh lệch tải giữa các tài xế* và *rủi ro trễ hạn*.
- **Tại sao thêm:** Để doanh nghiệp có thể chọn: *"tiết kiệm xăng tối đa"* hay *"chia việc công bằng cho anh em tài xế"*. Đây là nhu cầu **thực tế** mà bài toán học thuật thuần túy bỏ qua.

### ⚡ Bộ phận 6 — Chèn đơn hàng mới tức thì (Dynamic Insertion)
- **Là gì:** Khi đang giao mà có **đơn mới phát sinh trong ngày**, hệ thống nhét đơn đó vào lộ trình đang chạy trong **dưới 1 phần nghìn giây**, thay vì phải tính lại từ đầu (mất 10–30 giây).
- **Tại sao thêm:** Vì thực tế đơn hàng đến liên tục, không thể dừng cả đội xe lại để tính lại.

### 🖥️ Bộ phận 7 — Cổng điều phối trực quan trên web + cập nhật trực tiếp
- **Là gì:** Một trang web hiển thị **bản đồ lộ trình từng xe theo màu**, biểu đồ khung giờ, và **tiến độ chạy cập nhật thời gian thực** (công nghệ SSE — đẩy dữ liệu về trình duyệt ngay, không cần bấm tải lại).
- **Tại sao thêm:** Biến một thuật toán trong phòng thí nghiệm thành **công cụ người điều phối thật sự dùng được**.

### 🏗️ Bộ phận 8 — Nền tảng để tăng tốc bằng C++/Rust (cho tương lai)
- **Là gì:** Một "ổ cắm" (`cpp_hooks.py`) sẵn sàng cắm thêm phần lõi viết bằng ngôn ngữ siêu nhanh (C++/Rust) khi cần chạy quy mô công nghiệp.
- **Trạng thái hiện tại:** Mới là **khung chờ sẵn** — nếu chưa cắm lõi native thì hệ thống tự động dùng bản Python. Đây là chuẩn bị cho tương lai, chưa phải tăng tốc đã có.

---

## 7. Tóm tắt trong một trang

| Câu hỏi của bạn | Câu trả lời ngắn gọn |
| :--- | :--- |
| **Tối ưu cái gì?** | Số xe (NV) và tổng quãng đường (TD) khi giao hàng, không vi phạm khung giờ & tải trọng. |
| **Tối ưu ở đâu?** | Ở khâu lập kế hoạch lộ trình trước khi xe chạy. |
| **Quy trình ra sao?** | Vẽ nháp → phá–sửa hàng nghìn lần → AI chỉ đạo chiến thuật → ghép mảnh tốt → đánh bóng. |
| **Sao biết nó tối ưu?** | So với kỷ lục thế giới (BKS), so với Google OR-Tools, và kiểm định thống kê chống ăn may. |
| **Tối ưu bao nhiêu?** | Cắt ~47% xe dư và ~83% đường lãng phí (bài 100 khách); tiết kiệm tới ~2.2 xe & ~9.4% đường (bài 1000 khách). |
| **Thêm core gì?** | Bộ não AI (DDQN), quy tắc chấp nhận học được (LAC), mắt nhìn bản đồ (GNN/GAT), kho ghép tuyến (MILP), cân bằng tài xế, chèn đơn tức thì, web trực quan, khung tăng tốc C++. |
| **Tại sao thêm?** | Để chọn chiến thuật khôn hơn, cắt được nhiều xe hơn, chạy ổn định hơn, và dùng được trong thực tế. |
| **Nhược điểm chính?** | Phức tạp hơn và chạy chậm hơn (1.5–4 lần) — đổi thời gian lấy chất lượng. |

---

### 📌 Một câu để nhớ
> **Bản gốc là một người thợ cần cù dò dẫm. Dự án này lắp cho người thợ đó một "bộ não biết học", một "đôi mắt nhìn bản đồ", và một "bộ đồ nghề tinh chỉnh" — nhờ đó giải nhanh hơn, tiết kiệm hơn, và quan trọng nhất: chứng minh được bằng số liệu rằng nó thật sự giỏi hơn, chứ không phải nói suông.**
