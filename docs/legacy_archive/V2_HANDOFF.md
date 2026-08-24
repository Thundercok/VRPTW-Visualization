# V2 Upgrade — Bàn giao & Tóm tắt phiên làm việc (2026-07-24 → 07-27)

Tài liệu này tóm tắt toàn bộ đợt nâng cấp V2: cải tiến code, chạy lại benchmark,
cập nhật paper, và các sự cố hạ tầng đã xử lý. Đọc file này trước khi resume.

Tài liệu liên quan: `docs/RERUN_CHECKLIST.md` (chi tiết từng thay đổi + kiểm chứng),
`plan.md` (kế hoạch gốc), `CLAUDE.md` (handoff cũ).

---

## 1. Mục tiêu

Từ `plan_estimated` (không thay kiến trúc DDQN/ALNS, chỉ mở rộng) + `plan.md`
(chạy lại toàn bộ benchmark bằng code mới): **giảm thời gian và tăng độ chính xác
so với baseline**, rồi sinh lại số liệu paper và xác minh có thật sự cải tiến.

Quy trình: sàng lọc trên A/B ghép cặp → triển khai cái sống sót → sweep sản xuất
16h → cập nhật paper. Bài học chi phối: **mọi thay đổi chất lượng phải đánh giá ở
cùng thời gian tường (iso-time), không phải cùng số vòng lặp** (4 ý tưởng V1 từng
thắng ở equal-iters nhưng thua iso-time).

---

## 2. Cải tiến code đã triển khai

### Tầng A — lợi ích cơ học, giống hệt từng bit (verified bằng golden)
| Mã | Nội dung | File |
|---|---|---|
| A1+A2 | `_PlanCache` cập nhật tăng dần (tái dùng theo content-key), bỏ Python `set`/`dict` → mảng | `local_search.py` |
| A3 | Memo hoá kết quả quét theo cặp (`scan_memo`), tái dùng qua các move không chạm tuyến đó. **Chọn thay cho don't-look bits** vì DLB đổi quỹ đạo, memo thì không | `local_search.py` |
| A4 | PER duy trì `priorities**alpha` tăng dần (`_pri_alpha`); **bỏ sum-tree** vì không bit-identical | `rl.py` |
| A5 | Pha đuôi biết deadline — mọi pha kiểm `_out_of_time()`, `td_converge_polish` nhận deadline | `solvers.py`, `local_search.py` |
| A6 | Greedy repair (`op_greedy`/`op_tw_greedy`) dùng ma trận + refresh cột (như `_regret`) | `operators.py` |
| A7 | Cache đặc trưng trạng thái RL theo danh tính `cur` (`_StructuralFeatureCache`) | `solvers.py` |
| A8 | Pin 1 luồng/worker khi chạy song song + cờ `--time-limit`/`--no-time-limit` | `run_benchmark.py` |

### Tầng B — đổi quỹ đạo, nhắm số xe (qua cổng G3 iso-time)
| Mã | Nội dung | Bằng chứng |
|---|---|---|
| B1 | **Guided Ejection Search** (`_guided_ejection_search`): ejection pool LIFO + bộ đếm phạt theo khách + chọn ejection cực tiểu Σp + pha nhiễu loạn; gọi khi beam search bó tay | RC105 chạm sàn BKS 13 xe 4/5 seed (baseline 0/5) |
| B2 | **SREX crossover** thay crossover cũ (cũ tạo con *vụn*, 0/600 qua cổng `nv<best`; SREX 491/600) | `rl.py::EliteArchive.crossover` |
| B3 | Chọn cột SP theo giao chặt (`_select_milp_columns`) + cache MILP. Bản rộng min_cover=3 **đã bác bỏ** (mất 1 xe rc1_2_1) → dùng bản chỉ cứu ca `row_sums==0` | `pool.py` |

### Đã thử và bác bỏ bằng đo đạc (đừng thêm lại nếu không có iso-time win)
Don't-look bits (đổi quỹ đạo, memo đạt cùng mục tiêu bit-identical) · sum-tree PER
(khác RNG stream) · SP reshuffle rộng (mất 1 xe rc1_2_1). Cùng 4 ý tưởng V1: lọc
kNN, slack FTS, SISR, lịch răng cưa.

### Golden: tái-chuẩn dưới numpy 2.3.5
Golden fingerprint (`tests/golden/baseline.json`) tái tạo. Sau đó numpy bị cài lại
2.3.5 lúc gỡ ortools → 13/16 golden trôi (vd r1_2_1 ALNS NV 22→21). **Không phải
lỗi code** (ALNS-Base không dùng path tôi sửa) mà do numpy đổi version. Đã tái-chuẩn
golden dưới 2.3.5 → **49/49 xanh**. Đã ghim `numpy>=1.26,<2.4` (`requirements.txt` +
`pyproject.toml`).

---

## 3. Sweep sản xuất — hai giao thức

`scripts/run_rerun_sweeps.sh`. Đã chạy xong sạch **8/8 shard, 984 dòng, 0 null**.

| Giao thức | Shard | Output |
|---|---|---|
| Giới hạn vòng lặp (`--no-time-limit`) | Solomon×3 (5000 iter), H200 (800), H400 (600) | `results/rerun_iters/` |
| Giới hạn thời gian (anytime 0.6s×n) | H600, H800, H1000 | `results/rerun_time/` |

Gộp: `results/rerun_combined.csv`. **Không dùng GNN** (heatmap không cải thiện chất lượng).

Cầu chì an toàn (thêm sau sự cố ngủ): solomon-wide 1200s, H200 1500s, H400 3600s —
cao gấp ≥2.5× run khỏe mạnh nên chỉ cắt runaway, không đổi kết quả.

---

## 4. Kết quả (trung thực)

### Số xe (mục tiêu hạng nhất) — CẢI THIỆN
- Solomon DDQN NV diff **+0.089** (cũ +0.143); hơn ALNS 65%, hơn OR-Tools 83%.
  Wilcoxon vs ALNS p=1.78e-3, vs OR-Tools p=2.96e-5.
- **DDQN = Rule về NV** (p=1.0, giống hệt 56/56) → đóng góp RL là ở khoảng cách.
- Điểm nhấn: RC101 đạt sàn BKS **14 xe** (ALNS 15, OR-Tools 16); rc1_2_1 đạt **18=BKS**
  (khác đều 19); R101 đạt đúng **BKS distance 1650.80**.
- Quy mô lớn: n=1000 DDQN **60.55 xe** vs OR-Tools **70.00** ở iso-time (~400s) —
  hơn **9.45 xe**, và cách biệt tăng theo quy mô (+1.34/+4.92/+8.50/+9.45 tại n=400/600/800/1000).

### Khoảng cách — ĐÁNH ĐỔI (không phải Pareto win)
- Giao chặt N=40: DDQN **+0.575%** — tốt nhất trong 4 thuật toán (ALNS 1.642, Fixed
  0.707, Rule 0.734).
- **NHƯNG** kém bản cũ (+0.185%): đánh đổi xe/khoảng cách — giảm xe thì đường dài hơn.
- Gap thô "xấu đi" ở R1/R2/RC1 là **hệ quả số học của ít xe**, không phải thụt lùi.
- **RC2** là điểm âm thật: cùng số xe nhưng gap +0.46pp (họ GES không giúp).

### Tốc độ
- Lõi (Tầng A) nhanh, giống hệt từng bit. GES thêm ~20% thời gian đuôi trên Solomon
  — chính phần đó mua lại số xe. Đây là đánh đổi có chủ đích, đã ghi trong paper.

---

## 5. Cập nhật paper (`docs/paper.tex` → `docs/paper.pdf`)

6 bảng sinh tự động vào `docs/tables/*.tex` (bằng `scripts/make_paper_tables.py`),
`\input` vào paper nên **lần sau sinh lại tự đồng bộ**.

Văn xuôi cập nhật: abstract, ablation (NV giảm đơn điệu / TD thô là artifact /
giao chặt DDQN tốt nhất), NV-summary, distance, GH, head-to-head OR-Tools.

**Ba mâu thuẫn nội dung đã sửa (không chỉ thay số):**
1. **GNN** — viết lại thành kết quả *khả-mở-rộng* (bộ nhớ 1517→1.3 MB, latency
   3.96→0.049s) + *ablation âm trung thực* (guidance không cải thiện chất lượng,
   gap +0.22pp, p=0.683). Bỏ mọi tuyên bố "giảm xe/tăng tốc" sai.
2. **Tường thuật tốc độ** — thừa nhận GES đánh đổi thời gian lấy số xe.
3. **DDQN vs Rule** — hạ xuống "biên marginal p=0.064" (không còn phóng đại <0.05).

Nhãn OR-Tools "120s" → "iso-time" (đúng thực tế: OR-Tools = 0.95×thời gian DDQN).

**PDF:** biên dịch sạch bằng MiKTeX 25.12 (đã cài trong phiên), 10 trang, 0 undefined ref.

---

## 6. Sự cố hạ tầng đã xử lý (4 loại, đều có phòng vệ)

1. **Encoding** — console cp1258 không mã hoá được ký tự `✓`/`─` → crash. Fix:
   `PYTHONIOENCODING=utf-8` trong driver + ASCII trong `compare_sweeps.py`.
2. **ortools thiếu** — mọi dòng OR-Tools FAILED. Fix: `pip install ortools`; pip kéo
   numpy 2.4.6 làm hỏng numba → ghim `numpy<2.4` (2.3.5). `ORTOOLS_OK=True`.
3. **Máy ngủ 2 lần** (Modern Standby, `powercfg` không đủ) → run ngốn 22h/8000s wall
   nhưng ~0 CPU, làm nhiễu Time_s và khuếch đại ngân sách OR-Tools. Fix:
   `scripts/keep_awake.ps1` (SetThreadExecutionState, chủ động, không cần admin).
   **nv/cost luôn an toàn — chỉ Time_s bị nhiễu; checkpoint không mất dữ liệu.**
4. **Lỗi code `_deadline`** — `HybridFixedSolver.solve()`/`HybridRuleSolver.solve()`
   thiếu tham số `_deadline` mà A5 thêm vào lớp cha → crash `TypeError` khi split RL
   kích hoạt (instance ≥200 khách + có deadline). Ẩn vì golden/A-B dùng
   `split_enabled=False`. Fix: thêm `_deadline` vào 2 override; repro
   `scratchpad/repro_split.py`.

---

## 7. File thay đổi / tạo mới

**Code solver:** `src/vrptw/{local_search,operators,pool,rl,solvers}.py`
**Runner:** `docs/run_benchmark.py`, `run_full_production.sh`
**Script mới:** `scripts/{compare_sweeps,make_paper_tables,run_rerun_sweeps.sh,keep_awake.ps1}`
**Test:** `tests/{test_ges.py (mới), test_crossover_isolated.py, golden/baseline.json}`
**Paper:** `docs/paper.tex`, `docs/tables/*.tex`, `docs/paper.pdf`, `docs/RERUN_CHECKLIST.md`
**Deps:** `requirements.txt`, `pyproject.toml` (numpy<2.4)
**Kết quả:** `results/rerun_iters/`, `results/rerun_time/`, `results/rerun_combined.csv`

Toàn bộ đang ở working tree — **chưa commit** (theo nguyên tắc chỉ commit khi được yêu cầu).

---

## 8. Cách resume / việc còn có thể làm

- **Kiểm chứng nhanh:** `python -m pytest tests/ -q --ignore=tests/e2e` (phải 49/49).
- **Biên dịch lại PDF:** `pdflatex -interaction=nonstopmode docs/paper.tex` ×3
  (MiKTeX tại `C:\Users\han\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`).
- **Sinh lại bảng nếu chạy lại sweep:** `python scripts/make_paper_tables.py --sweep results/rerun_combined.csv --out-dir docs/tables`.
- **So cũ/mới:** `python scripts/compare_sweeps.py <old.csv> results/rerun_combined.csv --algorithms Hybrid-DDQN` (cần `PYTHONIOENCODING=utf-8`).
- **Nếu chạy lại sweep dài:** nhớ chạy `scripts/keep_awake.ps1` để máy không ngủ.

### Việc còn để ngỏ (tuỳ chọn)
- Commit toàn bộ V2 (khi bạn quyết định).
- Kiểm bảng giao chặt N=40 có đơn điệu ALNS→Fixed→Rule→DDQN không (Fixed 0.707 <
  Rule 0.734 — không hoàn toàn đơn điệu, nhưng DDQN tốt nhất; paper đã diễn giải).
- Nếu muốn GNN thành quả chất lượng: hiệu chỉnh lại `gnn_guidance_strength` cho
  kiến trúc thưa (hiện 0.45 tuned cho dense) — hiện paper báo cáo GNN là kết quả
  khả-mở-rộng, không phải chất lượng.
