"""
Tkinter GUI to load an image, choose K and algorithms, run and show comparisons.

Responsive UI:
- Lưới thẻ (cards) bên trái tự tính số cột & kích thước thumbnail theo chiều rộng -> lấp đầy khoảng trống.
- Non-blocking: chạy nền bằng thread, cập nhật UI bằng after().
- Tabs bên phải: Scores | Histogram | Difference + chọn 'Diff reference' và metric (FE/PSNR/SSIM/Time).
- Single-K & Multi-K:
    * MFWOA: chạy đa nhiệm thật (nhiều K trong 1 lần).
    * Otsu/FCM/PSO/GA/WOA: chạy nhiều lần đơn nhiệm cho từng K.
- Tính năng: Zoom thumbnail (click), fullscreen (nút + F11/Esc), stability (30 seeds).

[CHANGED] Bổ sung:
- Tab Difference có combobox A/B để so sánh tự do giữa mọi thuật toán/K (kể cả khác K).
- Toolbar cho tab Difference.
- Logging per-seed: truyền seed/image_stem/noise_tag xuống compare runner, lưu CSV + mask + overlay.
"""
from __future__ import annotations
import csv, datetime, time, platform
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

import cv2
import numpy as np

# === algo utils ===
from src.cli.compare import run_algorithms_on_image
try:
    from src.cli.compare import run_mfwoa_multitask_on_image as compare_run_mf_multitask
except Exception:
    compare_run_mf_multitask = None

from src.metrics.metrics import psnr, ssim
from src.metrics.fuzzy_entropy import histogram_from_image, compute_fuzzy_entropy
from src.optim.mfwoa_multitask import mfwoa_multitask
from src.optim.woa import woa_optimize
from src.optim.pso import pso_optimize
from src.optim.ga import ga_optimize
from src.optim.fcm import fcm_thresholding
from src.seg.thresholding import apply_thresholds

try:
    from src.metrics.metrics import dice_macro  # nếu bạn đã thêm vào metrics.py trước đó
except Exception:
    dice_macro = None
# ============== Styling helpers ==============
def init_styles(root: tk.Tk):
    style = ttk.Style(root)
    for theme in ('vista', 'clam', 'alt', 'default'):
        try:
            style.theme_use(theme)
            break
        except tk.TclError:
            continue
    style.configure('Title.TLabel', font=('Segoe UI Semibold', 11))
    style.configure('Metric.TLabel', font=('Consolas', 9))
    style.configure('Small.TLabel', font=('Segoe UI', 9))
    style.configure('Toolbar.TButton', padding=(8, 4))
    style.configure('Toolbar.TCheckbutton', padding=(4, 2))
    style.configure('Toolbar.TRadiobutton', padding=(4, 2))
    style.configure('Toolbar.TLabel', padding=(6, 2))
    style.configure('Card.TLabelframe', padding=(6, 6))
    style.configure('Card.TLabelframe.Label', font=('Segoe UI Semibold', 10))
    style.configure('TNotebook.Tab', padding=(10, 4))
    return style


# ============== Algo helpers ==============
def _visualize_midpoint(img_gray: np.ndarray, thresholds: list[int]) -> np.ndarray:
    labels = apply_thresholds(img_gray, thresholds)
    bins = [0] + sorted(thresholds) + [255]
    mids = [int(round((bins[i] + bins[i+1]) / 2.0)) for i in range(len(bins)-1)]
    out = np.zeros_like(img_gray, dtype=np.uint8)
    for cls_id, mid in enumerate(mids):
        out[labels == cls_id] = mid
    return out


def _run_single_algo_for_K(img: np.ndarray, algo: str, K: int, pop: int = 30, iters: int = 100) -> dict:
    hist = histogram_from_image(img)
    def fitness_from_pos(pos):
        ths = [int(round(x)) for x in pos]
        return compute_fuzzy_entropy(hist, ths, for_minimization=True)

    t0 = time.perf_counter()
    if algo == 'Otsu':
        levels = np.arange(256)
        prob = hist / (hist.sum() + 1e-12)
        def bc_var(thresholds):
            bins = [0] + sorted(thresholds) + [256]
            total_mean = (prob * levels).sum()
            v = 0.0
            for i in range(len(bins)-1):
                a, b = bins[i], bins[i+1]
                p = prob[a:b].sum()
                if p <= 0:
                    continue
                mean = (prob[a:b] * levels[a:b]).sum()/(p+1e-12)
                v += p*(mean-total_mean)**2
            return float(v)
        thresholds, cand = [], list(range(1,255))
        for _ in range(K):
            best_t, best_s = None, -1.0
            for t in cand:
                s = bc_var(thresholds+[t])
                if s > best_s: best_s, best_t = s, t
            if best_t is None: break
            thresholds.append(best_t)
        ths = sorted(thresholds)
    elif algo == 'FCM':
        ths = fcm_thresholding(hist, K)
    elif algo == 'PSO':
        ths, _ = pso_optimize(hist, K, pop_size=pop, iters=iters, objective=fitness_from_pos)
    elif algo == 'GA':
        ths, _ = ga_optimize(hist, K, pop_size=pop, iters=iters, objective=fitness_from_pos)
    elif algo == 'WOA':
        ths, _ = woa_optimize(hist, K, pop_size=pop, iters=iters, objective=fitness_from_pos)
    else:
        raise ValueError(f"Unknown algo {algo}")

    elapsed = time.perf_counter() - t0
    vis = _visualize_midpoint(img, ths)
    fe = compute_fuzzy_entropy(hist, ths, for_minimization=False)
    return {"K": int(K), "thresholds": [int(v) for v in ths], "time": float(elapsed),
            "vis": vis, "fe": float(fe), "fitness": float(-fe)}


def _fallback_run_mfwoa_multitask_on_image(img: np.ndarray, Ks: list[int], pop: int, iters: int) -> dict:
    hist = histogram_from_image(img)
    hists = [hist for _ in Ks]
    t0 = time.perf_counter()
    best_ths_list, best_scores_list, mf_diag = mfwoa_multitask(hists, Ks, pop_size=pop, iters=iters)
    elapsed = time.perf_counter() - t0

    out = {}
    for K_i, ths, fe in zip(Ks, best_ths_list, best_scores_list):
        vis = _visualize_midpoint(img, ths)
        out[f"MFWOA_K{int(K_i)}"] = {
            "K": int(K_i), "thresholds": [int(v) for v in ths], "time": float(elapsed),
            "vis": vis, "fe": float(fe), "fitness": float(-fe), "mf_diag": mf_diag
        }
    return out


# ============== App ==============
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        init_styles(root)
        root.title('MFWOA Segmentation demo')
        root.minsize(1100, 640)

        # Fullscreen & maximize on start
        self._prev_geom = None
        self._maximize_start()
        root.bind('<F11>', self._toggle_fullscreen)
        root.bind('<Escape>', self._exit_fullscreen)

        self.img = None
        self.img_paths: list[str] = []
        self.results_store = {}
        self._last_results: dict | None = None
        self._last_img: np.ndarray | None = None

        # responsive grid state
        self.cards: dict[str, dict] = {}      # key -> {'frame','img_lbl','score_lbl','tkimg'}
        self._thumb_px = 320                   # current thumbnail size
        self._card_cols = 3                    # current columns
        self._card_min_w = 260                 # min width for one card
        self._card_pad = 12                    # padding used in layout

        # --- Toolbar (2 rows) ---
        self._build_toolbar()

        # --- Main split ---
        self.frame = ttk.Frame(root)
        self.frame.pack(fill='both', expand=True)
        self.frame.grid_rowconfigure(0, weight=1)
        self._build_left_panel()
        self._build_right_panel()
        self._build_statusbar()

        # default grid: single-K
        self._rebuild_results_grid(['Otsu', 'FCM', 'PSO', 'GA', 'WOA', 'MFWOA'])

    # ---------- UI building ----------
    def _build_toolbar(self):
        # Row 1
        bar1 = ttk.Frame(self.root); bar1.pack(fill='x', padx=10, pady=(8, 2))
        ttk.Button(bar1, text='Open Image(s)', style='Toolbar.TButton', command=self.open_image).pack(side='left')

        algo_box = ttk.Frame(bar1); algo_box.pack(side='left', padx=(14, 0))
        ttk.Label(algo_box, text='Algorithms:', style='Toolbar.TLabel').pack(side='left')
        self.algos_vars = {}
        for name in ['Otsu', 'FCM', 'PSO', 'GA', 'WOA', 'MFWOA']:
            v = tk.BooleanVar(value=True)
            ttk.Checkbutton(algo_box, text=name, variable=v, style='Toolbar.TCheckbutton').pack(side='left')
            self.algos_vars[name] = v

        self.pop_var = tk.IntVar(value=40)
        self.iters_var = tk.IntVar(value=100)
        ttk.Label(bar1, text='pop:', style='Toolbar.TLabel').pack(side='left', padx=(16, 2))
        ttk.Spinbox(bar1, from_=10, to=1000, textvariable=self.pop_var, width=6).pack(side='left')
        ttk.Label(bar1, text='iters:', style='Toolbar.TLabel').pack(side='left', padx=(8, 2))
        ttk.Spinbox(bar1, from_=20, to=5000, increment=20, textvariable=self.iters_var, width=7).pack(side='left')

        ttk.Button(bar1, text='Run', style='Toolbar.TButton', command=self.run).pack(side='left', padx=8)
        ttk.Button(bar1, text='Export Results', style='Toolbar.TButton', command=self.export_results).pack(side='left', padx=4)

        # Row 2
        bar2 = ttk.Frame(self.root); bar2.pack(fill='x', padx=10, pady=(0, 6))
        mode_frame = ttk.Frame(bar2); mode_frame.pack(side='left')
        ttk.Label(mode_frame, text='Mode:', style='Toolbar.TLabel').pack(side='left')
        self.mode_var = tk.StringVar(value='single')
        ttk.Radiobutton(mode_frame, text='Single K', value='single', variable=self.mode_var,
                        style='Toolbar.TRadiobutton', command=self._toggle_mode_controls).pack(side='left')
        ttk.Radiobutton(mode_frame, text='Multi-K', value='multi', variable=self.mode_var,
                        style='Toolbar.TRadiobutton', command=self._toggle_mode_controls).pack(side='left')

        self.single_frame = ttk.Frame(bar2); self.single_frame.pack(side='left', padx=(10, 0))
        ttk.Label(self.single_frame, text='K:', style='Toolbar.TLabel').pack(side='left')
        self.k_var = tk.IntVar(value=3)
        ttk.Spinbox(self.single_frame, from_=1, to=10, textvariable=self.k_var, width=5).pack(side='left')

        self.multi_frame = ttk.Frame(bar2); self.multi_frame.pack_forget()
        ttk.Label(self.multi_frame, text='Ks (CSV):', style='Toolbar.TLabel').pack(side='left')
        self.ks_var = tk.StringVar(value='2,3,4,5,6')
        ttk.Entry(self.multi_frame, textvariable=self.ks_var, width=16).pack(side='left')

        right_block = ttk.Frame(bar2); right_block.pack(side='right')
        ttk.Label(right_block, text='Noise:', style='Toolbar.TLabel').pack(side='left')
        self.noise_var = tk.StringVar(value='none')
        ttk.Combobox(right_block, textvariable=self.noise_var,
                     values=['none', 'gaussian', 'salt-pepper'], width=12, state='readonly').pack(side='left')
        ttk.Label(right_block, text='Amt %', style='Toolbar.TLabel').pack(side='left', padx=(6, 2))
        self.noise_amt = tk.DoubleVar(value=0.0)
        ttk.Spinbox(right_block, from_=0.0, to=5.0, increment=0.1, textvariable=self.noise_amt, width=6).pack(side='left', padx=(0, 8))
        self.stability_btn = ttk.Button(right_block, text='Stability (30 seeds)', style='Toolbar.TButton',
                                        command=self.run_stability)
        self.stability_btn.pack(side='left', padx=(0, 10))
        self.full_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(right_block, text='Fullscreen', variable=self.full_var,
                        command=self._on_fullscreen_toggle).pack(side='left')

    def _build_left_panel(self):
        self.left_container = ttk.Frame(self.frame)
        self.left_container.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)

        self.left_listbox = tk.Listbox(self.left_container, height=5)
        self.left_listbox.grid(row=0, column=0, columnspan=2, sticky='ew')

        self.scroll_canvas = tk.Canvas(self.left_container, highlightthickness=0, background='#F7F8FA')
        self.vscroll = ttk.Scrollbar(self.left_container, orient='vertical', command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=self.vscroll.set)
        self.scroll_canvas.grid(row=1, column=0, sticky='nsew', pady=(8,0))
        self.vscroll.grid(row=1, column=1, sticky='ns', pady=(8,0))

        self.left_container.grid_rowconfigure(1, weight=1)
        self.left_container.grid_columnconfigure(0, weight=1)

        self.cards_frame = ttk.Frame(self.scroll_canvas)
        self.cards_window = self.scroll_canvas.create_window((0, 0), window=self.cards_frame, anchor='nw')
        self.cards_frame.bind('<Configure>', lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox('all')))
        self.scroll_canvas.bind('<Configure>', self._on_canvas_resize)
        self.scroll_canvas.bind_all('<MouseWheel>', lambda e: self.scroll_canvas.yview_scroll(int(-1*(e.delta/120)), 'units'))

    def _build_right_panel(self):
        self.right_container = ttk.Frame(self.frame)
        self.right_container.grid(row=0, column=1, sticky='nsew', padx=8, pady=8)
        self.frame.grid_columnconfigure(0, weight=3)
        self.frame.grid_columnconfigure(1, weight=2)

        right_top = ttk.Frame(self.right_container); right_top.pack(fill='x', pady=(0,4))
        ttk.Label(right_top, text='Diff reference:').pack(side='left')
        self.ref_algo_var = tk.StringVar(value='MFWOA')
        self.ref_algo_combo = ttk.Combobox(right_top, textvariable=self.ref_algo_var,
                                           values=['MFWOA','WOA','GA','PSO','FCM','Otsu'],
                                           state='readonly', width=16)
        self.ref_algo_combo.pack(side='left', padx=6)
        self.ref_algo_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_tabs())
        ttk.Label(right_top, text='Metric:').pack(side='left', padx=(12,2))
        self.metric_var = tk.StringVar(value='FE')
        ttk.Combobox(right_top, textvariable=self.metric_var,
                     values=['FE','PSNR','SSIM','Time'], state='readonly', width=7).pack(side='left')
        ttk.Button(right_top, text='Redraw',
                   command=lambda: self.update_diagnostics(self._last_results or {}, self._last_img)).pack(side='left', padx=6)

        self.nb = ttk.Notebook(self.right_container); self.nb.pack(fill='both', expand=True)

        # --- Scores tab ---
        tab_scores = ttk.Frame(self.nb); self.nb.add(tab_scores, text='Scores')
        self.conv_fig = Figure(figsize=(5, 3), constrained_layout=True)
        self.conv_ax = self.conv_fig.add_subplot(111)
        self.conv_canvas = FigureCanvasTkAgg(self.conv_fig, master=tab_scores)
        self.conv_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.conv_toolbar = NavigationToolbar2Tk(self.conv_canvas, tab_scores)
        self.conv_toolbar.update()

        # --- Histogram tab ---
        tab_hist = ttk.Frame(self.nb); self.nb.add(tab_hist, text='Histogram')
        self.hist_fig = Figure(figsize=(5, 3), constrained_layout=True)
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=tab_hist)
        self.hist_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.hist_toolbar = NavigationToolbar2Tk(self.hist_canvas, tab_hist); self.hist_toolbar.update()

        # --- Difference tab (with A/B selectors) ---
        tab_diff = ttk.Frame(self.nb); self.nb.add(tab_diff, text='Difference')

        diff_bar = ttk.Frame(tab_diff)
        diff_bar.pack(fill='x', pady=(2, 4))
        ttk.Label(diff_bar, text='A:').pack(side='left')
        self.diffA_var = tk.StringVar(value='')
        self.diffA_combo = ttk.Combobox(diff_bar, textvariable=self.diffA_var,
                                        state='readonly', width=20)
        self.diffA_combo.pack(side='left', padx=(2, 8))
        ttk.Label(diff_bar, text='B:').pack(side='left')
        self.diffB_var = tk.StringVar(value='')
        self.diffB_combo = ttk.Combobox(diff_bar, textvariable=self.diffB_var,
                                        state='readonly', width=20)
        self.diffB_combo.pack(side='left', padx=(2, 8))
        ttk.Button(diff_bar, text='Swap', command=self._swap_diff_pair).pack(
            side='left', padx=(0, 6))
        for cb in (self.diffA_combo, self.diffB_combo):
            cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_tabs())

        self.diff_fig = Figure(figsize=(5, 3), constrained_layout=True)
        self.diff_ax = self.diff_fig.add_subplot(111)
        self.diff_canvas = FigureCanvasTkAgg(self.diff_fig, master=tab_diff)
        self.diff_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.diff_toolbar = NavigationToolbar2Tk(self.diff_canvas, tab_diff)
        self.diff_toolbar.update()

    def _build_statusbar(self):
        status = ttk.Frame(self.root); status.pack(fill='x', padx=8, pady=(0,8))
        self.progress = ttk.Label(status, text='Idle'); self.progress.pack(side='left')
        self.pb = ttk.Progressbar(status, mode='indeterminate', length=160); self.pb.pack(side='left', padx=8)

    # ---------- layout / window helpers ----------
    def _maximize_start(self):
        try:
            if platform.system() == 'Windows':
                self.root.state('zoomed')
            elif platform.system() == 'Darwin':
                pass
            else:
                self.root.attributes('-zoomed', True)
        except Exception:
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            self.root.geometry(f'{sw}x{sh}+0+0')

    def _on_fullscreen_toggle(self):
        fs = self.full_var.get()
        try:
            if fs:
                try: self._prev_geom = self.root.geometry()
                except Exception: self._prev_geom = None
                self.root.attributes('-fullscreen', True)
            else:
                self.root.attributes('-fullscreen', False)
                if self._prev_geom:
                    try: self.root.geometry(self._prev_geom)
                    except Exception: pass
                self._maximize_start()
        except Exception:
            pass

    def _toggle_fullscreen(self, event=None):
        self.full_var.set(not self.full_var.get())
        self._on_fullscreen_toggle()

    def _exit_fullscreen(self, event=None):
        if self.full_var.get():
            self.full_var.set(False)
            self._on_fullscreen_toggle()

    # --- Responsive grid core ---
    def _on_canvas_resize(self, event):
        self.scroll_canvas.itemconfig(self.cards_window, width=event.width)
        self._relayout_cards()

    def _relayout_cards(self):
        """Tính lại số cột & kích thước thumbnail để lấp đầy chỗ trống."""
        cw = max(1, self.scroll_canvas.winfo_width())
        n = len(self.cards)
        if n == 0:
            return
        max_cols = max(1, cw // (self._card_min_w + self._card_pad))
        cols = min(max_cols, max(1, n))
        self._card_cols = cols
        thumb = int((cw - self._card_pad * (cols + 1)) / cols)
        thumb = max(180, min(thumb, 600))  # clamp
        self._thumb_px = thumb

        # đặt lại grid cho các card
        for i, (name, entry) in enumerate(self.cards.items()):
            r, c = divmod(i, cols)
            entry['frame'].grid_configure(row=r, column=c, padx=8, pady=8, sticky='nsew')
            self.cards_frame.grid_columnconfigure(c, weight=1)

        # nếu đã có kết quả thì render lại thumbnail theo kích cỡ mới
        if self._last_results:
            self._rerender_thumbs()

    def _render_thumb(self, img_np: np.ndarray) -> ImageTk.PhotoImage:
        pil = Image.fromarray(img_np).resize((self._thumb_px, self._thumb_px))
        return ImageTk.PhotoImage(pil.convert('L'))

    def _rerender_thumbs(self):
        for key, entry in self.cards.items():
            vis = None
            if self._last_results:
                vis = self._last_results.get(key, {}).get('vis')
            if vis is None:
                continue
            tkimg = self._render_thumb(vis)
            entry['img_lbl'].image = tkimg
            entry['img_lbl'].configure(image=tkimg)

    # ---------- mode / grid ----------
    def _toggle_mode_controls(self):
        if self.mode_var.get() == 'multi':
            self.single_frame.pack_forget()
            self.multi_frame.pack(side='left', padx=(10,0))
        else:
            self.multi_frame.pack_forget()
            self.single_frame.pack(side='left', padx=(10,0))

    def _picked_algorithms(self) -> list[str]:
        return [name for name, var in self.algos_vars.items() if var.get()]

    def _rebuild_results_grid(self, names: list[str], columns_fallback: int = 3):
        # clear
        for child in list(self.cards_frame.children.values()):
            child.destroy()
        self.cards.clear()

        # build new cards (chưa cần biết số cột, _relayout_cards sẽ tính)
        for name in names:
            card = ttk.LabelFrame(self.cards_frame, text=name, style='Card.TLabelframe')
            img_lbl = ttk.Label(card)
            img_lbl.pack(fill='both', expand=True)
            img_lbl.bind('<Button-1>', lambda e, k=name: self._open_zoom(k))
            score_lbl = ttk.Label(card, text='FE: -- | PSNR: -- | SSIM: -- | T: --s', style='Metric.TLabel')
            score_lbl.pack(pady=(6,2))
            self.cards[name] = {'frame': card, 'img_lbl': img_lbl, 'score_lbl': score_lbl, 'tkimg': None}

        # tạm đặt cột mặc định rồi gọi relayout để tính đúng
        self._card_cols = columns_fallback
        for i, name in enumerate(names):
            r, c = divmod(i, self._card_cols)
            self.cards[name]['frame'].grid(row=r, column=c, padx=8, pady=8, sticky='nsew')
            self.cards_frame.grid_columnconfigure(c, weight=1)

        self._update_ref_algo_options(names)
        self._update_diff_options(names)
        self.root.after(0, self._relayout_cards)  # defer để Canvas có width đúng

    def _update_ref_algo_options(self, names: list[str]):
        self.ref_algo_combo['values'] = names
        cur = self.ref_algo_var.get()
        if cur not in names and names:
            self.ref_algo_var.set(names[0])

    # Cập nhật danh sách A/B cho tab Difference
    def _update_diff_options(self, names: list[str]):
        if not hasattr(self, 'diffA_combo'):  # phòng khi gọi trước khi _build_right_panel xong
            return
        self.diffA_combo['values'] = names
        self.diffB_combo['values'] = names
        if not names:
            self.diffA_var.set(''); self.diffB_var.set(''); return
        if self.diffA_var.get() not in names:
            self.diffA_var.set(names[0])
        if (self.diffB_var.get() not in names) or (self.diffB_var.get() == self.diffA_var.get()):
            self.diffB_var.set(next((n for n in names if n != self.diffA_var.get()), names[0]))

    # Swap nhanh cặp so sánh A/B
    def _swap_diff_pair(self):
        a, b = self.diffA_var.get(), self.diffB_var.get()
        self.diffA_var.set(b); self.diffB_var.set(a)
        self._refresh_tabs()

    # ---------- session / IO ----------
    def _get_session_dir(self) -> Path:
        if not hasattr(self, 'session_dir') or self.session_dir is None:
            stamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            base = Path("results") / stamp
            base.mkdir(parents=True, exist_ok=True)
            self.session_dir = base
        return self.session_dir

    # [CHANGED] nhận thêm seed để tái lập nhiễu theo seed
    def _apply_noise(self, img: np.ndarray, seed: int | None = None):
        kind = self.noise_var.get()
        amt = float(self.noise_amt.get())
        if kind == 'none' or amt <= 0:
            return img, {'type': 'none', 'amt_percent': 0.0}
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        if kind == 'salt-pepper':
            p = amt / 100.0
            noisy = img.copy()
            m = rng.random(img.shape)
            noisy[m < p/2] = 0
            noisy[m > 1 - p/2] = 255
            return noisy, {'type': 'salt-pepper', 'amt_percent': amt}
        elif kind == 'gaussian':
            sigma = (amt / 100.0) * 255.0
            noise = rng.normal(0.0, sigma, size=img.shape)
            noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            return noisy, {'type': 'gaussian', 'amt_percent': amt}
        else:
            return img, {'type': 'unknown', 'amt_percent': amt}

    def _save_results_one_image(self, img_path: str, original: np.ndarray, used_img: np.ndarray,
                                results: dict, noise_meta: dict, K_or_Ks):
        out_dir = self._get_session_dir()
        csv_path = out_dir / "results.csv"
        need_header = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if need_header:
                w.writerow([
                    "datetime", "image", "algo", "K",
                    "thresholds", "time_sec", "FE", "fitness",
                    "PSNR", "SSIM", "noise_type", "noise_amt_percent"
                ])
            for algo, info in results.items():
                vis = info.get("vis")
                if vis is None:
                    continue
                K_val = info.get("K", None)
                if K_val is None and "_K" in algo:
                    try: K_val = int(algo.split("_K", 1)[1])
                    except Exception: K_val = K_or_Ks if isinstance(K_or_Ks, int) else None
                stem = Path(img_path).stem
                cv2.imwrite(str(out_dir / f"{stem}_{algo}.png"), vis)
                ps_val = psnr(original, vis); ss_val = ssim(original, vis)
                w.writerow([
                    datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    Path(img_path).name, algo, (K_val if K_val is not None else K_or_Ks),
                    ";".join(map(str, info.get("thresholds", []))),
                    f"{info.get('time', 0.0):.6f}", f"{info.get('fe', 0.0):.6f}",
                    f"{info.get('fitness', 0.0):.6f}", f"{ps_val:.6f}", f"{ss_val:.6f}",
                    noise_meta.get("type", "none"), f"{float(noise_meta.get('amt_percent', 0.0)):.2f}",
                ])

    # --- helpers cho logging per seed ---
    def _safe_image_stem(self, path_or_none=None) -> str:
        """Lấy tên file (không đuôi) để đặt thư mục kết quả."""
        try:
            if path_or_none is not None:
                return Path(path_or_none).stem
            return Path(self.img_paths[self.left_listbox.curselection()[0]]).stem
        except Exception:
            return "image"

    def _safe_noise_tag(self, noise_meta: dict | None = None) -> str:
        """Tạo nhãn nhiễu: 'gaussian_1' / 'salt-pepper_0.5' / 'clean' / 'noise'."""
        try:
            if isinstance(noise_meta, dict) and "type" in noise_meta:
                kind = str(noise_meta.get("type", "none"))
                amt = float(noise_meta.get("amt_percent", 0.0))
                return f"{kind}_{amt:g}" if kind != "none" else "clean"
        except Exception:
            pass
        try:
            return self._current_noise_label() if hasattr(self, "_current_noise_label") else "noise"
        except Exception:
            return "noise"

    # ---------- actions ----------
    def open_image(self):
        paths = filedialog.askopenfilenames(
            filetypes=[('Images', '*.png;*.bmp;*.jpg;*.jpeg;*.tif;*.tiff'), ('All', '*.*')]
        )
        if not paths:
            return
        self.img_paths = list(paths)
        self.left_listbox.delete(0, tk.END)
        for p in self.img_paths:
            self.left_listbox.insert(tk.END, Path(p).name)
        img = cv2.imread(self.img_paths[0], cv2.IMREAD_GRAYSCALE)
        if img is None:
            return
        self.img = img
        self._open_preview(img, title='Preview')

    def _open_preview(self, img: np.ndarray, title='Preview'):
        pil = Image.fromarray(img).resize((420,420))
        tkimg = ImageTk.PhotoImage(pil.convert('L'))
        win = tk.Toplevel(self.root); win.title(title)
        l = ttk.Label(win, image=tkimg); l.image = tkimg
        l.pack(padx=8, pady=8)

    def _open_zoom(self, key: str):
        if self._last_results is None:
            return
        vis = self._last_results.get(key, {}).get('vis')
        if vis is None:
            return
        self._open_preview(vis, title=f'Zoom – {key}')

    def run(self):
        if not self.img_paths:
            return
        picked = self._picked_algorithms()
        if not picked:
            self.progress.config(text='Pick at least 1 algorithm.')
            return
        self.progress.config(text='Running...'); self.pb.start(8)

        if self.mode_var.get() == 'multi':
            try:
                Ks_list = [int(x) for x in self.ks_var.get().split(',') if x.strip()]
                if not Ks_list: raise ValueError
            except Exception:
                self.progress.config(text='Ks invalid (use CSV, e.g., 2,3,4)')
                self.pb.stop(); return
            names = []
            for a in picked:
                names += [f"MFWOA_K{k}" for k in Ks_list] if a=='MFWOA' else [f"{a}_K{k}" for k in Ks_list]
            self._rebuild_results_grid(names)
            threading.Thread(target=self._run_on_multiple_multiK, args=(picked, Ks_list), daemon=True).start()
        else:
            K = int(self.k_var.get())
            self._rebuild_results_grid(picked)
            threading.Thread(target=self._run_on_multiple_singleK, args=(picked, K), daemon=True).start()

    # ---------- workers ----------
    def _run_on_multiple_singleK(self, picked_algos: list[str], K: int):
        self.results_store = {}
        pop, iters = int(self.pop_var.get()), int(self.iters_var.get())

        for idx, p in enumerate(self.img_paths, 1):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # [CHANGED] seed + cờ stability
            save_flag = bool(getattr(self, "_is_stability_run", False))
            seed_val  = int(getattr(self, "_current_seed", 42))

            # áp nhiễu theo seed (nếu stability); không thì seed=None
            used_img, noise_meta = self._apply_noise(img, seed=seed_val if save_flag else None)

            img_stem = self._safe_image_stem(p)
            noise_tag = self._safe_noise_tag(noise_meta)
            save_root_dir = "results/stability" if save_flag else "results"

            # [CHANGED] truyền seed/image_stem/noise_tag xuống compare runner
            full = run_algorithms_on_image(
                used_img, K, pop=pop, iters=iters,
                seed=seed_val,
                save_per_seed=save_flag,
                save_root=save_root_dir,
                image_stem=img_stem,
                noise_tag=noise_tag,
            )

            res = {a: full[a] for a in picked_algos if a in full}
            self.results_store[p] = res

            self.root.after(0, lambda p=p, r=res, im=used_img: self._apply_run_results(p, r, im))

            try:
                self._save_results_one_image(
                    p, original=img, used_img=used_img,
                    results=res, noise_meta=noise_meta, K_or_Ks=K
                )
            except Exception as ex:
                print("save error:", ex)

            self.root.after(0, lambda i=idx: self.progress.config(text=f'Done {i}/{len(self.img_paths)}'))

        self.root.after(0, lambda: (self.progress.config(text='Done'), self.pb.stop()))

    def _run_on_multiple_multiK(self, picked_algos: list[str], Ks_list: list[int]):
        self.results_store = {}
        pop, iters = int(self.pop_var.get()), int(self.iters_var.get())
        for idx, p in enumerate(self.img_paths, 1):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            used_img, noise_meta = self._apply_noise(img)  # multi-K không cần seed cố định
            res: dict[str, dict] = {}
            if 'MFWOA' in picked_algos:
                if compare_run_mf_multitask is not None:
                    list_res = compare_run_mf_multitask(used_img, Ks_list, pop=pop, iters=iters)
                    for r in list_res: res[f"MFWOA_K{int(r['K'])}"] = r
                else:
                    res.update(_fallback_run_mfwoa_multitask_on_image(used_img, Ks_list, pop=pop, iters=iters))
            for algo in ['Otsu','FCM','PSO','GA','WOA']:
                if algo not in picked_algos: continue
                for k in Ks_list:
                    info = _run_single_algo_for_K(used_img, algo, k, pop=pop, iters=iters)
                    res[f"{algo}_K{k}"] = info
            self.results_store[p] = res
            self.root.after(0, lambda p=p, r=res, im=used_img: self._apply_run_results(p, r, im))
            try:
                self._save_results_one_image(p, original=img, used_img=used_img,
                                             results=res, noise_meta=noise_meta, K_or_Ks=','.join(map(str, Ks_list)))
            except Exception as ex:
                print("save error:", ex)
            self.root.after(0, lambda i=idx: self.progress.config(text=f'Done {i}/{len(self.img_paths)}'))
        self.root.after(0, lambda: (self.progress.config(text='Done'), self.pb.stop()))

    # ---------- export & diagnostics ----------
    def export_results(self):
        if not self.results_store: return
        out = filedialog.askdirectory()
        if not out: return
        outp = Path(out)
        for img_path, res in self.results_store.items():
            stem = Path(img_path).stem
            for algo, info in res.items():
                vis = info.get('vis')
                if vis is None: continue
                cv2.imwrite(str(outp / f"{stem}_{algo}.png"), vis)

    def _algo_base(self, key: str) -> str:
        return key.split('_K')[0] if '_K' in key else key

    def _short_name(self, key: str) -> str:
        base = self._algo_base(key)
        short = {'MFWOA':'MF','WOA':'WO','GA':'GA','PSO':'PS','FCM':'FC','Otsu':'O'}.get(base, base[:2])
        return f"{short}_{key.split('_K')[1]}" if '_K' in key else short

    def update_diagnostics(self, results: dict, original_img: np.ndarray):
        self._last_results = results
        self._last_img = original_img
        if not results: return

        scores = []
        for algo, info in results.items():
            vis = info.get('vis'); thr = info.get('thresholds', [])
            t = info.get('time', 0.0)
            ps_val = psnr(original_img, vis) if vis is not None else 0.0
            ss_val = ssim(original_img, vis) if vis is not None else 0.0
            fe = info.get('fe', 0.0)
            scores.append({'algo': algo, 'fe': fe, 'psnr': ps_val, 'ssim': ss_val, 'time': t, 'thr': thr, 'vis': vis})
            lbl = self.cards.get(algo, {}).get('score_lbl')
            if lbl is not None:
                lbl.config(text=f"FE: {fe:.6f} | PSNR: {ps_val:.2f} | SSIM: {ss_val:.3f} | T: {t:.2f}s")

        # redraw thumbs to current size
        self._rerender_thumbs()

        # --- Scores chart ---
        metric = self.metric_var.get()
        ykey = {'FE':'fe','PSNR':'psnr','SSIM':'ssim','Time':'time'}[metric]
        yvals = [s[ykey] for s in scores]
        labels = [self._short_name(s['algo']) for s in scores]
        bases = [self._algo_base(s['algo']) for s in scores]
        cmap = {'MFWOA':'#4F46E5','WOA':'#059669','GA':'#D97706','PSO':'#2563EB','FCM':'#DC2626','Otsu':'#6B7280'}
        colors = [cmap.get(b,'#888888') for b in bases]

        self.conv_ax.clear()
        self.conv_ax.bar(labels, yvals, color=colors, edgecolor='#222')
        self.conv_ax.set_title(f'{metric} by Algorithm / Task')
        self.conv_ax.set_ylabel(metric)
        self.conv_ax.tick_params(axis='x', rotation=45)
        self.conv_ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        handles = [Patch(facecolor=c, edgecolor='#222', label=b) for b,c in cmap.items()]
        self.conv_ax.legend(handles=handles, frameon=False, loc='upper left', ncols=3)
        ref = self.ref_algo_var.get()
        if ref in [s['algo'] for s in scores]:
            try:
                idx = [s['algo'] for s in scores].index(ref)
                bars = self.conv_ax.patches
                if 0 <= idx < len(bars):
                    bars[idx].set_linewidth(2.5); bars[idx].set_edgecolor('#F59E0B')
            except Exception:
                pass
        self.conv_canvas.draw()

        # --- Histogram ---
        self.hist_ax.clear()
        hist = histogram_from_image(original_img)
        levels = np.arange(256)
        self.hist_ax.fill_between(levels, 0, hist, step='pre', alpha=0.35, color='#9CA3AF')
        ymax = float(hist.max()) if len(hist) else 1.0
        for s in scores:
            thr = s['thr']
            if not thr: continue
            color = cmap.get(self._algo_base(s['algo']), '#2563EB')
            lw = 2 if s['algo'] == ref else 1
            alpha = 0.9 if s['algo'] == ref else 0.4
            self.hist_ax.vlines(thr, ymin=0, ymax=ymax, colors=color, linewidth=lw, alpha=alpha, label=s['algo'])
        self.hist_ax.set_xlim(0, 255)
        self.hist_ax.set_ylim(0, ymax*1.05)
        self.hist_ax.set_xlabel('Gray level'); self.hist_ax.set_ylabel('Count')
        self.hist_ax.set_title('Histogram + Thresholds')
        self.hist_ax.legend(frameon=False, loc='upper right', ncols=2)
        self.hist_canvas.draw()

        # --- Difference (|A - B|) ---
        self.diff_ax.clear()
        a_key = getattr(self, 'diffA_var', tk.StringVar(value='')).get()
        b_key = getattr(self, 'diffB_var', tk.StringVar(value='')).get()
        if results and a_key in results and b_key in results:
            a_vis = results[a_key].get('vis')
            b_vis = results[b_key].get('vis')
            if a_vis is not None and b_vis is not None and a_vis.shape == b_vis.shape:
                diff = np.abs(a_vis.astype(np.int16) - b_vis.astype(np.int16)).astype(np.uint8)
                self.diff_ax.imshow(diff, cmap='gray', aspect='auto'); self.diff_ax.axis('off')
                self.diff_ax.set_title(f"|{a_key} - {b_key}|")
            else:
                self.diff_ax.text(0.5, 0.5, 'Invalid pair (no image or size mismatch)',
                                  ha='center', va='center', transform=self.diff_ax.transAxes)
                self.diff_ax.set_axis_off()
        else:
            self.diff_ax.text(0.5, 0.5, 'Pick A and B in the dropdowns above',
                              ha='center', va='center', transform=self.diff_ax.transAxes)
            self.diff_ax.set_axis_off()
        self.diff_canvas.draw()

    def _apply_run_results(self, path: Path, res: dict, img: np.ndarray):
        self.results_store[path] = res
        self._last_results = res; self._last_img = img
        self._rerender_thumbs()
        # update metrics text now, charts below
        for algo, info in res.items():
            lbl = self.cards.get(algo, {}).get('score_lbl')
            if lbl is None: continue
            vis = info.get('vis')
            ps_val = psnr(img, vis) if vis is not None else 0.0
            ss_val = ssim(img, vis) if vis is not None else 0.0
            fe = info.get('fe', 0.0)
            t = info.get('time', 0.0)
            lbl.config(text=f"FE: {fe:.6f} | PSNR: {ps_val:.2f} | SSIM: {ss_val:.3f} | T: {t:.2f}s")
        self.update_diagnostics(res, img)

    def _refresh_tabs(self):
        if self._last_results is not None and self._last_img is not None:
            self.update_diagnostics(self._last_results, self._last_img)

    # ---------- stability (single-K only) ----------
    def run_stability(self):
        """Chạy Stability (30 seeds) cho chế độ Single-K; log ảnh + CSV theo từng seed."""
        if not self.img_paths or self.mode_var.get() == 'multi':
            return
        K = int(self.k_var.get())
        picked = self._picked_algorithms()

        # khoá nút + báo trạng thái
        self.stability_btn.config(state='disabled')
        self.progress.config(text='Stability: running...')
        self.pb.start(8)

        # bật cờ stability để worker single-K biết bật save_per_seed
        self._is_stability_run = True

        def worker():
            try:
                seeds = list(range(1, 31))  # 30 seeds
                pop, iters = int(self.pop_var.get()), int(self.iters_var.get())

                for i, sd in enumerate(seeds, 1):
                    self._current_seed = sd  # seed hiện hành cho logging
                    self.root.after(0, lambda i=i, n=len(seeds):
                                    self.progress.config(text=f'Stability: seed {i}/{n}'))

                    for p in self.img_paths:
                        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue

                        # áp nhiễu theo seed để tái lập
                        used_img, noise_meta = self._apply_noise(img, seed=sd)

                        # gọi runner single-K, bật lưu per-seed
                        full = run_algorithms_on_image(
                            used_img, K, pop=pop, iters=iters,
                            seed=sd,
                            save_per_seed=True,
                            save_root="results/stability",
                            image_stem=self._safe_image_stem(p),
                            noise_tag=self._safe_noise_tag(noise_meta),
                        )

                        res = {a: full[a] for a in picked if a in full}
                        self.root.after(0, lambda p=p, r=res, im=used_img: self._apply_run_results(p, r, im))

                self.root.after(0, lambda: self.progress.config(text='Stability: done'))

            except Exception as e:
                self.root.after(0, lambda: self.progress.config(text=f'Stability crashed: {e}'))

            finally:
                # tắt cờ + reset seed + mở nút
                self._is_stability_run = False
                self._current_seed = 42
                self.root.after(0, lambda: (self.stability_btn.config(state='normal'), self.pb.stop()))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
