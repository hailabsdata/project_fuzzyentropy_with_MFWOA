"""Tkinter GUI to load an image, choose K and algorithms, run and show comparisons.

Non-blocking: runs optimization in a background thread and updates UI via after().
Right diagnostics panel: Tabs (Scores | Histogram | Difference) + "Diff reference" combobox.
"""
from __future__ import annotations

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

import cv2
import numpy as np

from src.cli.compare import run_algorithms_on_image
from src.metrics.metrics import psnr, ssim
from src.metrics.fuzzy_entropy import histogram_from_image


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('MFWOA Segmentation demo')

        self.img = None
        self.img_paths: list[str] = []
        self.results_store = {}
        self._last_results: dict | None = None
        self._last_img: np.ndarray | None = None

        # =========================
        # Top controls
        # =========================
        top = ttk.Frame(root)
        top.pack(fill='x', padx=8, pady=8)

        ttk.Button(top, text='Open Image(s)', command=self.open_image).pack(side='left')
        ttk.Label(top, text='K:').pack(side='left', padx=(10, 0))
        self.k_var = tk.IntVar(value=2)
        ttk.Spinbox(top, from_=1, to=6, textvariable=self.k_var, width=4).pack(side='left')
        ttk.Button(top, text='Run', command=self.run).pack(side='left', padx=8)
        ttk.Button(top, text='Export Results', command=self.export_results).pack(side='left', padx=8)

        self.progress = ttk.Label(top, text='Idle')
        self.progress.pack(side='left', padx=10)

        # Noise controls (tuỳ chọn – placeholder)
        noise_frame = ttk.Frame(top)
        noise_frame.pack(side='right')
        ttk.Label(noise_frame, text='Noise:').pack(side='left')
        self.noise_var = tk.StringVar(value='none')
        ttk.Combobox(noise_frame, textvariable=self.noise_var,
                     values=['none', 'gaussian', 'salt-pepper'], width=12).pack(side='left')
        ttk.Label(noise_frame, text='Amt %').pack(side='left')
        self.noise_amt = tk.DoubleVar(value=0.0)
        ttk.Spinbox(noise_frame, from_=0.0, to=5.0, increment=0.1,
                    textvariable=self.noise_amt, width=5).pack(side='left')

        # Stability (30 seeds)
        self.stability_btn = ttk.Button(top, text='Stability (30 seeds)', command=self.run_stability)
        self.stability_btn.pack(side='right', padx=8)

        # =========================
        # Main two-column layout
        # =========================
        self.frame = ttk.Frame(root)
        self.frame.pack(fill='both', expand=True)

        # Left: image list + results thumbnails
        self.left_container = ttk.Frame(self.frame)
        self.left_container.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        # Right: diagnostics (tabs)
        self.right_container = ttk.Frame(self.frame)
        self.right_container.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)

        self.frame.grid_columnconfigure(0, weight=3)
        self.frame.grid_columnconfigure(1, weight=2)

        self.left_container.grid_rowconfigure(0, weight=1)
        self.left_container.grid_rowconfigure(1, weight=2)
        self.left_container.grid_columnconfigure(0, weight=1)

        # ---------------- Left content ----------------
        self.left_listbox = tk.Listbox(self.left_container, height=8)
        self.left_listbox.grid(row=0, column=0, columnspan=3, padx=8, pady=8, sticky='ew')

        self.result_frame = ttk.Frame(self.left_container)
        self.result_frame.grid(row=1, column=0, columnspan=3, padx=8, pady=8, sticky='ew')

        self.result_labels: dict[str, ttk.Label] = {}
        self.score_labels: dict[str, ttk.Label] = {}
        algorithms = ['Otsu', 'FCM', 'PSO', 'GA', 'WOA', 'MFWOA']
        row_size = 3  # 3 algo mỗi hàng

        for i, algo in enumerate(algorithms):
            row = i // row_size
            col = i % row_size

            algo_frame = ttk.Frame(self.result_frame)
            algo_frame.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
            self.result_frame.grid_columnconfigure(col, weight=1)

            ttk.Label(algo_frame, text=algo).pack()
            img_lbl = ttk.Label(algo_frame)
            img_lbl.pack(pady=4, fill='both', expand=True)
            score_lbl = ttk.Label(algo_frame, text='FE: -- | PSNR: -- | SSIM: -- | T: --s')
            score_lbl.pack()

            self.result_labels[algo] = img_lbl
            self.score_labels[algo] = score_lbl

        # ---------------- Right content ----------------
        # Controls on top of right panel
        right_top = ttk.Frame(self.right_container)
        right_top.pack(fill='x', padx=4, pady=2)

        ttk.Label(right_top, text='Diff reference:').pack(side='left')
        self.ref_algo_var = tk.StringVar(value='MFWOA')
        self.ref_algo_combo = ttk.Combobox(
            right_top, textvariable=self.ref_algo_var,
            values=['MFWOA', 'WOA', 'GA', 'PSO', 'FCM', 'Otsu'],
            state='readonly', width=8
        )
        self.ref_algo_combo.pack(side='left', padx=4)
        self.ref_algo_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_tabs())

        # Notebook Tabs
        self.nb = ttk.Notebook(self.right_container)
        self.nb.pack(fill='both', expand=True)

        # Tab 1: Scores (bar chart FE)
        tab_scores = ttk.Frame(self.nb)
        self.nb.add(tab_scores, text='Scores')
        self.conv_fig = Figure(figsize=(5, 3), constrained_layout=True)
        self.conv_ax = self.conv_fig.add_subplot(111)
        self.conv_canvas = FigureCanvasTkAgg(self.conv_fig, master=tab_scores)
        self.conv_canvas.get_tk_widget().pack(fill='both', expand=True)

        # Tab 2: Histogram (+ thresholds + toolbar)
        tab_hist = ttk.Frame(self.nb)
        self.nb.add(tab_hist, text='Histogram')
        self.hist_fig = Figure(figsize=(5, 3), constrained_layout=True)
        self.hist_ax = self.hist_fig.add_subplot(111)
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=tab_hist)
        self.hist_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.hist_toolbar = NavigationToolbar2Tk(self.hist_canvas, tab_hist)
        self.hist_toolbar.update()

        # Tab 3: Difference (|Ref - Algo|)
        tab_diff = ttk.Frame(self.nb)
        self.nb.add(tab_diff, text='Difference')
        self.diff_fig = Figure(figsize=(5, 3), constrained_layout=True)
        self.diff_ax = self.diff_fig.add_subplot(111)
        self.diff_canvas = FigureCanvasTkAgg(self.diff_fig, master=tab_diff)
        self.diff_canvas.get_tk_widget().pack(fill='both', expand=True)

    # =========================
    # Helpers & Actions
    # =========================
    def open_image(self):
        paths = filedialog.askopenfilenames(
            filetypes=[('Images', '*.png;*.bmp;*.jpg;*.jpeg'), ('All', '*.*')]
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
        self.show_image_on_label(img, self.left_listbox)

    def show_image_on_label(self, img: np.ndarray, label):
        pil = Image.fromarray(img)
        pil = pil.resize((300, 300))
        tkimg = ImageTk.PhotoImage(pil.convert('L'))
        if isinstance(label, tk.Listbox):
            win = tk.Toplevel(self.root)
            win.title('Preview')
            l = ttk.Label(win, image=tkimg)
            l.image = tkimg
            l.pack()
        else:
            label.image = tkimg
            label.config(image=tkimg)

    def run(self):
        if not self.img_paths:
            return
        K = int(self.k_var.get())
        self.progress.config(text='Running...')
        thread = threading.Thread(target=self._run_on_multiple, args=(K,), daemon=True)
        thread.start()

    def _run_on_multiple(self, K: int):
        self.results_store = {}
        for p in self.img_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            res = run_algorithms_on_image(img, K, pop=30, iters=100)
            self.results_store[p] = res
            self.root.after(0, lambda p=p, r=res, im=img: self._apply_run_results(p, r, im))
        self.root.after(0, lambda: self.progress.config(text='Done'))

    def _run_bg(self, K: int):
        res = run_algorithms_on_image(self.img, K, pop=30, iters=100)
        vis = res.get('MFWOA', {}).get('vis')
        if vis is not None:
            self.root.after(0, lambda: self.show_image_on_label(vis, self.result_labels.get('MFWOA')))
        self.root.after(0, lambda r=res, im=self.img: self.update_diagnostics(r, im))
        self.root.after(0, lambda: self.progress.config(text='Done'))

    def export_results(self):
        if not self.results_store:
            return
        out = filedialog.askdirectory()
        if not out:
            return
        outp = Path(out)
        for img_path, res in self.results_store.items():
            stem = Path(img_path).stem
            for algo, info in res.items():
                vis = info.get('vis')
                if vis is None:
                    continue
                save_path = outp / f"{stem}_{algo}.png"
                cv2.imwrite(str(save_path), vis)

    def update_diagnostics(self, results: dict, original_img: np.ndarray):
        """Rebuild all three tabs based on latest results."""
        self._last_results = results
        self._last_img = original_img

        algo_order = ['Otsu', 'FCM', 'PSO', 'GA', 'WOA', 'MFWOA']
        scores = []
        for algo in algo_order:
            info = results.get(algo, {})
            vis = info.get('vis')
            thr = info.get('thresholds', [])
            t = info.get('time', 0.0)
            if vis is not None:
                ps = psnr(original_img, vis)
                ss = ssim(original_img, vis)
            else:
                ps = 0.0
                ss = 0.0
            fe = info.get('fe', 0.0)
            scores.append({'algo': algo, 'fe': fe, 'psnr': ps, 'ssim': ss, 'time': t, 'thr': thr, 'vis': vis})

            lbl = self.score_labels.get(algo)
            if lbl is not None:
                try:
                    txt = f"FE: {fe:.6f} | PSNR: {ps:.2f} | SSIM: {ss:.3f} | T: {t:.2f}s"
                except Exception:
                    txt = f"FE: {fe} | PSNR: {ps} | SSIM: {ss} | T: {t}s"
                self.root.after(0, lambda lb=lbl, tt=txt: lb.config(text=tt))

        # ---- Tab 1: Scores (bar chart) ----
        self.conv_ax.clear()
        names = [s['algo'] for s in scores]
        fes = [s['fe'] for s in scores]
        bars = self.conv_ax.bar(names, fes)
        ref = self.ref_algo_var.get()
        for bar, name in zip(bars, names):
            if name == ref:
                bar.set_edgecolor('red')
                bar.set_linewidth(2.0)
        self.conv_ax.set_title('Fuzzy Entropy by Algorithm')
        self.conv_ax.set_ylabel('FE')
        if fes:
            self.conv_ax.set_ylim(bottom=min(fes) - 0.01)
        self.conv_ax.grid(True, axis='y', linestyle='--', alpha=0.4)
        self.conv_canvas.draw()

        # ---- Tab 2: Histogram + thresholds ----
        self.hist_ax.clear()
        hist = histogram_from_image(original_img)
        levels = np.arange(256)
        self.hist_ax.fill_between(levels, 0, hist, step='pre', alpha=0.35)
        ymax = float(hist.max()) if len(hist) else 1.0
        for s in scores:
            thr = s['thr']
            if not thr:
                continue
            if s['algo'] == ref:
                self.hist_ax.vlines(thr, ymin=0, ymax=ymax, colors='red', linewidth=2, alpha=0.9, label=f'{ref}')
            else:
                self.hist_ax.vlines(thr, ymin=0, ymax=ymax, colors='blue', linewidth=1, alpha=0.45)
        self.hist_ax.set_xlim(0, 255)
        self.hist_ax.set_ylim(0, ymax * 1.05)
        self.hist_ax.set_xlabel('Gray level')
        self.hist_ax.set_ylabel('Count')
        self.hist_ax.set_title('Histogram + Thresholds')
        handles, labels = self.hist_ax.get_legend_handles_labels()
        if handles:
            self.hist_ax.legend(frameon=False, loc='upper right')
        self.hist_canvas.draw()

        # ---- Tab 3: Difference (|Ref - Algo|) ----
        self.diff_ax.clear()
        ref_vis = next((s['vis'] for s in scores if s['algo'] == ref), None)
        shown = False
        if ref_vis is not None:
            for s in scores:
                if s['algo'] == ref:
                    continue
                vis = s['vis']
                if vis is None:
                    continue
                diff = np.abs(ref_vis.astype(np.int16) - vis.astype(np.int16)).astype(np.uint8)
                self.diff_ax.imshow(diff, cmap='gray', aspect='auto')
                self.diff_ax.axis('off')
                # NOTE: f-string fixed to avoid quote clash
                self.diff_ax.set_title(f"|{ref} - {s['algo']}|")
                shown = True
                break
        if not shown:
            self.diff_ax.text(0.5, 0.5, 'No diff', ha='center', va='center',
                              transform=self.diff_ax.transAxes)
            self.diff_ax.set_axis_off()
        self.diff_canvas.draw()

    def _apply_run_results(self, path: Path, res: dict, img: np.ndarray):
        """Cập nhật thumbnail + diagnostics sau khi chạy xong 1 ảnh."""
        self.results_store[path] = res
        for algo, lbl in self.result_labels.items():
            vis = res.get(algo, {}).get('vis')
            if vis is not None:
                self.show_image_on_label(vis, lbl)
        self.update_diagnostics(res, img)

    def _refresh_tabs(self):
        """Redraw tabs when user changes diff reference."""
        if self._last_results is not None and self._last_img is not None:
            self.update_diagnostics(self._last_results, self._last_img)

    def run_stability(self):
        if not self.img_paths:
            return
        K = int(self.k_var.get())
        self.stability_btn.config(state='disabled')
        self.progress.config(text='Stability: running...')

        def worker():
            try:
                seeds = list(range(30))
                all_stats = []
                for idx, s in enumerate(seeds):
                    self.root.after(0, lambda i=idx, n=len(seeds):
                                    self.progress.config(text=f'Stability: seed {i+1}/{n}'))
                    for p in self.img_paths:
                        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        np.random.seed(s)
                        res = run_algorithms_on_image(img, K, pop=30, iters=100)
                        for algo, info in res.items():
                            vis = info.get('vis')
                            ps = psnr(img, vis) if vis is not None else 0.0
                            ss = ssim(img, vis) if vis is not None else 0.0
                            all_stats.append({
                                'seed': s, 'image': Path(p).name, 'algo': algo,
                                'fe': info.get('fe', 0.0), 'psnr': ps,
                                'ssim': ss, 'time': info.get('time', 0.0)
                            })
                    self.root.after(0, lambda p=p, r=res, im=img: self._apply_run_results(p, r, im))

                # Vẽ boxplot (nếu có seaborn)
                try:
                    import pandas as pd
                    import seaborn as sns
                except Exception as e:
                    self.root.after(0, lambda:
                                    self.progress.config(text=f'Stability done (no seaborn): {e}'))
                    self.root.after(0, lambda: self.stability_btn.config(state='normal'))
                    return

                df = pd.DataFrame(all_stats)
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                sns.boxplot(data=df, x='algo', y='fe', ax=axes[0]); axes[0].set_title('FE (30 seeds)')
                sns.boxplot(data=df, x='algo', y='psnr', ax=axes[1]); axes[1].set_title('PSNR (30 seeds)')
                sns.boxplot(data=df, x='algo', y='time', ax=axes[2]); axes[2].set_title('Time (30 seeds)')
                fig.suptitle('Stability over 30 seeds')

                def show_fig():
                    win = tk.Toplevel(self.root)
                    win.title('Stability (30 seeds)')
                    canvas = FigureCanvasTkAgg(fig, master=win)
                    canvas.get_tk_widget().pack(fill='both', expand=True)
                    canvas.draw()
                self.root.after(0, show_fig)
                self.root.after(0, lambda: self.progress.config(text='Stability: done'))
            except Exception as e:
                self.root.after(0, lambda: self.progress.config(text=f'Stability crashed: {e}'))
            finally:
                self.root.after(0, lambda: self.stability_btn.config(state='normal'))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
