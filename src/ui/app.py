"""Tkinter GUI to load an image, choose K and algorithms, run and show comparisons.

Non-blocking: runs optimization in a background thread and updates UI via after().
"""
from __future__ import annotations

import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import cv2
import numpy as np

from src.cli.compare import run_algorithms_on_image
from src.metrics.metrics import psnr, ssim
from src.metrics.fuzzy_entropy import histogram_from_image
from matplotlib.figure import Figure


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title('MFWOA Segmentation demo')
        self.img = None
        self.display_images = {}
        # top controls
        top = ttk.Frame(root)
        top.pack(fill='x', padx=8, pady=8)

        ttk.Button(top, text='Open Image(s)', command=self.open_image).pack(side='left')
        ttk.Label(top, text='K:').pack(side='left', padx=(10,0))
        self.k_var = tk.IntVar(value=2)
        ttk.Spinbox(top, from_=1, to=6, textvariable=self.k_var, width=4).pack(side='left')
        ttk.Button(top, text='Run', command=self.run).pack(side='left', padx=8)
        ttk.Button(top, text='Export Results', command=self.export_results).pack(side='left', padx=8)

        self.progress = ttk.Label(top, text='Idle')
        self.progress.pack(side='left', padx=10)

        # canvas area: two-column layout
        self.frame = ttk.Frame(root)
        self.frame.pack(fill='both', expand=True)

        # left container (images + results)
        self.left_container = ttk.Frame(self.frame)
        self.left_container.grid(row=0, column=0, sticky='nsew', padx=4, pady=4)

        # right container (diagnostics)
        self.right_container = ttk.Frame(self.frame)
        self.right_container.grid(row=0, column=1, sticky='nsew', padx=4, pady=4)

        # let left take more space
        self.frame.grid_columnconfigure(0, weight=3)
        self.frame.grid_columnconfigure(1, weight=1)

        # allow left_container rows/cols to expand
        self.left_container.grid_rowconfigure(0, weight=1)
        self.left_container.grid_rowconfigure(1, weight=2)
        self.left_container.grid_columnconfigure(0, weight=1)

        # Diagnostics combined figure with three stacked axes
        self.diagnostics_fig = Figure(figsize=(4, 8))
        self.conv_ax, self.hist_ax, self.diff_ax = self.diagnostics_fig.subplots(3, 1)
        self.diagnostics_canvas = FigureCanvasTkAgg(self.diagnostics_fig, master=self.right_container)
        self.diagnostics_canvas.get_tk_widget().pack(fill='both', expand=True)

        # top: original image(s) list; bottom: algorithm results horizontally
        self.left_listbox = tk.Listbox(self.left_container, height=8)
        self.left_listbox.grid(row=0, column=0, columnspan=3, padx=8, pady=8, sticky='ew')

        # result frame with horizontal layout
        self.result_frame = ttk.Frame(self.left_container)
        self.result_frame.grid(row=1, column=0, columnspan=3, padx=8, pady=8, sticky='ew')

        # dynamic result labels for algorithms in horizontal layout with 2 rows
        self.result_labels = {}
        self.score_labels = {}
        algorithms = ['Otsu', 'FCM', 'PSO', 'GA', 'WOA', 'MFWOA']
        row_size = 3  # 3 algorithms per row

        for i, algo in enumerate(algorithms):
            row = i // row_size
            col = i % row_size

            # create a frame for each algorithm and let it expand
            algo_frame = ttk.Frame(self.result_frame)
            algo_frame.grid(row=row, column=col, padx=8, pady=8, sticky='nsew')
            # make columns in result_frame expandable
            self.result_frame.grid_columnconfigure(col, weight=1)

            # algorithm name label
            lbl = ttk.Label(algo_frame, text=algo)
            lbl.pack()

            # image label below the name
            img_lbl = ttk.Label(algo_frame)
            img_lbl.pack(pady=4, fill='both', expand=True)
            # scoreboard label
            score_lbl = ttk.Label(algo_frame, text='FE: -- | PSNR: -- | SSIM: -- | T: --s')
            score_lbl.pack()
            self.result_labels[algo] = img_lbl
            self.score_labels[algo] = score_lbl

        # paths and results
        self.img_paths: list[str] = []
        self.results_store = {}

        # noise controls
        noise_frame = ttk.Frame(top)
        noise_frame.pack(side='right')
        ttk.Label(noise_frame, text='Noise:').pack(side='left')
        self.noise_var = tk.StringVar(value='none')
        ttk.Combobox(noise_frame, textvariable=self.noise_var, values=['none', 'gaussian', 'salt-pepper'], width=12).pack(side='left')
        ttk.Label(noise_frame, text='Amt %').pack(side='left')
        self.noise_amt = tk.DoubleVar(value=0.0)
        ttk.Spinbox(noise_frame, from_=0.0, to=5.0, increment=0.1, textvariable=self.noise_amt, width=5).pack(side='left')

        # stability run button (keep reference so we can disable while running)
        self.stability_btn = ttk.Button(top, text='Stability (30 seeds)', command=self.run_stability)
        self.stability_btn.pack(side='right', padx=8)

    def open_image(self):
        paths = filedialog.askopenfilenames(filetypes=[('Images', '*.png;*.bmp;*.jpg;*.jpeg'), ('All', '*.*')])
        if not paths:
            return
        # load first image as preview; store list
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
        pil = pil.resize((300,300))
        tkimg = ImageTk.PhotoImage(pil.convert('L'))
        # label can be Listbox or Label
        if isinstance(label, tk.Listbox):
            # show preview in a transient top-level window
            win = tk.Toplevel(self.root)
            win.title('Preview')
            l = ttk.Label(win, image=tkimg)
            l.image = tkimg
            l.pack()
        else:
            label.image = tkimg
            label.config(image=tkimg)

    def run(self):
        if not hasattr(self, 'img_paths') or not self.img_paths:
            return
        K = int(self.k_var.get())
        self.progress.config(text='Running...')
        thread = threading.Thread(target=self._run_on_multiple, args=(K,), daemon=True)
        thread.start()

    def _run_on_multiple(self, K: int):
        # process each selected image sequentially and update GUI
        self.results_store = {}
        for p in self.img_paths:
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            res = run_algorithms_on_image(img, K, pop=30, iters=100)
            self.results_store[p] = res
            # update UI previews for this image (show MFWOA as example)
            # update UI previews and diagnostics on main thread
            self.root.after(0, lambda p=p, r=res, im=img: self._apply_run_results(p, r, im))
        self.root.after(0, lambda: self.progress.config(text='Done'))

    def _run_bg(self, K: int):
        res = run_algorithms_on_image(self.img, K, pop=30, iters=100)
        # pick MFWOA vis
        vis = res.get('MFWOA', {}).get('vis')
        if vis is not None:
            self.root.after(0, lambda: self.show_image_on_label(vis, self.result_labels.get('MFWOA')))
        # update diagnostics for the single image run
        self.root.after(0, lambda r=res, im=self.img: self.update_diagnostics(r, im))
        self.root.after(0, lambda: self.progress.config(text='Done'))

    def export_results(self):
        if not hasattr(self, 'results_store') or not self.results_store:
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
                # cv2.imwrite expects BGR or single-channel uint8
                cv2.imwrite(str(save_path), vis)

    def update_diagnostics(self, results: dict, original_img: np.ndarray):
        # Update scoreboard labels and compute PSNR/SSIM and plot convergence/histogram/diff
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
            # use true fuzzy entropy value (fe) for display and plots
            fe = info.get('fe', 0.0)
            scores.append({'algo': algo, 'fe': fe, 'psnr': ps, 'ssim': ss, 'time': t, 'thr': thr})
            # update text label
            lbl = self.score_labels.get(algo)
            if lbl is not None:
                # always prepare label text (show numbers even if zero)
                try:
                    # show FE with 6 decimals as requested
                    txt = f"FE: {fe:.6f} | PSNR: {ps:.2f} | SSIM: {ss:.3f} | T: {t:.2f}s"
                except Exception:
                    txt = f"FE: {fe} | PSNR: {ps} | SSIM: {ss} | T: {t}s"
                # update label on the main thread
                self.root.after(0, lambda lb=lbl, tt=txt: lb.config(text=tt))
        # Convergence plot (show FE per algorithm as single point curve for now)
        # Convergence plot: place each algorithm at an x coordinate so points are visible
        self.conv_ax.clear()
        xs = list(range(len(scores)))
        ys = [s['fe'] for s in scores]
        colors = ['red' if s['algo'] == 'MFWOA' else 'C0' for s in scores]
        for i, s in enumerate(scores):
            self.conv_ax.plot(xs[i], ys[i], marker='o', color=colors[i], markersize=6)
        self.conv_ax.set_title('Fuzzy Entropy (per-algo)')
        self.conv_ax.set_ylabel('FE')
        self.conv_ax.set_xticks(xs)
        self.conv_ax.set_xticklabels([s['algo'] for s in scores], rotation=45)
        self.conv_ax.grid(True, axis='y', linestyle='--', alpha=0.4)

        # Histogram overlay with thresholds
        self.hist_ax.clear()
        hist = histogram_from_image(original_img)
        levels = np.arange(256)
        self.hist_ax.bar(levels, hist, width=1.0, color='gray')
        # draw thresholds for each algorithm (MFWOA thicker)
        for s in scores:
            thr = s['thr']
            if not thr:
                continue
            xs = thr
            if s['algo'] == 'MFWOA':
                self.hist_ax.vlines(xs, ymin=0, ymax=hist.max(), colors='red', linewidth=2)
            else:
                self.hist_ax.vlines(xs, ymin=0, ymax=hist.max(), colors='blue', linewidth=1, alpha=0.6)
        self.hist_ax.set_title('Histogram + Thresholds')
        # Difference map: compare each algorithm to MFWOA and show first non-MFWOA
        mf_vis = results.get('MFWOA', {}).get('vis')
        self.diff_ax.clear()
        shown = False
        for algo in algo_order:
            if algo == 'MFWOA':
                continue
            vis = results.get(algo, {}).get('vis')
            if vis is None or mf_vis is None:
                continue
            diff = np.abs(mf_vis.astype(np.int16) - vis.astype(np.int16)).astype(np.uint8)
            # show without axis ticks and keep aspect
            self.diff_ax.imshow(diff, cmap='gray', aspect='auto')
            self.diff_ax.axis('off')
            self.diff_ax.set_title(f'|MFWOA - {algo}|')
            shown = True
            break
        if not shown:
            self.diff_ax.text(0.5, 0.5, 'No diff', ha='center')

        # final redraw combined diagnostics canvas once (on main thread)
        self.root.after(0, lambda: self.diagnostics_canvas.draw())

    def _apply_run_results(self, path: Path, res: dict, img: np.ndarray):
        """Helper to update thumbnails, store results and refresh diagnostics on the main thread."""
        # store
        self.results_store[path] = res
        # update thumbnails
        for algo, lbl in self.result_labels.items():
            vis = res.get(algo, {}).get('vis')
            if vis is not None:
                self.show_image_on_label(vis, lbl)
        # update diagnostics (scoreboard, histogram, diff)
        self.update_diagnostics(res, img)

    def run_stability(self):
        # Run 30 seeds sequentially on selected images and show boxplots in diagnostics
        if not self.img_paths:
            return
        K = int(self.k_var.get())
        # disable button while running to avoid re-entry
        self.stability_btn.config(state='disabled')
        self.progress.config(text='Stability: running...')

        def worker():
            try:
                seeds = list(range(30))
                all_stats = []
                for idx, s in enumerate(seeds):
                    # update progress every seed
                    self.root.after(0, lambda i=idx, n=len(seeds): self.progress.config(text=f'Stability: seed {i+1}/{n}'))
                    for p in self.img_paths:
                        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                        if img is None:
                            continue
                        # set rng seed via numpy for reproducibility
                        np.random.seed(s)
                        res = run_algorithms_on_image(img, K, pop=30, iters=100)
                        # collect FE, PSNR, time per algo
                        for algo, info in res.items():
                            vis = info.get('vis')
                            ps = psnr(img, vis) if vis is not None else 0.0
                            ss = ssim(img, vis) if vis is not None else 0.0
                            # record true FE for stability
                            all_stats.append({'seed': s, 'image': Path(p).name, 'algo': algo, 'fe': info.get('fe', 0.0), 'psnr': ps, 'ssim': ss, 'time': info.get('time', 0.0)})
                            # also update UI with latest result for this image (main thread)
                        self.root.after(0, lambda p=p, r=res, im=img: self._apply_run_results(p, r, im))

                # produce boxplots
                try:
                    import pandas as pd
                    import seaborn as sns
                except Exception as e:
                    # surface import errors to user
                    self.root.after(0, lambda: self.progress.config(text=f'Stability failed: {e}'))
                    return

                df = pd.DataFrame(all_stats)
                fig, axes = plt.subplots(1,3, figsize=(12,4))
                sns.boxplot(data=df, x='algo', y='fe', ax=axes[0])
                axes[0].set_title('FE (30 seeds)')
                sns.boxplot(data=df, x='algo', y='psnr', ax=axes[1])
                axes[1].set_title('PSNR (30 seeds)')
                sns.boxplot(data=df, x='algo', y='time', ax=axes[2])
                axes[2].set_title('Time (30 seeds)')
                fig.suptitle('Stability over 30 seeds')

                # show in new window
                def show_fig():
                    win = tk.Toplevel(self.root)
                    win.title('Stability (30 seeds)')
                    canvas = FigureCanvasTkAgg(fig, master=win)
                    canvas.get_tk_widget().pack(fill='both', expand=True)
                    canvas.draw()
                self.root.after(0, show_fig)

                # final progress
                self.root.after(0, lambda: self.progress.config(text='Stability: done'))
            except Exception as e:
                # surface any unexpected exception to the progress label
                self.root.after(0, lambda: self.progress.config(text=f'Stability crashed: {e}'))
            finally:
                # re-enable button
                self.root.after(0, lambda: self.stability_btn.config(state='normal'))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
