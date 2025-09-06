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
matplotlib.use('Agg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import cv2
import numpy as np

from src.cli.compare import run_algorithms_on_image


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

        # canvas area
        self.frame = ttk.Frame(root)
        self.frame.pack(fill='both', expand=True)

        # left: original image(s) list; right: algorithm results
        self.left_listbox = tk.Listbox(self.frame, height=8)
        self.left_listbox.grid(row=0, column=0, padx=8, pady=8, sticky='n')
        self.result_frame = ttk.Frame(self.frame)
        self.result_frame.grid(row=0, column=1, padx=8, pady=8, sticky='n')
        # dynamic result labels for algorithms
        self.result_labels = {}
        for j, algo in enumerate(['Otsu', 'WOA', 'MFWOA']):
            lbl = ttk.Label(self.result_frame, text=algo)
            lbl.grid(row=j*2, column=0)
            img_lbl = ttk.Label(self.result_frame)
            img_lbl.grid(row=j*2+1, column=0, padx=4, pady=4)
            self.result_labels[algo] = img_lbl

        # paths and results
        self.img_paths: list[str] = []
        self.results_store = {}

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
            for algo, lbl in self.result_labels.items():
                vis = res.get(algo, {}).get('vis')
                if vis is not None:
                    self.root.after(0, lambda v=vis, l=lbl: self.show_image_on_label(v, l))
        self.root.after(0, lambda: self.progress.config(text='Done'))

    def _run_bg(self, K: int):
        res = run_algorithms_on_image(self.img, K, pop=30, iters=100)
        # pick MFWOA vis
        vis = res.get('MFWOA', {}).get('vis')
        if vis is not None:
            self.root.after(0, lambda: self.show_image_on_label(vis, self.result_labels.get('MFWOA')))
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


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == '__main__':
    main()
