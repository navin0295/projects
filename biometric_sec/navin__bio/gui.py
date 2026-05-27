"""
gui.py
──────
Tkinter GUI for the Fingerprint Biometric Cryptosystem.

Tabs
----
1. Enroll    — add a new user with fingerprint images + live image preview
2. Verify    — authenticate a query fingerprint against a stored template
3. Users     — browse, inspect, and delete enrolled users
4. Train     — fine-tune the CNN on the full SOCOFing dataset with live loss chart
5. About     — system info

Run:
    python gui.py
"""

from __future__ import annotations

import glob
import re
import threading
import traceback
from pathlib import Path
from typing import List, Optional

# ── tk imports ────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# ── project imports (lazy-loaded inside threads to avoid blocking UI) ──────────
import config

# ═══════════════════════════════════════════════════════════════════════════════
# THEME / STYLE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

CLR = {
    "bg"        : "#1e1e2e",   # main background
    "surface"   : "#2a2a3e",   # card / panel surface
    "surface2"  : "#313150",   # slightly lighter surface
    "border"    : "#44446a",   # border / separator
    "accent"    : "#7c6af7",   # purple accent
    "accent2"   : "#56d6a0",   # teal accent (success)
    "danger"    : "#f75050",   # red (delete / deny)
    "warning"   : "#f7c948",   # amber (caution)
    "text"      : "#e0e0f0",   # primary text
    "text2"     : "#9999bb",   # secondary text
    "btn_bg"    : "#7c6af7",   # button background
    "btn_fg"    : "#ffffff",   # button text
    "entry_bg"  : "#252540",   # entry background
    "entry_fg"  : "#e0e0f0",   # entry text
    "granted"   : "#56d6a0",
    "denied"    : "#f75050",
}

FONT_TITLE  = ("Segoe UI", 18, "bold")
FONT_HEAD   = ("Segoe UI", 12, "bold")
FONT_BODY   = ("Segoe UI", 10)
FONT_SMALL  = ("Segoe UI", 9)
FONT_MONO   = ("Consolas", 9)


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY COMPONENT LOADER  (runs once in a background thread on first use)
# ═══════════════════════════════════════════════════════════════════════════════

class ComponentLoader:
    """
    Loads heavy ML components once and caches them.
    Exposes a ready Event that the GUI can wait on.
    """

    _instance = None

    def __init__(self):
        self.ready   = threading.Event()
        self.error:  Optional[str] = None
        self._thread = threading.Thread(target=self._load, daemon=True)
        self._thread.start()

    @classmethod
    def get(cls) -> "ComponentLoader":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        try:
            from preprocessor      import FingerprintPreprocessor
            from feature_extractor import FingerprintCNNExtractor
            from crypto            import AESCryptosystem
            from database          import FingerprintDatabase
            from enroll            import EnrollmentPipeline
            from authenticate      import AuthenticationPipeline

            from camera_preprocessor import CameraFingerprintPreprocessor

            self.preprocessor        = FingerprintPreprocessor()
            self.camera_preprocessor = CameraFingerprintPreprocessor()
            self.extractor     = FingerprintCNNExtractor()
            self.aes           = AESCryptosystem()
            self.db            = FingerprintDatabase()
            self.enroller      = EnrollmentPipeline(
                self.preprocessor, self.extractor, self.aes, self.db
            )
            self.camera_enroller = EnrollmentPipeline(
                self.camera_preprocessor, self.extractor, self.aes, self.db
            )
            self.authenticator = AuthenticationPipeline(
                self.preprocessor, self.extractor, self.aes, self.db
            )
            self.camera_authenticator = AuthenticationPipeline(
                self.camera_preprocessor, self.extractor, self.aes, self.db
            )
            self.ready.set()
        except Exception as exc:
            self.error = traceback.format_exc()
            self.ready.set()   # unblock waiters even on error


# ═══════════════════════════════════════════════════════════════════════════════
# REUSABLE WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

def styled_button(parent, text, command, color=None, **kw):
    bg = color or CLR["btn_bg"]
    btn = tk.Button(
        parent, text=text, command=command,
        bg=bg, fg=CLR["btn_fg"],
        activebackground=CLR["surface2"], activeforeground=CLR["text"],
        relief="flat", bd=0, padx=14, pady=7,
        font=FONT_BODY, cursor="hand2",
        **kw
    )
    btn.bind("<Enter>", lambda e: btn.configure(bg=_lighten(bg)))
    btn.bind("<Leave>", lambda e: btn.configure(bg=bg))
    return btn


def styled_entry(parent, width=30, **kw):
    return tk.Entry(
        parent, width=width,
        bg=CLR["entry_bg"], fg=CLR["entry_fg"],
        insertbackground=CLR["text"],
        relief="flat", bd=0, font=FONT_BODY,
        highlightthickness=1, highlightcolor=CLR["accent"],
        highlightbackground=CLR["border"],
        **kw
    )


def section_label(parent, text):
    return tk.Label(
        parent, text=text,
        bg=CLR["surface"], fg=CLR["accent"],
        font=FONT_HEAD, anchor="w"
    )


def body_label(parent, text, **kw):
    return tk.Label(
        parent, text=text,
        bg=CLR["surface"], fg=CLR["text"],
        font=FONT_BODY, anchor="w",
        **kw
    )


def _lighten(hex_color: str, amount: int = 20) -> str:
    """Lighten a hex color for hover effects."""
    try:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = min(255, r + amount)
        g = min(255, g + amount)
        b = min(255, b + amount)
        return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        return hex_color


def make_card(parent, padx=16, pady=16) -> tk.Frame:
    """A surface-coloured rounded card frame."""
    f = tk.Frame(parent, bg=CLR["surface"], padx=padx, pady=pady,
                 highlightthickness=1, highlightbackground=CLR["border"])
    return f


def separator(parent):
    return tk.Frame(parent, bg=CLR["border"], height=1)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PREVIEW PANEL
# ═══════════════════════════════════════════════════════════════════════════════

class ImagePreviewPanel(tk.Frame):
    """
    Shows a scrollable grid of fingerprint thumbnails.
    Used in the Enroll and Verify tabs.
    """
    THUMB = 90

    def __init__(self, parent, max_images: int = 10, **kw):
        super().__init__(parent, bg=CLR["surface"], **kw)
        self.max_images = max_images
        self._paths: List[str] = []
        self._refs:  List[ImageTk.PhotoImage] = []   # prevent GC

        self._canvas = tk.Canvas(self, bg=CLR["surface2"], height=120,
                                 highlightthickness=0)
        self._canvas.pack(fill="x", expand=True)
        self._placeholder()

    def _placeholder(self):
        self._canvas.delete("all")
        self._canvas.create_text(
            200, 60, text="No images selected",
            fill=CLR["text2"], font=FONT_SMALL
        )

    def set_paths(self, paths: List[str]):
        self._paths = paths[:self.max_images]
        self._refs  = []
        self._canvas.delete("all")

        x = 8
        for p in self._paths:
            try:
                img = Image.open(p).convert("L")
                img.thumbnail((self.THUMB, self.THUMB))
                photo = ImageTk.PhotoImage(img)
                self._refs.append(photo)
                self._canvas.create_image(x, 8, anchor="nw", image=photo)
                # filename below thumb
                fname = Path(p).name[:14]
                self._canvas.create_text(
                    x + self.THUMB // 2, 104,
                    text=fname, fill=CLR["text2"], font=FONT_SMALL, anchor="n"
                )
                x += self.THUMB + 12
            except Exception:
                pass

        if not self._paths:
            self._placeholder()

    def clear(self):
        self._paths = []
        self._refs  = []
        self._placeholder()

    def get_paths(self) -> List[str]:
        return self._paths


# ═══════════════════════════════════════════════════════════════════════════════
# STATUS BAR
# ═══════════════════════════════════════════════════════════════════════════════

class StatusBar(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CLR["surface2"], **kw)
        self._var = tk.StringVar(value="Ready")
        self._dot = tk.Label(self, text="●", bg=CLR["surface2"],
                             fg=CLR["accent2"], font=FONT_SMALL)
        self._dot.pack(side="left", padx=(8, 4))
        self._lbl = tk.Label(self, textvariable=self._var,
                             bg=CLR["surface2"], fg=CLR["text2"],
                             font=FONT_SMALL, anchor="w")
        self._lbl.pack(side="left", fill="x", expand=True)

    def set(self, msg: str, color: str = CLR["text2"]):
        self._var.set(msg)
        self._lbl.configure(fg=color)
        self._dot.configure(fg=color)
        self.update_idletasks()

    def ok(self, msg: str):   self.set(msg, CLR["accent2"])
    def err(self, msg: str):  self.set(msg, CLR["danger"])
    def info(self, msg: str): self.set(msg, CLR["text2"])
    def busy(self, msg: str): self.set(msg, CLR["warning"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ENROLL
# ═══════════════════════════════════════════════════════════════════════════════

class EnrollTab(tk.Frame):
    def __init__(self, parent, loader: ComponentLoader, status: StatusBar, **kw):
        super().__init__(parent, bg=CLR["bg"], **kw)
        self.loader = loader
        self.status = status
        self._build()

    def _build(self):
        # Header
        tk.Label(self, text="Enroll New User", font=FONT_TITLE,
                 bg=CLR["bg"], fg=CLR["text"]).pack(anchor="w", padx=20, pady=(18, 4))
        tk.Label(self, text="Register a fingerprint template in the secure database.",
                 font=FONT_BODY, bg=CLR["bg"], fg=CLR["text2"]).pack(anchor="w", padx=20)
        separator(self).pack(fill="x", padx=20, pady=10)

        # Form card
        card = make_card(self)
        card.pack(fill="x", padx=20, pady=6)

        # User ID
        section_label(card, "User Identity").grid(row=0, column=0, sticky="w", pady=(0, 2))
        self._uid_var = tk.StringVar()
        uid_entry = styled_entry(card, textvariable=self._uid_var, width=36)
        uid_entry.grid(row=1, column=0, sticky="w", pady=(0, 10))
        body_label(card, "e.g.  alice  |  user_0042  |  john.doe").grid(
            row=2, column=0, sticky="w")

        separator(card).grid(row=3, column=0, sticky="ew", pady=10)

        # Image selection
        section_label(card, "Fingerprint Images").grid(row=4, column=0, sticky="w")
        body_label(card, "Select 1–5 images of the SAME finger. More images = more stable key.").grid(
            row=5, column=0, sticky="w", pady=(2, 8))

        btn_row = tk.Frame(card, bg=CLR["surface"])
        btn_row.grid(row=6, column=0, sticky="w", pady=(0, 10))
        styled_button(btn_row, "📂  Browse Images", self._browse_images).pack(side="left", padx=(0, 8))
        styled_button(btn_row, "✕  Clear", self._clear_images, color=CLR["surface2"]).pack(side="left")

        self._count_label = tk.Label(card, text="0 images selected",
                                     bg=CLR["surface"], fg=CLR["text2"], font=FONT_SMALL)
        self._count_label.grid(row=7, column=0, sticky="w")

        # Preview
        section_label(card, "Preview").grid(row=8, column=0, sticky="w", pady=(10, 4))
        self._preview = ImagePreviewPanel(card, max_images=5)
        self._preview.grid(row=9, column=0, sticky="ew")

        separator(card).grid(row=10, column=0, sticky="ew", pady=12)

        # Image mode toggle
        separator(card).grid(row=10, column=0, sticky="ew", pady=(4, 8))
        mode_row = tk.Frame(card, bg=CLR["surface"])
        mode_row.grid(row=10, column=0, sticky="w", pady=(0, 4))

        section_label(card, "Image Type").grid(row=10, column=0, sticky="w")
        mode_frame = tk.Frame(card, bg=CLR["surface"])
        mode_frame.grid(row=11, column=0, sticky="w", pady=(4, 8))
        self._img_mode = tk.StringVar(value="sensor")
        tk.Radiobutton(mode_frame, text="📷  Sensor / Scanner image  (BMP, SOCOFing)",
                       variable=self._img_mode, value="sensor",
                       bg=CLR["surface"], fg=CLR["text"], selectcolor=CLR["surface2"],
                       activebackground=CLR["surface"], font=FONT_SMALL,
                       command=self._on_mode_change).pack(anchor="w")
        tk.Radiobutton(mode_frame, text="📱  Camera photo  (JPG/PNG from smartphone)",
                       variable=self._img_mode, value="camera",
                       bg=CLR["surface"], fg=CLR["text"], selectcolor=CLR["surface2"],
                       activebackground=CLR["surface"], font=FONT_SMALL,
                       command=self._on_mode_change).pack(anchor="w")
        self._mode_hint = tk.Label(card, text="",
                                    bg=CLR["surface"], fg=CLR["warning"], font=FONT_SMALL,
                                    wraplength=520, justify="left")
        self._mode_hint.grid(row=12, column=0, sticky="w")

        # Overwrite option
        self._overwrite = tk.BooleanVar(value=False)
        tk.Checkbutton(
            card, text="Overwrite if user already exists",
            variable=self._overwrite,
            bg=CLR["surface"], fg=CLR["text2"],
            selectcolor=CLR["surface2"], activebackground=CLR["surface"],
            font=FONT_SMALL
        ).grid(row=13, column=0, sticky="w")

        separator(card).grid(row=14, column=0, sticky="ew", pady=12)

        # Enroll button
        self._enroll_btn = styled_button(card, "🔒  Enroll User", self._do_enroll,
                                          color=CLR["accent"])
        self._enroll_btn.grid(row=15, column=0, sticky="w")

        # Result area
        self._result_frame = tk.Frame(card, bg=CLR["surface"])
        self._result_frame.grid(row=16, column=0, sticky="ew", pady=(12, 0))

    # ── events ────────────────────────────────────────────────────────────────

    def _browse_images(self):
        paths = filedialog.askopenfilenames(
            title="Select fingerprint images",
            filetypes=[
                ("Image files", "*.bmp *.BMP *.png *.PNG *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*.*"),
            ]
        )
        if paths:
            self._preview.set_paths(list(paths))
            n = len(self._preview.get_paths())
            self._count_label.configure(
                text=f"{n} image{'s' if n != 1 else ''} selected",
                fg=CLR["accent2"] if n > 0 else CLR["text2"]
            )

    def _clear_images(self):
        self._preview.clear()
        self._count_label.configure(text="0 images selected", fg=CLR["text2"])

    def _on_mode_change(self):
        if self._img_mode.get() == "camera":
            self._mode_hint.configure(
                text="📱 Camera mode: images will be automatically inverted and "
                     "enhanced to match sensor-style before enrollment. "
                     "Use clear macro photos of your fingertip."
            )
        else:
            self._mode_hint.configure(text="")

    def _do_enroll(self):
        user_id = self._uid_var.get().strip()
        paths   = self._preview.get_paths()

        if not user_id:
            messagebox.showwarning("Missing input", "Please enter a User ID.")
            return
        if not re.match(r"^[\w.\-]+$", user_id):
            messagebox.showwarning("Invalid ID",
                "User ID may only contain letters, digits, dots, hyphens, and underscores.")
            return
        if not paths:
            messagebox.showwarning("No images", "Please select at least one fingerprint image.")
            return

        self._enroll_btn.configure(state="disabled", text="⏳  Enrolling…")
        self.status.busy(f"Enrolling {user_id}…")
        for w in self._result_frame.winfo_children():
            w.destroy()

        use_camera = self._img_mode.get() == "camera"

        def worker():
            try:
                loader = self.loader
                if not loader.ready.is_set():
                    self.after(0, lambda: self.status.busy("Loading ML components…"))
                    loader.ready.wait()
                if loader.error:
                    raise RuntimeError(loader.error)

                enroller = loader.camera_enroller if use_camera else loader.enroller
                if use_camera:
                    self.after(0, lambda: self.status.busy(
                        f"Preprocessing camera images for {user_id}…"))
                result = enroller.enroll(
                    user_id     = user_id,
                    image_paths = paths,
                    overwrite   = self._overwrite.get(),
                )
                self.after(0, lambda: self._on_success(result))
            except Exception as exc:
                self.after(0, lambda: self._on_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_success(self, result: dict):
        self._enroll_btn.configure(state="normal", text="🔒  Enroll User")
        self.status.ok(f"✅  User '{result['user_id']}' enrolled successfully.")

        # Result card
        for w in self._result_frame.winfo_children():
            w.destroy()
        res_card = tk.Frame(self._result_frame, bg="#1a3a2a",
                            highlightthickness=1, highlightbackground=CLR["accent2"],
                            padx=12, pady=10)
        res_card.pack(fill="x")
        tk.Label(res_card, text="✅  Enrollment Successful", font=FONT_HEAD,
                 bg="#1a3a2a", fg=CLR["accent2"]).pack(anchor="w")
        for k, v in result.items():
            tk.Label(res_card, text=f"  {k}: {v}",
                     bg="#1a3a2a", fg=CLR["text"], font=FONT_MONO).pack(anchor="w")

    def _on_error(self, msg: str):
        self._enroll_btn.configure(state="normal", text="🔒  Enroll User")
        self.status.err(f"❌  Enrollment failed: {msg}")

        for w in self._result_frame.winfo_children():
            w.destroy()
        err_card = tk.Frame(self._result_frame, bg="#3a1a1a",
                            highlightthickness=1, highlightbackground=CLR["danger"],
                            padx=12, pady=10)
        err_card.pack(fill="x")
        tk.Label(err_card, text="❌  Enrollment Failed", font=FONT_HEAD,
                 bg="#3a1a1a", fg=CLR["danger"]).pack(anchor="w")
        tk.Label(err_card, text=msg, bg="#3a1a1a", fg=CLR["text"],
                 font=FONT_SMALL, wraplength=500, justify="left").pack(anchor="w")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VERIFY
# ═══════════════════════════════════════════════════════════════════════════════

class VerifyTab(tk.Frame):
    def __init__(self, parent, loader: ComponentLoader, status: StatusBar, **kw):
        super().__init__(parent, bg=CLR["bg"], **kw)
        self.loader = loader
        self.status = status
        self._build()

    def _build(self):
        tk.Label(self, text="Verify Fingerprint", font=FONT_TITLE,
                 bg=CLR["bg"], fg=CLR["text"]).pack(anchor="w", padx=20, pady=(18, 4))
        tk.Label(self, text="Authenticate a query fingerprint against a stored template.",
                 font=FONT_BODY, bg=CLR["bg"], fg=CLR["text2"]).pack(anchor="w", padx=20)
        separator(self).pack(fill="x", padx=20, pady=10)

        card = make_card(self)
        card.pack(fill="x", padx=20, pady=6)

        # User ID
        section_label(card, "Claimed User ID").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self._uid_var = tk.StringVar()
        styled_entry(card, textvariable=self._uid_var, width=36).grid(
            row=1, column=0, sticky="w", pady=(0, 12))

        separator(card).grid(row=2, column=0, sticky="ew", pady=6)

        # Query image
        section_label(card, "Query Fingerprint Image").grid(row=3, column=0, sticky="w")
        self._path_var = tk.StringVar(value="No image selected")
        tk.Label(card, textvariable=self._path_var,
                 bg=CLR["surface"], fg=CLR["text2"], font=FONT_SMALL,
                 anchor="w", wraplength=480).grid(row=4, column=0, sticky="w", pady=(4, 8))

        btn_row = tk.Frame(card, bg=CLR["surface"])
        btn_row.grid(row=5, column=0, sticky="w", pady=(0, 10))
        styled_button(btn_row, "📂  Browse Image", self._browse_image).pack(side="left", padx=(0, 8))

        # Preview
        self._preview = ImagePreviewPanel(card, max_images=1)
        self._preview.grid(row=6, column=0, sticky="ew")

        separator(card).grid(row=7, column=0, sticky="ew", pady=(4, 8))

        # Image mode toggle
        section_label(card, "Image Type").grid(row=8, column=0, sticky="w")
        mode_frame = tk.Frame(card, bg=CLR["surface"])
        mode_frame.grid(row=9, column=0, sticky="w", pady=(4, 8))
        self._img_mode = tk.StringVar(value="sensor")
        tk.Radiobutton(mode_frame, text="📷  Sensor / Scanner image  (BMP, SOCOFing)",
                       variable=self._img_mode, value="sensor",
                       bg=CLR["surface"], fg=CLR["text"], selectcolor=CLR["surface2"],
                       activebackground=CLR["surface"], font=FONT_SMALL,
                       command=self._on_mode_change).pack(anchor="w")
        tk.Radiobutton(mode_frame, text="📱  Camera photo  (JPG/PNG from smartphone)",
                       variable=self._img_mode, value="camera",
                       bg=CLR["surface"], fg=CLR["text"], selectcolor=CLR["surface2"],
                       activebackground=CLR["surface"], font=FONT_SMALL,
                       command=self._on_mode_change).pack(anchor="w")
        self._mode_hint = tk.Label(card, text="",
                                    bg=CLR["surface"], fg=CLR["warning"], font=FONT_SMALL,
                                    wraplength=520, justify="left")
        self._mode_hint.grid(row=10, column=0, sticky="w")

        separator(card).grid(row=11, column=0, sticky="ew", pady=12)

        self._verify_btn = styled_button(card, "🔍  Verify", self._do_verify, color=CLR["accent"])
        self._verify_btn.grid(row=12, column=0, sticky="w")

        # Result
        self._result_frame = tk.Frame(card, bg=CLR["surface"])
        self._result_frame.grid(row=13, column=0, sticky="ew", pady=(12, 0))

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select query fingerprint",
            filetypes=[("Image files", "*.bmp *.BMP *.png *.PNG *.jpg *.jpeg *.tif *.tiff"), ("All files", "*.*")]
        )
        if path:
            self._query_path = path
            self._path_var.set(path)
            self._preview.set_paths([path])
            # Auto-detect camera photos by extension
            if path.lower().endswith(('.jpg', '.jpeg')):
                self._img_mode.set("camera")
                self._on_mode_change()

    def _on_mode_change(self):
        if self._img_mode.get() == "camera":
            self._mode_hint.configure(
                text="📱 Camera mode: image will be automatically inverted and "
                     "enhanced to match sensor-style before verification."
            )
        else:
            self._mode_hint.configure(text="")

    def _do_verify(self):
        user_id = self._uid_var.get().strip()
        path    = getattr(self, "_query_path", None)

        if not user_id:
            messagebox.showwarning("Missing input", "Please enter the claimed User ID.")
            return
        if not path:
            messagebox.showwarning("No image", "Please select a query fingerprint image.")
            return

        self._verify_btn.configure(state="disabled", text="⏳  Verifying…")
        self.status.busy(f"Verifying {user_id}…")
        for w in self._result_frame.winfo_children():
            w.destroy()

        use_camera = self._img_mode.get() == "camera"

        def worker():
            try:
                loader = self.loader
                if not loader.ready.is_set():
                    loader.ready.wait()
                if loader.error:
                    raise RuntimeError(loader.error)
                authenticator = (loader.camera_authenticator
                                 if use_camera else loader.authenticator)
                result = authenticator.authenticate(user_id, path)
                self.after(0, lambda: self._on_result(result))
            except Exception as exc:
                self.after(0, lambda: self._on_error(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_result(self, result):
        self._verify_btn.configure(state="normal", text="🔍  Verify")
        granted  = result.granted
        bg_color = "#1a3a2a" if granted else "#3a1a1a"
        brd      = CLR["accent2"] if granted else CLR["danger"]
        symbol   = "✅  GRANTED" if granted else "❌  DENIED"
        self.status.ok(symbol) if granted else self.status.err(symbol)

        for w in self._result_frame.winfo_children():
            w.destroy()
        res_card = tk.Frame(self._result_frame, bg=bg_color,
                            highlightthickness=1, highlightbackground=brd,
                            padx=14, pady=12)
        res_card.pack(fill="x")
        tk.Label(res_card, text=symbol, font=FONT_HEAD,
                 bg=bg_color, fg=brd).pack(anchor="w", pady=(0, 8))

        rows = [
            ("User ID",           result.user_id),
            ("Cosine Similarity", f"{result.cosine_sim:.4f}  (threshold {config.COSINE_THRESHOLD})"),
            ("Hamming Distance",  f"{result.hamming_dist:.4f}"),
            ("Reason",            result.reason or "—"),
        ]
        for label, val in rows:
            row = tk.Frame(res_card, bg=bg_color)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{label}:", width=20, anchor="w",
                     bg=bg_color, fg=CLR["text2"], font=FONT_SMALL).pack(side="left")
            tk.Label(row, text=str(val), anchor="w",
                     bg=bg_color, fg=CLR["text"], font=FONT_MONO).pack(side="left")

        # Similarity bar
        bar_frame = tk.Frame(res_card, bg=bg_color)
        bar_frame.pack(fill="x", pady=(10, 0))
        tk.Label(bar_frame, text="Similarity:", bg=bg_color,
                 fg=CLR["text2"], font=FONT_SMALL).pack(side="left")
        bar_bg = tk.Frame(bar_frame, bg=CLR["surface2"], height=12, width=300)
        bar_bg.pack(side="left", padx=8)
        fill_w = int(max(0, min(1, result.cosine_sim)) * 300)
        fill_clr = CLR["accent2"] if granted else CLR["danger"]
        tk.Frame(bar_bg, bg=fill_clr, height=12, width=fill_w).place(x=0, y=0)
        tk.Label(bar_frame, text=f"{result.cosine_sim*100:.1f}%",
                 bg=bg_color, fg=CLR["text"], font=FONT_SMALL).pack(side="left")

    def _on_error(self, msg: str):
        self._verify_btn.configure(state="normal", text="🔍  Verify")
        self.status.err(f"Error: {msg}")
        messagebox.showerror("Verification Error", msg)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — USERS (browse + delete)
# ═══════════════════════════════════════════════════════════════════════════════

class UsersTab(tk.Frame):
    def __init__(self, parent, loader: ComponentLoader, status: StatusBar, **kw):
        super().__init__(parent, bg=CLR["bg"], **kw)
        self.loader = loader
        self.status = status
        self._build()

    def _build(self):
        # Header row
        hdr = tk.Frame(self, bg=CLR["bg"])
        hdr.pack(fill="x", padx=20, pady=(18, 4))
        tk.Label(hdr, text="Enrolled Users", font=FONT_TITLE,
                 bg=CLR["bg"], fg=CLR["text"]).pack(side="left")
        styled_button(hdr, "🔄  Refresh", self.refresh, color=CLR["surface2"]).pack(side="right")

        tk.Label(self, text="Manage enrolled users stored in the encrypted database.",
                 font=FONT_BODY, bg=CLR["bg"], fg=CLR["text2"]).pack(anchor="w", padx=20)
        separator(self).pack(fill="x", padx=20, pady=10)

        # Treeview card
        card = make_card(self, padx=0, pady=0)
        card.pack(fill="both", expand=True, padx=20, pady=6)

        # Search bar
        search_bar = tk.Frame(card, bg=CLR["surface"], pady=8, padx=12)
        search_bar.pack(fill="x")
        tk.Label(search_bar, text="🔍", bg=CLR["surface"], fg=CLR["text2"],
                 font=FONT_BODY).pack(side="left", padx=(0, 6))
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._filter())
        styled_entry(search_bar, textvariable=self._search_var, width=30).pack(side="left")
        self._count_lbl = tk.Label(search_bar, text="",
                                   bg=CLR["surface"], fg=CLR["text2"], font=FONT_SMALL)
        self._count_lbl.pack(side="right", padx=8)

        # Treeview
        cols = ("user_id", "enrolled_at", "images_used")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("FP.Treeview",
                        background=CLR["surface2"],
                        foreground=CLR["text"],
                        fieldbackground=CLR["surface2"],
                        rowheight=28,
                        font=FONT_BODY)
        style.configure("FP.Treeview.Heading",
                        background=CLR["surface"],
                        foreground=CLR["accent"],
                        font=FONT_HEAD, relief="flat")
        style.map("FP.Treeview",
                  background=[("selected", CLR["accent"])],
                  foreground=[("selected", "#ffffff")])

        tree_frame = tk.Frame(card, bg=CLR["surface"])
        tree_frame.pack(fill="both", expand=True)

        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        self._tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            style="FP.Treeview", yscrollcommand=scrollbar.set
        )
        scrollbar.configure(command=self._tree.yview)

        self._tree.heading("user_id",     text="User ID")
        self._tree.heading("enrolled_at", text="Enrolled At")
        self._tree.heading("images_used", text="Images")
        self._tree.column("user_id",     width=200, anchor="w")
        self._tree.column("enrolled_at", width=230, anchor="w")
        self._tree.column("images_used", width=80,  anchor="center")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        # Action bar
        act_bar = tk.Frame(card, bg=CLR["surface"], pady=10, padx=12)
        act_bar.pack(fill="x")
        self._del_btn = styled_button(
            act_bar, "🗑  Delete Selected", self._delete_selected, color=CLR["danger"]
        )
        self._del_btn.pack(side="left", padx=(0, 8))
        self._del_btn.configure(state="disabled")

        styled_button(act_bar, "🗑  Delete ALL Users", self._delete_all,
                      color=CLR["danger"]).pack(side="left")

        # Detail panel
        self._detail_frame = tk.Frame(self, bg=CLR["surface"],
                                      highlightthickness=1,
                                      highlightbackground=CLR["border"],
                                      padx=14, pady=10)
        self._detail_frame.pack(fill="x", padx=20, pady=(0, 10))
        tk.Label(self._detail_frame, text="Select a user to see details.",
                 bg=CLR["surface"], fg=CLR["text2"], font=FONT_SMALL).pack(anchor="w")

        self._all_rows: list = []
        self.refresh()

    # ── data ──────────────────────────────────────────────────────────────────

    def refresh(self):
        self.status.busy("Loading users…")

        def worker():
            try:
                loader = self.loader
                if not loader.ready.is_set():
                    loader.ready.wait()
                if loader.error:
                    raise RuntimeError(loader.error)
                rows = loader.db.all_user_info()
                self.after(0, lambda: self._populate(rows))
            except Exception as exc:
                self.after(0, lambda: self.status.err(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _populate(self, rows: list):
        self._all_rows = rows
        self._render(rows)
        self.status.ok(f"{len(rows)} user(s) enrolled.")

    def _render(self, rows: list):
        self._tree.delete(*self._tree.get_children())
        for r in rows:
            enrolled_str = r["enrolled_at"][:19].replace("T", "  ")
            self._tree.insert("", "end", iid=r["user_id"],
                              values=(r["user_id"], enrolled_str, r["images_used"]))
        self._count_lbl.configure(text=f"{len(rows)} user(s)")

    def _filter(self):
        q = self._search_var.get().lower()
        filtered = [r for r in self._all_rows if q in r["user_id"].lower()] if q else self._all_rows
        self._render(filtered)

    def _on_select(self, _event):
        sel = self._tree.selection()
        self._del_btn.configure(state="normal" if sel else "disabled")
        for w in self._detail_frame.winfo_children():
            w.destroy()
        if not sel:
            tk.Label(self._detail_frame, text="Select a user to see details.",
                     bg=CLR["surface"], fg=CLR["text2"], font=FONT_SMALL).pack(anchor="w")
            return

        uid  = sel[0]
        info = next((r for r in self._all_rows if r["user_id"] == uid), None)
        if not info:
            return

        tk.Label(self._detail_frame, text=f"👤  {uid}",
                 bg=CLR["surface"], fg=CLR["text"], font=FONT_HEAD).pack(anchor="w")
        for k, v in info.items():
            row = tk.Frame(self._detail_frame, bg=CLR["surface"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{k}:", width=16, anchor="w",
                     bg=CLR["surface"], fg=CLR["text2"], font=FONT_SMALL).pack(side="left")
            tk.Label(row, text=str(v)[:60], anchor="w",
                     bg=CLR["surface"], fg=CLR["text"], font=FONT_MONO).pack(side="left")

    # ── actions ───────────────────────────────────────────────────────────────

    def _delete_selected(self):
        sel = self._tree.selection()
        if not sel:
            return
        uid = sel[0]
        if not messagebox.askyesno("Confirm Delete",
                f"Permanently delete user '{uid}' from the database?\n\nThis cannot be undone."):
            return
        self.status.busy(f"Deleting {uid}…")

        def worker():
            try:
                ok = self.loader.db.delete(uid)
                self.after(0, lambda: self._after_delete(uid, ok))
            except Exception as exc:
                self.after(0, lambda: self.status.err(str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _after_delete(self, uid: str, ok: bool):
        if ok:
            self.status.ok(f"Deleted '{uid}'.")
        else:
            self.status.err(f"User '{uid}' not found.")
        self.refresh()

    def _delete_all(self):
        n = len(self._all_rows)
        if n == 0:
            messagebox.showinfo("Empty", "No users to delete.")
            return
        if not messagebox.askyesno("Confirm Delete ALL",
                f"Delete ALL {n} enrolled users?\n\nThis CANNOT be undone."):
            return

        def worker():
            try:
                for r in self._all_rows:
                    self.loader.db.delete(r["user_id"])
                self.after(0, lambda: self.status.ok("All users deleted."))
                self.after(0, self.refresh)
            except Exception as exc:
                self.after(0, lambda: self.status.err(str(exc)))

        threading.Thread(target=worker, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRAIN
# ═══════════════════════════════════════════════════════════════════════════════

class TrainTab(tk.Frame):
    def __init__(self, parent, loader: ComponentLoader, status: StatusBar, **kw):
        super().__init__(parent, bg=CLR["bg"], **kw)
        self.loader = loader
        self.status = status
        self._loss_vals: List[float] = []
        self._build()

    def _build(self):
        tk.Label(self, text="Train CNN Feature Extractor", font=FONT_TITLE,
                 bg=CLR["bg"], fg=CLR["text"]).pack(anchor="w", padx=20, pady=(18, 4))
        tk.Label(self, text="Fine-tune MobileNetV2 on SOCOFing using triplet loss.",
                 font=FONT_BODY, bg=CLR["bg"], fg=CLR["text2"]).pack(anchor="w", padx=20)
        separator(self).pack(fill="x", padx=20, pady=10)

        card = make_card(self)
        card.pack(fill="x", padx=20, pady=6)

        # Dataset path
        section_label(card, "SOCOFing Dataset Directory").grid(
            row=0, column=0, sticky="w", columnspan=2)
        path_row = tk.Frame(card, bg=CLR["surface"])
        path_row.grid(row=1, column=0, sticky="ew", columnspan=2, pady=(4, 10))
        self._dir_var = tk.StringVar(value=str(config.SOCOFING_REAL))
        styled_entry(path_row, textvariable=self._dir_var, width=50).pack(side="left", padx=(0, 8))
        styled_button(path_row, "Browse", self._browse_dir, color=CLR["surface2"]).pack(side="left")

        separator(card).grid(row=2, column=0, sticky="ew", columnspan=2, pady=8)

        # Hyperparams
        section_label(card, "Hyperparameters").grid(row=3, column=0, sticky="w", columnspan=2)

        params = [
            ("Epochs",          "epochs_var",  str(config.TRAIN_EPOCHS)),
            ("Steps / Epoch",   "steps_var",   str(config.TRAIN_STEPS)),
            ("Batch Size",      "batch_var",   str(config.TRAIN_BATCH_SIZE)),
            ("Triplet Margin",  "margin_var",  str(config.TRIPLET_MARGIN)),
            ("Fine-tune Layers","ft_var",      str(config.FINE_TUNE_LAYERS)),
        ]
        for i, (label, attr, default) in enumerate(params):
            r = 4 + i
            tk.Label(card, text=label, bg=CLR["surface"], fg=CLR["text"],
                     font=FONT_BODY, width=18, anchor="w").grid(row=r, column=0, sticky="w")
            var = tk.StringVar(value=default)
            setattr(self, attr, var)
            styled_entry(card, textvariable=var, width=12).grid(row=r, column=1, sticky="w", pady=2)

        separator(card).grid(row=9, column=0, sticky="ew", columnspan=2, pady=8)

        # Buttons
        btn_row = tk.Frame(card, bg=CLR["surface"])
        btn_row.grid(row=10, column=0, sticky="w", columnspan=2)
        self._train_btn = styled_button(btn_row, "🚀  Start Training", self._do_train,
                                         color=CLR["accent"])
        self._train_btn.pack(side="left", padx=(0, 8))
        self._stop_btn  = styled_button(btn_row, "■  Stop", self._stop_training,
                                         color=CLR["danger"])
        self._stop_btn.pack(side="left")
        self._stop_btn.configure(state="disabled")

        # Progress
        self._progress = ttk.Progressbar(card, mode="indeterminate", length=400)
        self._progress.grid(row=11, column=0, sticky="w", columnspan=2, pady=(12, 4))

        self._epoch_lbl = tk.Label(card, text="", bg=CLR["surface"],
                                   fg=CLR["text2"], font=FONT_SMALL)
        self._epoch_lbl.grid(row=12, column=0, sticky="w", columnspan=2)

        # Loss log
        section_label(card, "Training Log").grid(row=13, column=0, sticky="w",
                                                  columnspan=2, pady=(10, 4))
        self._log_text = tk.Text(card, height=14, width=72,
                                  bg=CLR["entry_bg"], fg=CLR["text"], font=FONT_MONO,
                                  state="disabled", relief="flat",
                                  highlightthickness=1, highlightbackground=CLR["border"])
        self._log_text.grid(row=14, column=0, sticky="ew", columnspan=2)

        self._stop_flag = threading.Event()

    def _browse_dir(self):
        d = filedialog.askdirectory(title="Select SOCOFing/Real directory")
        if d:
            self._dir_var.set(d)

    def _log(self, msg: str):
        self._log_text.configure(state="normal")
        self._log_text.insert("end", msg + "\n")
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def _do_train(self):
        real_dir = Path(self._dir_var.get().strip())
        if not real_dir.exists():
            messagebox.showerror("Not found", f"Directory not found:\n{real_dir}")
            return

        try:
            epochs = int(self.epochs_var.get())
            steps  = int(self.steps_var.get())
            batch  = int(self.batch_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Epochs, steps and batch must be integers.")
            return

        self._train_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._progress.start(12)
        self._stop_flag.clear()
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")
        self.status.busy("Training started…")

        def worker():
            try:
                loader = self.loader
                if not loader.ready.is_set():
                    loader.ready.wait()
                if loader.error:
                    raise RuntimeError(loader.error)

                # Load dataset
                self.after(0, lambda: self._log(f"Loading images from {real_dir}…"))
                all_imgs = sorted(glob.glob(str(real_dir / "*.BMP")))
                if not all_imgs:
                    all_imgs = sorted(glob.glob(str(real_dir / "*.png")))
                if not all_imgs:
                    raise FileNotFoundError("No .BMP or .png images found in that directory.")

                subject_imgs: dict = {}
                for p in all_imgs:
                    fname = Path(p).stem
                    sid   = int(fname.split("__")[0]) if "__" in fname else \
                            int(re.split(r"[^0-9]", fname)[0])
                    subject_imgs.setdefault(sid, []).append(p)

                self.after(0, lambda: self._log(
                    f"Found {len(all_imgs)} images across {len(subject_imgs)} subjects."
                ))

                # Patch config for this run
                import config as cfg
                cfg.TRAIN_EPOCHS     = epochs
                cfg.TRAIN_STEPS      = steps
                cfg.TRAIN_BATCH_SIZE = batch

                # Custom Keras callback to stream loss into the log
                import tensorflow as tf

                class GUICallback(tf.keras.callbacks.Callback):
                    def __init__(cb_self):
                        super().__init__()
                        cb_self.epoch = 0

                    def on_epoch_begin(cb_self, epoch, logs=None):
                        cb_self.epoch = epoch + 1
                        msg = f"\nEpoch {epoch+1}/{epochs}"
                        self.after(0, lambda m=msg: self._log(m))
                        self.after(0, lambda e=epoch+1: self._epoch_lbl.configure(
                            text=f"Epoch {e}/{epochs}"
                        ))
                        if self._stop_flag.is_set():
                            cb_self.model.stop_training = True

                    def on_batch_end(cb_self, batch, logs=None):
                        if logs and "loss" in logs:
                            loss = logs["loss"]
                            msg  = f"  step {batch:4d}  loss={loss:.6f}"
                            self.after(0, lambda m=msg: self._log(m))
                        if self._stop_flag.is_set():
                            cb_self.model.stop_training = True

                # Run training using the corrected two-phase pipeline
                from train_full import (
                    build_embedding_model, build_triplet_model,
                    triplet_loss, BalancedTripletGenerator
                )

                # ── Phase 1: head warm-up (backbone frozen) ───────────────
                self.after(0, lambda: self._log("\n── Phase 1: head warm-up (backbone frozen) ──"))
                emb_p1   = build_embedding_model(freeze_backbone=True)
                trip_p1  = build_triplet_model(emb_p1)
                trip_p1.compile(
                    optimizer=tf.keras.optimizers.Adam(1e-3),
                    loss=triplet_loss()
                )
                gen_train = BalancedTripletGenerator(
                    subject_imgs, loader.preprocessor,
                    batch_size=batch, steps=steps, seed=42
                ).as_dataset()
                phase1_epochs = max(1, epochs // 4)
                trip_p1.fit(gen_train, epochs=phase1_epochs,
                            callbacks=[GUICallback()], verbose=0)

                # Transfer head weights by layer name (safe, no index guessing)
                hw_256 = emb_p1.get_layer("proj_256").get_weights()
                hw_128 = emb_p1.get_layer("proj_128").get_weights()
                hw_bn  = emb_p1.get_layer("bn_256").get_weights()

                # ── Phase 2: fine-tune last N backbone layers ─────────────
                self.after(0, lambda: self._log("\n── Phase 2: backbone fine-tuning ──"))
                phase2_epochs = epochs - phase1_epochs
                if phase2_epochs > 0:
                    import config as cfg
                    emb_p2  = build_embedding_model(
                        freeze_backbone=False,
                        fine_tune_layers=cfg.FINE_TUNE_LAYERS
                    )
                    emb_p2.get_layer("proj_256").set_weights(hw_256)
                    emb_p2.get_layer("proj_128").set_weights(hw_128)
                    emb_p2.get_layer("bn_256").set_weights(hw_bn)
                    trip_p2 = build_triplet_model(emb_p2)
                    trip_p2.compile(
                        optimizer=tf.keras.optimizers.Adam(1e-5),
                        loss=triplet_loss()
                    )
                    gen_train2 = BalancedTripletGenerator(
                        subject_imgs, loader.preprocessor,
                        batch_size=batch, steps=steps, seed=99
                    ).as_dataset()
                    trip_p2.fit(gen_train2, epochs=phase2_epochs,
                                callbacks=[GUICallback()], verbose=0)
                    # Save phase-2 weights into the shared extractor
                    emb_p2.save_weights(str(config.MODEL_WEIGHTS))
                    # Reload into the live extractor so subsequent enrollments use new weights
                    loader.extractor.model.set_weights(emb_p2.get_weights())
                else:
                    emb_p1.save_weights(str(config.MODEL_WEIGHTS))
                    loader.extractor.model.set_weights(emb_p1.get_weights())

                self.after(0, lambda: self._on_train_done("Training complete. Weights saved."))
            except Exception as exc:
                err = traceback.format_exc()
                self.after(0, lambda: self._on_train_done(f"ERROR: {exc}", error=True))
                self.after(0, lambda: self._log(err))

        threading.Thread(target=worker, daemon=True).start()

    def _stop_training(self):
        self._stop_flag.set()
        self._log(">>> Stop requested…")
        self.status.info("Stopping after current step…")

    def _on_train_done(self, msg: str, error: bool = False):
        self._progress.stop()
        self._train_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._log(f"\n{'='*50}\n{msg}\n{'='*50}")
        if error:
            self.status.err(msg)
        else:
            self.status.ok(msg)
        self._epoch_lbl.configure(text=msg)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════

class AboutTab(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CLR["bg"], **kw)
        self._build()

    def _build(self):
        card = make_card(self)
        card.pack(fill="x", padx=20, pady=20)

        tk.Label(card, text="🔐  Fingerprint Biometric Cryptosystem",
                 font=FONT_TITLE, bg=CLR["surface"], fg=CLR["accent"]).pack(anchor="w", pady=(0, 8))

        lines = [
            ("Architecture",   "MobileNetV2 → 128-D L2-normalised embedding"),
            ("Binarisation",   "Per-dimension median thresholding + Fuzzy Extractor"),
            ("Key Derivation", "SHA-256(salt || packed_bits)  →  32-byte AES key"),
            ("Encryption",     "AES-256-CBC with fresh IV per enrollment"),
            ("Database",       "SQLite — ciphertext + IV + helper data only"),
            ("Dataset",        "SOCOFing (600 subjects × 10 fingers × 6000 images)"),
        ]

        separator(card).pack(fill="x", pady=8)
        for label, val in lines:
            row = tk.Frame(card, bg=CLR["surface"])
            row.pack(fill="x", pady=3)
            tk.Label(row, text=f"{label}:", width=18, anchor="w",
                     bg=CLR["surface"], fg=CLR["accent"], font=FONT_BODY).pack(side="left")
            tk.Label(row, text=val, anchor="w",
                     bg=CLR["surface"], fg=CLR["text"], font=FONT_MONO).pack(side="left")

        separator(card).pack(fill="x", pady=8)

        import sys
        tk.Label(card, text=f"Python {sys.version.split()[0]}  |  "
                             f"DB: {config.DB_PATH}",
                 bg=CLR["surface"], fg=CLR["text2"], font=FONT_SMALL).pack(anchor="w")


# ═══════════════════════════════════════════════════════════════════════════════
# SPLASH / LOADING SCREEN
# ═══════════════════════════════════════════════════════════════════════════════

class SplashScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("")
        self.configure(bg=CLR["bg"])
        self.resizable(False, False)
        try:
            self.overrideredirect(True)
        except Exception:
            pass   # macOS sometimes rejects overrideredirect — safe to ignore

        w, h = 400, 220
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        tk.Label(self, text="🔐", font=("Segoe UI", 48),
                 bg=CLR["bg"], fg=CLR["accent"]).pack(pady=(30, 8))
        tk.Label(self, text="Fingerprint Cryptosystem",
                 font=FONT_TITLE, bg=CLR["bg"], fg=CLR["text"]).pack()
        self._status = tk.Label(self, text="Loading ML components…",
                                font=FONT_SMALL, bg=CLR["bg"], fg=CLR["text2"])
        self._status.pack(pady=8)
        self._bar = ttk.Progressbar(self, mode="indeterminate", length=300)
        self._bar.pack(pady=4)
        self._bar.start(10)

    def set_status(self, msg: str):
        self._status.configure(text=msg)
        self.update_idletasks()

    def close(self):
        self._bar.stop()
        self.destroy()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class FingerprintApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Fingerprint Biometric Cryptosystem")
        self.configure(bg=CLR["bg"])
        self.minsize(760, 620)

        # Center window
        w, h = 900, 700
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        # Start loading components in background
        self.loader = ComponentLoader.get()

        # Show splash while loading
        self._splash = SplashScreen(self)
        self.withdraw()
        self.after(100, self._check_loader)

    def _check_loader(self):
        if self.loader.ready.is_set():
            self._splash.close()
            if self.loader.error:
                messagebox.showerror("Startup Error",
                    f"Failed to load ML components:\n\n{self.loader.error[:800]}")
            self.deiconify()
            self._build_ui()
        else:
            self._splash.set_status("Loading TensorFlow and CNN…")
            self.after(200, self._check_loader)

    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self, bg=CLR["surface2"], pady=10)
        title_bar.pack(fill="x")
        tk.Label(title_bar, text="🔐  Fingerprint Biometric Cryptosystem",
                 font=FONT_HEAD, bg=CLR["surface2"], fg=CLR["text"],
                 padx=16).pack(side="left")

        # Status bar (shared across all tabs)
        self._status = StatusBar(self)
        self._status.pack(side="bottom", fill="x")

        # Notebook
        style = ttk.Style()
        style.configure("FP.TNotebook",
                        background=CLR["bg"], borderwidth=0)
        style.configure("FP.TNotebook.Tab",
                        background=CLR["surface"],
                        foreground=CLR["text2"],
                        padding=[16, 8],
                        font=FONT_BODY)
        style.map("FP.TNotebook.Tab",
                  background=[("selected", CLR["surface2"])],
                  foreground=[("selected", CLR["accent"])])

        nb = ttk.Notebook(self, style="FP.TNotebook")
        nb.pack(fill="both", expand=True)

        tabs = [
            ("  Enroll  ",  EnrollTab(nb,  self.loader, self._status)),
            ("  Verify  ",  VerifyTab(nb,  self.loader, self._status)),
            ("  Users   ",  UsersTab(nb,   self.loader, self._status)),
            ("  Train   ",  TrainTab(nb,   self.loader, self._status)),
            ("  About   ",  AboutTab(nb)),
        ]
        for title, frame in tabs:
            nb.add(frame, text=title)

        # Show startup status
        user_count = 0
        try:
            user_count = self.loader.db.user_count()
        except Exception:
            pass
        self._status.ok(f"System ready — {user_count} user(s) enrolled.")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = FingerprintApp()
    app.mainloop()