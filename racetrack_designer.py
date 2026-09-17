import tkinter as tk
from tkinter import filedialog
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from PIL import Image, ImageTk
import cv2
import datetime

SNAP_THRESHOLD = 1.0  # metres
POINT_SELECT_RADIUS = 0.8  # metres — click within this to select a waypoint
OBSTACLE_SNAP_RADIUS = 0.6  # metres — cursor turns orange when near obstacle


class RacetrackDesigner:
    def __init__(self, root):
        self.root = root
        self.root.title("Racetrack Designer")

        # ── outer layout ──────────────────────────────────────────────────────
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = tk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # mode toolbar
        self._build_mode_toolbar(left_frame)

        # canvas
        self.canvas = tk.Canvas(left_frame, width=900, height=650, bg='white',
                                cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # status bar
        self.status_var = tk.StringVar(value="X: —  Y: —  |  Mode: Draw Track")
        status_bar = tk.Label(left_frame, textvariable=self.status_var,
                              anchor='w', relief=tk.SUNKEN, bd=1)
        status_bar.pack(fill=tk.X)

        # control panel
        ctrl = tk.Frame(main_frame, padx=10, pady=10)
        ctrl.pack(side=tk.RIGHT, fill=tk.Y)
        self._build_control_panel(ctrl)

        # ── state ─────────────────────────────────────────────────────────────
        self.points = []           # [(real_x, real_y), ...]
        self.point_widths = []     # [(left_w, right_w), ...] parallel to points
        self.obstacles = []        # [(real_x, real_y, radius), ...]
        self.closed = False
        self.grid_image = None
        self.cursor_crosshair = []
        self.preview_point = None
        self.selected_point_idx = None
        self.background_image = None   # PIL Image
        self.background_photo = None   # ImageTk (kept alive)
        self.calib_points = []         # canvas (x,y) clicks during CALIBRATE mode

        # ── bindings ──────────────────────────────────────────────────────────
        self.canvas.bind("<Button-1>",   self.on_left_click)
        self.canvas.bind("<Button-3>",   self.on_right_click)
        self.canvas.bind("<Motion>",     self.on_mouse_move)
        self.canvas.bind("<Configure>",  lambda e: self.update_display())
        # Linux scroll
        self.canvas.bind("<Button-4>",   lambda e: self.on_scroll(e, +1))
        self.canvas.bind("<Button-5>",   lambda e: self.on_scroll(e, -1))
        # Windows/macOS scroll
        self.canvas.bind("<MouseWheel>", self.on_scroll_wheel)

        for var in [self.resolution, self.real_width, self.real_height,
                    self.left_width, self.right_width, self.boundary_width,
                    self.obstacle_radius]:
            var.trace_add('write', lambda *_: self.update_display())

    # ── UI builders ───────────────────────────────────────────────────────────

    def _build_mode_toolbar(self, parent):
        bar = tk.Frame(parent, bd=1, relief=tk.RIDGE)
        bar.pack(fill=tk.X)
        self.mode = tk.StringVar(value='TRACK')
        modes = [('Draw Track', 'TRACK'), ('Place Obstacle', 'OBSTACLE'),
                 ('Select/Edit', 'SELECT'), ('Calibrate', 'CALIBRATE')]
        for label, value in modes:
            tk.Radiobutton(bar, text=label, variable=self.mode, value=value,
                           indicatoron=True,
                           command=self._on_mode_change).pack(side=tk.LEFT, padx=6, pady=2)
        tk.Label(bar, text=' ').pack(side=tk.LEFT, expand=True)

    def _build_control_panel(self, ctrl):
        row = 0

        def section(text):
            nonlocal row
            tk.Label(ctrl, text=text, font=('', 9, 'bold'),
                     anchor='w').grid(row=row, column=0, columnspan=2,
                                      sticky='w', pady=(8, 2))
            row += 1

        def entry(label, var):
            nonlocal row
            tk.Label(ctrl, text=label).grid(row=row, column=0, sticky='w')
            e = tk.Entry(ctrl, textvariable=var, width=10)
            e.grid(row=row, column=1, sticky='e')
            row += 1
            return e

        section("Space")
        self.resolution  = tk.DoubleVar(value=0.06250)
        self.real_width  = tk.DoubleVar(value=60.0)
        self.real_height = tk.DoubleVar(value=36.0)
        entry("Resolution (m/cell):", self.resolution)
        entry("Real Width (m):",      self.real_width)
        entry("Real Height (m):",     self.real_height)

        section("Track")
        self.left_width     = tk.DoubleVar(value=0.6)
        self.right_width    = tk.DoubleVar(value=0.6)
        self.boundary_width = tk.DoubleVar(value=0.1)
        entry("Left Width (m):",     self.left_width)
        entry("Right Width (m):",    self.right_width)
        entry("Boundary Width (m):", self.boundary_width)

        section("Obstacle")
        self.obstacle_radius = tk.DoubleVar(value=0.3)
        entry("Obstacle Radius (m):", self.obstacle_radius)

        section("Background")
        tk.Button(ctrl, text="Load Floor Plan",
                  command=self.load_background).grid(row=row, column=0,
                                                      columnspan=2, sticky='ew')
        row += 1
        self.bg_opacity = tk.IntVar(value=40)
        tk.Label(ctrl, text="BG Opacity (%):").grid(row=row, column=0, sticky='w')
        tk.Scale(ctrl, from_=0, to=100, orient=tk.HORIZONTAL,
                 variable=self.bg_opacity,
                 command=lambda _: self.update_display()).grid(row=row, column=1,
                                                               sticky='ew')
        row += 1
        self.bg_rotation = tk.IntVar(value=0)  # degrees, multiple of 90
        rot_frame = tk.Frame(ctrl)
        rot_frame.grid(row=row, column=0, columnspan=2, sticky='ew')
        tk.Label(rot_frame, text="Rotate:").pack(side=tk.LEFT)
        tk.Button(rot_frame, text="↺ 90° CCW",
                  command=lambda: self._rotate_bg(-90)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(rot_frame, text="↻ 90° CW",
                  command=lambda: self._rotate_bg(+90)).pack(side=tk.LEFT, expand=True, fill=tk.X)
        row += 1
        self.calib_units = tk.StringVar(value='m')
        calib_frame = tk.Frame(ctrl)
        calib_frame.grid(row=row, column=0, columnspan=2, sticky='ew')
        tk.Label(calib_frame, text="Calibrate:").pack(side=tk.LEFT)
        tk.Radiobutton(calib_frame, text="m",  variable=self.calib_units, value='m').pack(side=tk.LEFT)
        tk.Radiobutton(calib_frame, text="ft", variable=self.calib_units, value='ft').pack(side=tk.LEFT)
        row += 1
        self.calib_status = tk.StringVar(value="Switch to Calibrate mode, then click 2 points")
        tk.Label(ctrl, textvariable=self.calib_status, wraplength=180,
                 fg='gray40', font=('', 8)).grid(row=row, column=0, columnspan=2, sticky='w')
        row += 1

        # selected-point width panel (hidden until a point is selected)
        section("Selected Point Width")
        self.sel_left_var  = tk.DoubleVar(value=0.6)
        self.sel_right_var = tk.DoubleVar(value=0.6)
        self.sel_left_entry  = entry("Left Width (m):",  self.sel_left_var)
        self.sel_right_entry = entry("Right Width (m):", self.sel_right_var)
        self.sel_left_var.trace_add('write',  self._sel_width_changed)
        self.sel_right_var.trace_add('write', self._sel_width_changed)

        self.apply_all_btn = tk.Button(ctrl, text="Apply to All",
                                       command=self._apply_width_to_all)
        self.apply_all_btn.grid(row=row, column=0, columnspan=2, sticky='ew')
        row += 1

        section("Actions")
        self.save_button = tk.Button(ctrl, text="Save", state=tk.DISABLED,
                                     command=self.save_grid)
        self.save_button.grid(row=row, column=0, columnspan=2, sticky='ew')
        row += 1
        self.delete_button = tk.Button(ctrl, text="Delete Last Point",
                                       state=tk.DISABLED,
                                       command=self.delete_last_point)
        self.delete_button.grid(row=row, column=0, columnspan=2, sticky='ew')
        row += 1
        tk.Button(ctrl, text="Clear All",
                  command=self.clear_all).grid(row=row, column=0,
                                               columnspan=2, sticky='ew')
        row += 1

    # ── mode ──────────────────────────────────────────────────────────────────

    def _on_mode_change(self):
        if self.mode.get() != 'SELECT':
            self.selected_point_idx = None
        if self.mode.get() != 'CALIBRATE':
            self.calib_points = []
        else:
            self.calib_status.set("Click point 1 of 2 on the floor plan")
        self.update_display()
        self._update_status(None, None)

    # ── coordinate helpers ────────────────────────────────────────────────────

    def real_to_canvas(self, rx, ry):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        return (rx / self.real_width.get()) * cw, (ry / self.real_height.get()) * ch

    def canvas_to_real(self, x, y):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        return (x / cw) * self.real_width.get(), (y / ch) * self.real_height.get()

    # ── event handlers ────────────────────────────────────────────────────────

    def on_left_click(self, event):
        m = self.mode.get()
        if m == 'TRACK':
            self._track_click(event)
        elif m == 'OBSTACLE':
            self._obstacle_click(event)
        elif m == 'SELECT':
            self._select_click(event)
        elif m == 'CALIBRATE':
            self._calib_click(event)

    def on_right_click(self, event):
        if self.mode.get() == 'OBSTACLE':
            rx, ry = self.canvas_to_real(event.x, event.y)
            # remove nearest obstacle within snap radius
            for i, (ox, oy, _r) in enumerate(self.obstacles):
                if np.hypot(rx - ox, ry - oy) < OBSTACLE_SNAP_RADIUS:
                    self.obstacles.pop(i)
                    self.update_display()
                    return

    def on_mouse_move(self, event):
        rx, ry = self.canvas_to_real(event.x, event.y)
        self._update_status(rx, ry)
        self._draw_crosshair(event.x, event.y, rx, ry)
        if self.mode.get() == 'TRACK' and not self.closed:
            self._update_track_preview(rx, ry)
        else:
            self.preview_point = None
            self.update_display()

    def on_scroll(self, event, direction):
        step = 0.05 * direction
        shift = (event.state & 0x1) != 0
        ctrl  = (event.state & 0x4) != 0
        self._apply_scroll_step(step, shift, ctrl)

    def on_scroll_wheel(self, event):
        direction = 1 if event.delta > 0 else -1
        step = 0.05 * direction
        shift = (event.state & 0x1) != 0
        ctrl  = (event.state & 0x4) != 0
        self._apply_scroll_step(step, shift, ctrl)

    def _apply_scroll_step(self, step, shift_held, ctrl_held):
        """Adjust track width via scroll wheel.
        Shift = left only, Ctrl = right only, neither = both."""
        if self.selected_point_idx is not None and self.mode.get() == 'SELECT':
            idx = self.selected_point_idx
            lw, rw = self.point_widths[idx]
            if not ctrl_held:
                lw = max(0.05, lw + step)
            if not shift_held:
                rw = max(0.05, rw + step)
            self.point_widths[idx] = (lw, rw)
            self._sync_sel_vars_from_point(idx)
        else:
            if not ctrl_held:
                self.left_width.set(max(0.05, round(self.left_width.get() + step, 4)))
            if not shift_held:
                self.right_width.set(max(0.05, round(self.right_width.get() + step, 4)))
        self.update_display()

    # ── track placement ───────────────────────────────────────────────────────

    def _update_track_preview(self, rx, ry):
        if len(self.points) >= 3:
            fx, fy = self.points[0]
            if np.hypot(rx - fx, ry - fy) < SNAP_THRESHOLD:
                self.preview_point = (fx, fy)
                self.update_display()
                return
        self.preview_point = (rx, ry)
        self.update_display()

    def _track_click(self, event):
        if self.closed:
            return
        rx, ry = self.canvas_to_real(event.x, event.y)
        if len(self.points) >= 3:
            fx, fy = self.points[0]
            if np.hypot(rx - fx, ry - fy) < SNAP_THRESHOLD:
                rx, ry = fx, fy
                self.closed = True
                self.points.append((rx, ry))
                self.point_widths.append(self.point_widths[0])
                self.preview_point = None
                self.save_button.config(state=tk.NORMAL)
                self.delete_button.config(state=tk.NORMAL)
                self.update_display()
                return
        self.points.append((rx, ry))
        self.point_widths.append((self.left_width.get(), self.right_width.get()))
        self.delete_button.config(state=tk.NORMAL)
        self.update_display()

    def delete_last_point(self):
        if not self.points:
            return
        if self.closed:
            self.points.pop()
            self.point_widths.pop()
            self.closed = False
            self.save_button.config(state=tk.DISABLED)
        else:
            self.points.pop()
            self.point_widths.pop()
        if not self.points:
            self.delete_button.config(state=tk.DISABLED)
        self.update_display()

    def clear_all(self):
        self.points.clear()
        self.point_widths.clear()
        self.obstacles.clear()
        self.closed = False
        self.selected_point_idx = None
        self.preview_point = None
        self.save_button.config(state=tk.DISABLED)
        self.delete_button.config(state=tk.DISABLED)
        self.update_display()

    # ── obstacle placement ────────────────────────────────────────────────────

    def _obstacle_click(self, event):
        rx, ry = self.canvas_to_real(event.x, event.y)
        self.obstacles.append((rx, ry, self.obstacle_radius.get()))
        self.update_display()

    # ── select/edit mode ──────────────────────────────────────────────────────

    def _select_click(self, event):
        rx, ry = self.canvas_to_real(event.x, event.y)
        best_idx, best_dist = None, float('inf')
        # skip the duplicated closing point
        unique_pts = self.points[:-1] if self.closed else self.points
        for i, (px, py) in enumerate(unique_pts):
            d = np.hypot(rx - px, ry - py)
            if d < POINT_SELECT_RADIUS and d < best_dist:
                best_idx, best_dist = i, d
        self.selected_point_idx = best_idx
        if best_idx is not None:
            self._sync_sel_vars_from_point(best_idx)
        self.update_display()

    def _sync_sel_vars_from_point(self, idx):
        lw, rw = self.point_widths[idx]
        # block traces while syncing to avoid infinite loop
        self._sel_syncing = True
        self.sel_left_var.set(round(lw, 4))
        self.sel_right_var.set(round(rw, 4))
        self._sel_syncing = False

    def _sel_width_changed(self, *_):
        if getattr(self, '_sel_syncing', False):
            return
        if self.selected_point_idx is None:
            return
        try:
            lw = float(self.sel_left_var.get())
            rw = float(self.sel_right_var.get())
        except (ValueError, tk.TclError):
            return
        idx = self.selected_point_idx
        self.point_widths[idx] = (lw, rw)
        # if closed loop, keep first==last width in sync
        if self.closed and idx == 0:
            self.point_widths[-1] = (lw, rw)
        elif self.closed and idx == len(self.point_widths) - 1:
            self.point_widths[0] = (lw, rw)
        self.update_display()

    def _apply_width_to_all(self):
        if self.selected_point_idx is None:
            return
        lw, rw = self.point_widths[self.selected_point_idx]
        self.point_widths = [(lw, rw)] * len(self.point_widths)
        self.left_width.set(lw)
        self.right_width.set(rw)
        self.update_display()

    # ── calibration ───────────────────────────────────────────────────────────

    def _calib_click(self, event):
        self.calib_points.append((event.x, event.y))
        if len(self.calib_points) == 1:
            self.calib_status.set("Click point 2 of 2")
            self.update_display()
            return
        # Two points collected — ask for real-world distance
        (x1, y1), (x2, y2) = self.calib_points
        px_dist = np.hypot(x2 - x1, y2 - y1)
        if px_dist < 5:
            self.calib_points = []
            self.calib_status.set("Points too close — try again")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Calibrate scale")
        dialog.wait_visibility()
        dialog.grab_set()
        unit = self.calib_units.get()
        tk.Label(dialog, text=f"Real distance between the two points ({unit}):").pack(padx=12, pady=(12, 4))
        dist_var = tk.DoubleVar(value=20.0 if unit == 'ft' else 6.0)
        tk.Entry(dialog, textvariable=dist_var, width=10).pack(padx=12)

        def apply():
            try:
                d = float(dist_var.get())
            except ValueError:
                return
            if d <= 0:
                return
            d_m = d / 3.28084 if unit == 'ft' else d
            cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
            (x1, y1), (x2, y2) = self.calib_points
            dx_px = abs(x2 - x1)
            dy_px = abs(y2 - y1)

            # Decompose: the real vector has the same angle as the pixel vector.
            # Each axis is adjusted only by its own component so that a purely
            # horizontal measurement corrects real_width and a purely vertical
            # one corrects real_height independently.
            angle = np.arctan2(dy_px, dx_px)  # 0 = horizontal, π/2 = vertical
            real_dx_component = d_m * np.cos(angle)  # how much of d is in X
            real_dy_component = d_m * np.sin(angle)  # how much of d is in Y

            rw = self.real_width.get()
            rh = self.real_height.get()
            if dx_px > 1:
                current_dx_m = dx_px / cw * rw
                self.real_width.set(round(rw * real_dx_component / current_dx_m, 3))
            if dy_px > 1:
                current_dy_m = dy_px / ch * rh
                self.real_height.set(round(rh * real_dy_component / current_dy_m, 3))

            self.calib_points = []
            self.calib_status.set(f"Calibrated: {d}{unit} = {d_m:.2f}m  ✓")
            self.mode.set('TRACK')
            dialog.destroy()
            self.update_display()

        tk.Button(dialog, text="Apply", command=apply).pack(pady=8)
        dialog.wait_window()
        if self.calib_points:   # dismissed without applying
            self.calib_points = []
            self.calib_status.set("Cancelled — click 2 points to try again")
            self.update_display()

    # ── background image ──────────────────────────────────────────────────────

    def load_background(self):
        path = filedialog.askopenfilename(
            title="Select floor plan image or PDF",
            filetypes=[("Floor plans", "*.png *.jpg *.jpeg *.bmp *.tiff *.pdf"),
                       ("PDF", "*.pdf"), ("Images", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                       ("All", "*.*")])
        if not path:
            return
        if path.lower().endswith('.pdf'):
            self.background_image = self._render_pdf_page(path)
        else:
            self.background_image = Image.open(path).convert('RGB')
        self.update_display()

    def _rotate_bg(self, degrees):
        self.bg_rotation.set((self.bg_rotation.get() + degrees) % 360)
        self.update_display()

    def _render_pdf_page(self, path):
        """Render the first page of a PDF to a PIL Image using pymupdf."""
        import fitz  # pymupdf
        doc = fitz.open(path)
        if doc.page_count == 0:
            return None

        # If multi-page, ask which page
        page_num = 0
        if doc.page_count > 1:
            page_num = self._ask_page_number(doc.page_count)

        page = doc[page_num]
        # Render at 3× scale so the background is sharp when stretched to canvas
        mat = fitz.Matrix(3, 3)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        doc.close()
        return img

    def _ask_page_number(self, total):
        """Simple dialog to pick a page from a multi-page PDF."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select page")
        dialog.wait_visibility()
        dialog.grab_set()
        var = tk.IntVar(value=1)
        tk.Label(dialog, text=f"Page (1–{total}):").pack(padx=10, pady=(10, 2))
        tk.Spinbox(dialog, from_=1, to=total, textvariable=var,
                   width=6).pack(padx=10)
        tk.Button(dialog, text="OK",
                  command=dialog.destroy).pack(pady=8)
        dialog.wait_window()
        return max(0, min(var.get() - 1, total - 1))

    # ── spline + polygon ──────────────────────────────────────────────────────

    def generate_spline(self, include_preview=True):
        """Return (x, y, t_new, t_pts, is_closed).
        t_pts are the parameter values at each actual waypoint."""
        points = self.points.copy()
        if include_preview and self.preview_point and not self.closed:
            points.append(self.preview_point)

        if len(points) < 2:
            return None, None, None, None, False

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        is_closed = self.closed or (
            len(points) >= 3 and np.allclose(points[-1], points[0]))

        try:
            t_pts = np.linspace(0, 1, len(points))
            ax = Akima1DInterpolator(t_pts, x)
            ay = Akima1DInterpolator(t_pts, y)
            t_new = np.linspace(0, 1, 200)
            return ax(t_new), ay(t_new), t_new, t_pts, is_closed
        except Exception as e:
            print("Spline error:", e)
            return None, None, None, None, False

    def _interp_widths(self, t_new, t_pts, include_preview=False):
        """Interpolate per-waypoint widths to 200 spline samples."""
        ws = self.point_widths.copy()
        if include_preview and self.preview_point and not self.closed:
            ws.append((self.left_width.get(), self.right_width.get()))
        if not ws:
            lw = self.left_width.get()
            rw = self.right_width.get()
            return np.full(len(t_new), lw), np.full(len(t_new), rw)
        left_ws  = np.array([w[0] for w in ws])
        right_ws = np.array([w[1] for w in ws])
        return np.interp(t_new, t_pts, left_ws), np.interp(t_new, t_pts, right_ws)

    def generate_polygon(self, x, y, left_widths, right_widths):
        """Offset a spline by per-point left/right widths to form a closed polygon."""
        if x is None or len(x) < 2:
            return np.array([])
        dx = np.gradient(x)
        dy = np.gradient(y)
        norm = np.hypot(dx, dy)
        norm[norm == 0] = 1e-9
        dx /= norm
        dy /= norm

        left_x  = x - dy * left_widths
        left_y  = y + dx * left_widths
        right_x = x + dy * right_widths
        right_y = y - dx * right_widths

        return np.concatenate([np.column_stack([left_x, left_y]),
                               np.column_stack([right_x[::-1], right_y[::-1]])])

    # ── display ───────────────────────────────────────────────────────────────

    def update_display(self):
        self.canvas.delete("all")
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        res = 1.0 / self.resolution.get()
        grid_w = int(self.real_width.get()  * res)
        grid_h = int(self.real_height.get() * res)
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

        x_sp, y_sp, t_new, t_pts, is_closed = self.generate_spline(include_preview=True)

        if x_sp is not None:
            interp_preview = self.preview_point is not None and not self.closed
            lw_arr, rw_arr = self._interp_widths(t_new, t_pts,
                                                  include_preview=interp_preview)
            bw = self.boundary_width.get()

            outer = self.generate_polygon(x_sp, y_sp, lw_arr, rw_arr)
            if outer.size > 0:
                cv2.fillPoly(grid, [(outer * res).astype(np.int32)], color=1)

            inner_lw = np.maximum(lw_arr - bw, 0)
            inner_rw = np.maximum(rw_arr - bw, 0)
            inner = self.generate_polygon(x_sp, y_sp, inner_lw, inner_rw)
            if inner.size > 0:
                cv2.fillPoly(grid, [(inner * res).astype(np.int32)], color=0)

        # obstacles drawn last so they punch into whatever they overlap
        for ox, oy, r in self.obstacles:
            cx_g = int(round(ox * res))
            cy_g = int(round(oy * res))
            r_g  = max(1, int(round(r * res)))
            cv2.circle(grid, (cx_g, cy_g), r_g, color=1, thickness=-1)

        # render track grid as RGBA so we can composite over background
        track_rgba = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
        wall_mask = grid == 1
        track_rgba[wall_mask]  = [0, 0, 0, 255]   # black walls, fully opaque
        track_rgba[~wall_mask] = [255, 255, 255, 0]  # free space, transparent

        track_img = Image.fromarray(track_rgba, 'RGBA').resize((cw, ch), Image.Resampling.NEAREST)

        # compose: background (if any) → white canvas → track overlay
        canvas_img = Image.new('RGBA', (cw, ch), (255, 255, 255, 255))
        if self.background_image is not None:
            rot = self.bg_rotation.get()
            bg_src = self.background_image.rotate(-rot, expand=True) if rot else self.background_image
            bg = bg_src.resize((cw, ch), Image.Resampling.LANCZOS).convert('RGBA')
            bg_faded = Image.new('RGBA', (cw, ch), (255, 255, 255, 255))
            bg_faded = Image.blend(bg_faded, bg, self.bg_opacity.get() / 100.0)
            canvas_img = Image.alpha_composite(canvas_img, bg_faded)
        canvas_img = Image.alpha_composite(canvas_img, track_img)

        self.photo = ImageTk.PhotoImage(canvas_img.convert('RGB'))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self._draw_waypoints_and_preview()
        self._draw_obstacles_on_canvas()
        self._draw_scale_bar()
        self._draw_calib_points()
        self._redraw_crosshair_on_top()

    def _draw_waypoints_and_preview(self):
        unique_pts = self.points[:-1] if self.closed else self.points
        for i, (rx, ry) in enumerate(unique_pts):
            cx, cy = self.real_to_canvas(rx, ry)
            if i == self.selected_point_idx and self.mode.get() == 'SELECT':
                fill = 'yellow'
            elif i == 0 and len(self.points) > 2:
                fill = 'green'
            else:
                fill = 'red'
            self.canvas.create_oval(cx-6, cy-6, cx+6, cy+6,
                                    fill=fill, outline='black')
            self.canvas.create_text(cx+9, cy-9, text=str(i),
                                    font=('', 7), fill='black')

        if self.preview_point and not self.closed:
            px, py = self.real_to_canvas(*self.preview_point)
            self.canvas.create_oval(px-5, py-5, px+5, py+5,
                                    fill='blue', outline='')
            if self.points:
                lx, ly = self.real_to_canvas(*self.points[-1])
                self.canvas.create_line(lx, ly, px, py, fill='blue', dash=(3, 3))

    def _draw_obstacles_on_canvas(self):
        for ox, oy, r in self.obstacles:
            cx, cy = self.real_to_canvas(ox, oy)
            # scale radius to canvas pixels
            r_px = r / self.real_width.get() * self.canvas.winfo_width()
            self.canvas.create_oval(cx - r_px, cy - r_px,
                                    cx + r_px, cy + r_px,
                                    fill='gray40', outline='black', stipple='gray50')

    def _draw_scale_bar(self):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        rw = self.real_width.get()
        rh = self.real_height.get()

        def nice_step(max_m):
            for step in [50, 20, 10, 5, 2, 1, 0.5]:
                if step <= max_m:
                    return step
            return max_m

        # ── horizontal bar (bottom-left) ──────────────────────────────────────
        bar_m = nice_step(rw * 0.15)
        bar_px = (bar_m / rw) * cw
        x0, y0 = 16, ch - 24
        x1 = x0 + bar_px
        self.canvas.create_line(x0, y0, x1, y0, fill='black', width=3)
        self.canvas.create_line(x0, y0 - 5, x0, y0 + 5, fill='black', width=2)
        self.canvas.create_line(x1, y0 - 5, x1, y0 + 5, fill='black', width=2)
        label = f"{bar_m:g} m  /  {bar_m * 3.28084:.0f} ft"
        self.canvas.create_text(x0 + bar_px / 2, y0 - 10,
                                text=label, font=('', 8, 'bold'), fill='black')

        # ── vertical bar (top-left) ───────────────────────────────────────────
        bar_m_y = nice_step(rh * 0.15)
        bar_px_y = (bar_m_y / rh) * ch
        xv, yv0 = 16, 16
        yv1 = yv0 + bar_px_y
        self.canvas.create_line(xv, yv0, xv, yv1, fill='black', width=3)
        self.canvas.create_line(xv - 5, yv0, xv + 5, yv0, fill='black', width=2)
        self.canvas.create_line(xv - 5, yv1, xv + 5, yv1, fill='black', width=2)
        label_y = f"{bar_m_y:g} m  /  {bar_m_y * 3.28084:.0f} ft"
        self.canvas.create_text(xv + 10, yv0 + bar_px_y / 2,
                                text=label_y, font=('', 8, 'bold'),
                                fill='black', angle=0, anchor='w')

    def _draw_calib_points(self):
        for i, (cx, cy) in enumerate(self.calib_points):
            color = '#e67e00'
            self.canvas.create_oval(cx - 6, cy - 6, cx + 6, cy + 6,
                                    fill=color, outline='black', width=2)
            self.canvas.create_text(cx + 10, cy, text=f"P{i+1}",
                                    font=('', 8, 'bold'), fill=color)
        if len(self.calib_points) == 2:
            (x1, y1), (x2, y2) = self.calib_points
            self.canvas.create_line(x1, y1, x2, y2,
                                    fill='#e67e00', width=2, dash=(4, 3))

    def _redraw_crosshair_on_top(self):
        # crosshair items stored in self.cursor_crosshair are deleted in on_mouse_move;
        # they will be re-added there. Nothing to do here.
        pass

    # ── crosshair ─────────────────────────────────────────────────────────────

    def _draw_crosshair(self, cx, cy, rx, ry):
        for item in self.cursor_crosshair:
            self.canvas.delete(item)
        self.cursor_crosshair = []

        # check proximity to obstacle
        near_obstacle = any(np.hypot(rx - ox, ry - oy) < OBSTACLE_SNAP_RADIUS
                            for ox, oy, _ in self.obstacles)
        colour = 'orange' if near_obstacle else 'gray50'

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        self.cursor_crosshair.append(
            self.canvas.create_line(0, cy, cw, cy, fill=colour, dash=(2, 2)))
        self.cursor_crosshair.append(
            self.canvas.create_line(cx, 0, cx, ch, fill=colour, dash=(2, 2)))
        size = 5
        self.cursor_crosshair.append(
            self.canvas.create_rectangle(cx-size, cy-size, cx+size, cy+size,
                                         outline='black', fill='blue'))

    # ── status bar ────────────────────────────────────────────────────────────

    def _update_status(self, rx, ry):
        mode_label = {'TRACK': 'Draw Track', 'OBSTACLE': 'Place Obstacle',
                      'SELECT': 'Select/Edit', 'CALIBRATE': 'Calibrate'}[self.mode.get()]
        if rx is None:
            self.status_var.set(f"Mode: {mode_label}")
            return
        scroll_hint = "Scroll = width"
        if self.selected_point_idx is not None and self.mode.get() == 'SELECT':
            idx = self.selected_point_idx
            lw, rw = self.point_widths[idx]
            self.status_var.set(
                f"X: {rx:.2f}m  Y: {ry:.2f}m  |  Mode: {mode_label}  |  "
                f"Pt {idx}: L={lw:.2f}m R={rw:.2f}m  |  {scroll_hint}")
        else:
            self.status_var.set(
                f"X: {rx:.2f}m  Y: {ry:.2f}m  |  Mode: {mode_label}  |  "
                f"Default width: L={self.left_width.get():.2f}m "
                f"R={self.right_width.get():.2f}m  |  {scroll_hint}")

    # ── save ──────────────────────────────────────────────────────────────────

    def save_grid(self):
        res = 1.0 / self.resolution.get()
        grid_w = int(self.real_width.get()  * res)
        grid_h = int(self.real_height.get() * res)
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

        x_sp, y_sp, t_new, t_pts, _ = self.generate_spline(include_preview=False)
        if x_sp is not None:
            lw_arr, rw_arr = self._interp_widths(t_new, t_pts, include_preview=False)
            bw = self.boundary_width.get()

            outer = self.generate_polygon(x_sp, y_sp, lw_arr, rw_arr)
            if outer.size > 0:
                cv2.fillPoly(grid, [(outer * res).astype(np.int32)], color=1)

            inner_lw = np.maximum(lw_arr - bw, 0)
            inner_rw = np.maximum(rw_arr - bw, 0)
            inner = self.generate_polygon(x_sp, y_sp, inner_lw, inner_rw)
            if inner.size > 0:
                cv2.fillPoly(grid, [(inner * res).astype(np.int32)], color=0)

        for ox, oy, r in self.obstacles:
            cv2.circle(grid, (int(round(ox * res)), int(round(oy * res))),
                       max(1, int(round(r * res))), color=1, thickness=-1)

        img = Image.fromarray((1 - grid) * 255)
        ts  = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        png_file  = f"{ts}.png"
        yaml_file = f"{ts}.yaml"
        img.save(png_file)
        with open(yaml_file, "w") as f:
            f.write(f'image: "{png_file}"\n'
                    f'resolution: {self.resolution.get()}\n'
                    f'origin: [0.0, 0.0, 0.000000]\n'
                    f'negate: 0\n'
                    f'occupied_thresh: 0.45\n'
                    f'free_thresh: 0.196\n')
        print(f"Saved {png_file} and {yaml_file}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RacetrackDesigner(root)
    root.mainloop()