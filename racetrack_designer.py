import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from scipy.interpolate import Akima1DInterpolator
from PIL import Image, ImageTk
import cv2
import datetime
import json
import os
import pathlib

SNAP_THRESHOLD = 1.0  # metres
POINT_SELECT_RADIUS = 0.8  # metres — click within this to select a waypoint
OBSTACLE_SNAP_RADIUS = 0.6  # metres — cursor turns orange when near obstacle

DRAG_THRESHOLD_PX = 6      # shorter than this and a press+release is still a click
SWEEP_MIN_SAMPLES = 4      # a sweep needs at least this many cursor samples
WIDTH_HANDLE_RADIUS_PX = 8  # grab radius for the on-canvas width handles
MIN_WIDTH = 0.05           # metres — floor for any track half-width
UNDO_LIMIT = 60            # how many edits back Ctrl+Z can reach

PROJECT_FORMAT = 1
AUTOSAVE_DIR = pathlib.Path.home() / ".track_designer"
AUTOSAVE_FILE = AUTOSAVE_DIR / "autosave.json"
AUTOSAVE_INTERVAL_MS = 3000  # how often the in-memory design is checked for changes


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

        # control panel — scrollable, it is taller than a laptop screen
        panel = tk.Frame(main_frame)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        self._panel_canvas = tk.Canvas(panel, highlightthickness=0, width=215)
        vbar = tk.Scrollbar(panel, orient=tk.VERTICAL,
                            command=self._panel_canvas.yview)
        self._panel_canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._panel_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        ctrl = tk.Frame(self._panel_canvas, padx=10, pady=10)
        self._panel_canvas.create_window((0, 0), window=ctrl, anchor='nw')
        ctrl.bind("<Configure>", lambda e: self._panel_canvas.configure(
            scrollregion=self._panel_canvas.bbox("all"),
            width=ctrl.winfo_reqwidth()))
        self._build_control_panel(ctrl)
        self._bind_panel_scroll(panel)

        # ── state ─────────────────────────────────────────────────────────────
        self.points = []           # [(real_x, real_y), ...]
        self.point_widths = []     # [(left_w, right_w), ...] parallel to points
        self.obstacles = []        # [(real_x, real_y, radius), ...]
        self.closed = False
        self.grid_image = None
        self.cursor_crosshair = []
        self.photo = None              # composed canvas image (kept alive for Tk)
        self._photo_sig = None         # what self.photo was built from
        self._bg_layer = None          # scaled/rotated/faded floor plan
        self._bg_sig = None
        self.preview_point = None
        self.selected_point_idx = None
        self.background_image = None   # PIL Image
        self.background_photo = None   # ImageTk (kept alive)
        self.background_path = None    # source file, so autosave can reload it
        self.background_page = 0       # page index if the source was a PDF
        self.calib_points = []         # canvas (x,y) clicks during CALIBRATE mode

        # ── drag state ────────────────────────────────────────────────────────
        self._press_xy = None          # where the current press started (canvas px)
        self._press_moved = False      # has it moved far enough to be a drag?
        self._sweep_canvas = []        # live cursor trail while sweeping (canvas px)
        self._sweep_real = []          # the same samples in metres
        self._sweep_widths = []        # (left, right) width painted at each sample
        self._sweep_source_widths = [] # widths of the finished sweep, kept for re-simplifying
        self._sweep_source = None      # raw sweep in real coords, kept for re-simplifying
        self._sweep_base = []          # waypoints the current sweep continues from
        self._sweep_base_widths = []
        self._sweep_owns_points = False  # True while self.points came from that sweep
        self._dragging_point = False
        self._drag_handle = None       # 'left' | 'right' while a width handle is held
        self._undo_stack = []
        self._redo_stack = []

        # ── bindings ──────────────────────────────────────────────────────────
        self.canvas.bind("<ButtonPress-1>",   self.on_left_press)
        self.canvas.bind("<B1-Motion>",       self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>",   self.on_right_click)
        self.canvas.bind("<Motion>",     self.on_mouse_move)
        self.canvas.bind("<Configure>",  lambda e: self.update_display())
        # Linux scroll
        self.canvas.bind("<Button-4>",   lambda e: self.on_scroll(e, +1))
        self.canvas.bind("<Button-5>",   lambda e: self.on_scroll(e, -1))
        # Windows/macOS scroll
        self.canvas.bind("<MouseWheel>", self.on_scroll_wheel)
        # arrows do what the wheel does: width of the selected point, the
        # default width, or the width being painted mid-sweep
        self.root.bind("<Up>",   lambda e: self._on_arrow(+0.05, e))
        self.root.bind("<Down>", lambda e: self._on_arrow(-0.05, e))
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Control-Shift-Z>", self.redo)

        for var in [self.resolution, self.real_width, self.real_height,
                    self.left_width, self.right_width, self.boundary_width,
                    self.obstacle_radius]:
            var.trace_add('write', lambda *_: self.update_display())

        # ── autosave ──────────────────────────────────────────────────────────
        self._last_saved_state = None
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after_idle(self._offer_restore)
        self.root.after(AUTOSAVE_INTERVAL_MS, self._autosave_tick)

    # ── UI builders ───────────────────────────────────────────────────────────

    def _bind_panel_scroll(self, widget):
        """Wheel anywhere over the control panel scrolls it.

        Tk delivers wheel events to the widget under the cursor and does not
        bubble them, so every child needs the binding."""
        def scroll(delta):
            self._panel_canvas.yview_scroll(delta, 'units')
            return "break"
        widget.bind("<Button-4>",   lambda e: scroll(-2))
        widget.bind("<Button-5>",   lambda e: scroll(2))
        widget.bind("<MouseWheel>", lambda e: scroll(-2 if e.delta > 0 else 2))
        for child in widget.winfo_children():
            self._bind_panel_scroll(child)

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

        def slider(label, var, lo, hi, step, command=None):
            nonlocal row
            tk.Label(ctrl, text=label).grid(row=row, column=0, sticky='w')
            s = tk.Scale(ctrl, from_=lo, to=hi, resolution=step,
                         orient=tk.HORIZONTAL, variable=var, length=110,
                         command=command)
            s.grid(row=row, column=1, sticky='ew')
            row += 1
            return s

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
        slider("Left Width (m):",  self.left_width,  MIN_WIDTH, 5.0, 0.05)
        slider("Right Width (m):", self.right_width, MIN_WIDTH, 5.0, 0.05)
        entry("Boundary Width (m):", self.boundary_width)

        section("Freehand Draw")
        tk.Label(ctrl, text="Hold and sweep to draw. Each sweep continues "
                            "the track; release near point 0 to close.",
                 wraplength=190, justify='left', fg='gray40',
                 font=('', 8)).grid(row=row, column=0, columnspan=2, sticky='w')
        row += 1
        self.detail = tk.IntVar(value=70)
        slider("Detail (coarse → fine):", self.detail, 1, 100, 1,
               command=lambda _: self._on_detail_changed())

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
        self.lock_widths   = tk.BooleanVar(value=True)
        self.sel_left_entry  = slider("Left Width (m):",  self.sel_left_var,
                                      MIN_WIDTH, 5.0, 0.05)
        self.sel_right_entry = slider("Right Width (m):", self.sel_right_var,
                                      MIN_WIDTH, 5.0, 0.05)
        tk.Checkbutton(ctrl, text="Lock L = R", variable=self.lock_widths).grid(
            row=row, column=0, columnspan=2, sticky='w')
        row += 1
        self.sel_left_var.trace_add('write',  lambda *_: self._sel_width_changed('left'))
        self.sel_right_var.trace_add('write', lambda *_: self._sel_width_changed('right'))

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
        tk.Button(ctrl, text="Open Project…",
                  command=self.open_project).grid(row=row, column=0,
                                                  columnspan=2, sticky='ew')
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

    def on_left_press(self, event):
        """A press is ambiguous — it becomes a click or a drag on release."""
        self._press_xy = (event.x, event.y)
        self._press_moved = False
        self._dragging_point = False
        self._drag_handle = None
        m = self.mode.get()

        if m == 'TRACK':
            self._sweep_canvas = [(event.x, event.y)]
            self._sweep_real = [self.canvas_to_real(event.x, event.y)]
            self._sweep_widths = [self._current_default_width()]
        elif m == 'SELECT':
            # a width handle sits on top of the waypoint it belongs to
            if self._grab_width_handle(event):
                self._push_undo()
                return
            self._select_click(event)
            if self.selected_point_idx is not None:
                self._dragging_point = True
                self._push_undo()

    def on_left_drag(self, event):
        if self._press_xy is None:
            return
        if not self._press_moved:
            dx = event.x - self._press_xy[0]
            dy = event.y - self._press_xy[1]
            if np.hypot(dx, dy) < DRAG_THRESHOLD_PX:
                return          # still within click tolerance
            self._press_moved = True

        if self.mode.get() == 'TRACK':
            self._sweep_canvas.append((event.x, event.y))
            self._sweep_real.append(self.canvas_to_real(event.x, event.y))
            self._sweep_widths.append(self._current_default_width())
            self._draw_sweep_trail()        # cheap: no grid re-render mid-sweep
        elif self._drag_handle:
            self._drag_width_handle(event)
        elif self._dragging_point:
            self._move_selected_point(*self.canvas_to_real(event.x, event.y))
            self.update_display()

    def on_left_release(self, event):
        if self._press_xy is None:
            return
        was_drag = self._press_moved
        self._press_xy = None
        self._press_moved = False

        if not was_drag:
            # a plain click — behave exactly as before
            self._sweep_canvas = []
            self._sweep_real = []
            self._sweep_widths = []
            self.canvas.delete('sweep')
            self._dragging_point = False
            self._drag_handle = None
            m = self.mode.get()
            if m == 'TRACK':
                self._track_click(event)
            elif m == 'OBSTACLE':
                self._obstacle_click(event)
            elif m == 'CALIBRATE':
                self._calib_click(event)
            # SELECT already ran its selection on press
            return

        if self.mode.get() == 'TRACK' and self._sweep_canvas:
            self._finish_sweep()
        self._sweep_canvas = []
        self._sweep_real = []
        self._sweep_widths = []
        self._dragging_point = False
        self._drag_handle = None
        self.update_display()

    def on_left_click(self, event):
        """Kept for callers that synthesise a plain click."""
        self.on_left_press(event)
        self.on_left_release(event)

    def on_right_click(self, event):
        if self.mode.get() == 'OBSTACLE':
            rx, ry = self.canvas_to_real(event.x, event.y)
            # remove nearest obstacle within snap radius
            for i, (ox, oy, _r) in enumerate(self.obstacles):
                if np.hypot(rx - ox, ry - oy) < OBSTACLE_SNAP_RADIUS:
                    self._push_undo()
                    self.obstacles.pop(i)
                    self.update_display()
                    return

    def on_mouse_move(self, event):
        rx, ry = self.canvas_to_real(event.x, event.y)
        self._update_status(rx, ry)
        if self.mode.get() == 'TRACK' and not self.closed:
            self._update_track_preview(rx, ry)
        else:
            self.preview_point = None
            self.update_display()
        self._draw_crosshair(event.x, event.y, rx, ry)   # must come last

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

    def _on_arrow(self, step, event):
        """Arrows resize like the wheel — but not while typing in a field."""
        # key events are delivered to the focused widget, so that is the one
        # to check; focus_get() is only a fallback
        target = getattr(event, 'widget', None)
        if target is None:
            try:
                target = self.root.focus_get()
            except KeyError:
                target = None
        if isinstance(target, (tk.Entry, tk.Spinbox)):
            return
        shift = (event.state & 0x1) != 0
        ctrl  = (event.state & 0x4) != 0
        self._apply_scroll_step(step, shift, ctrl)
        return "break"

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
        self._push_undo()
        self._sweep_owns_points = False
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

    # ── freehand sweep ────────────────────────────────────────────────────────

    def _current_default_width(self):
        try:
            return (max(MIN_WIDTH, float(self.left_width.get())),
                    max(MIN_WIDTH, float(self.right_width.get())))
        except (tk.TclError, ValueError):
            return (0.6, 0.6)

    @staticmethod
    def _width_colour(width):
        """Narrow → blue, wide → red, so a width change reads at a glance."""
        stops = [(0.2, (0x21, 0x5f, 0xd0)), (0.8, (0x16, 0xa8, 0xa8)),
                 (1.6, (0xe2, 0x8b, 0x1f)), (3.0, (0xd0, 0x2b, 0x2b))]
        w = float(np.clip(width, stops[0][0], stops[-1][0]))
        for (w0, c0), (w1, c1) in zip(stops, stops[1:]):
            if w <= w1:
                f = 0.0 if w1 == w0 else (w - w0) / (w1 - w0)
                r, g, b = (int(round(a + (bb - a) * f)) for a, bb in zip(c0, c1))
                return f"#{r:02x}{g:02x}{b:02x}"
        return "#d02b2b"

    def _band_edges(self, real_pts, widths):
        """Left and right track edges in canvas pixels for a polyline."""
        pts = np.asarray(real_pts, dtype=float)
        w = np.asarray(widths, dtype=float)
        if len(pts) < 2:
            return None, None
        dx = np.gradient(pts[:, 0])
        dy = np.gradient(pts[:, 1])
        norm = np.hypot(dx, dy)
        norm[norm == 0] = 1e-9
        dx, dy = dx / norm, dy / norm
        left = np.column_stack([pts[:, 0] - dy * w[:, 0], pts[:, 1] + dx * w[:, 0]])
        right = np.column_stack([pts[:, 0] + dy * w[:, 1], pts[:, 1] - dx * w[:, 1]])
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        sx = cw / self.real_width.get()
        sy = ch / self.real_height.get()
        return left * [sx, sy], right * [sx, sy]

    def _draw_sweep_trail(self):
        """Draw the band being painted: true width, colour-ramped by width.

        Deliberately not update_display(): re-rasterising the track on every
        motion event is what made sweeping slow."""
        self.canvas.delete('sweep')
        if len(self._sweep_real) < 2:
            return

        real = list(self._sweep_real)
        widths = list(self._sweep_widths)
        # an open track is extended, so start the band at its last waypoint
        if self.points and not self.closed:
            real.insert(0, self.points[-1])
            widths.insert(0, self.point_widths[-1] if self.point_widths else widths[0])

        left, right = self._band_edges(real, widths)
        if left is None:
            return
        ring = np.concatenate([left, right[::-1]])
        self.canvas.create_polygon(*[c for xy in ring for c in xy],
                                   fill='#1560c0', stipple='gray25',
                                   outline='#1560c0', width=1, tags='sweep')

        # centerline in runs of one colour, so a width change is obvious
        centre = [self.real_to_canvas(*p) for p in real]
        colours = [self._width_colour((l + r) / 2) for l, r in widths]
        start = 0
        for i in range(1, len(centre) + 1):
            if i == len(centre) or colours[i] != colours[start]:
                if i - start >= 2:
                    seg = centre[start:i]
                    self.canvas.create_line(*[c for xy in seg for c in xy],
                                            fill=colours[start], width=3,
                                            capstyle=tk.ROUND, joinstyle=tk.ROUND,
                                            tags='sweep')
                start = i - 1
        # ring the waypoint to release on to close the loop
        target = self.points[0] if (self.points and not self.closed) else None
        sx, sy = (self.real_to_canvas(*target) if target
                  else self._sweep_canvas[0])
        self.canvas.create_oval(sx - 7, sy - 7, sx + 7, sy + 7,
                                outline='green', width=2, tags='sweep')

    def _finish_sweep(self):
        """Turn the raw cursor trail into waypoints, continuing the track."""
        # drop consecutive duplicates — they break the tangent maths
        pts, widths = [self._sweep_real[0]], [self._sweep_widths[0]]
        for p, w in zip(self._sweep_real[1:], self._sweep_widths[1:]):
            if np.hypot(p[0] - pts[-1][0], p[1] - pts[-1][1]) > 1e-6:
                pts.append(p)
                widths.append(w)
        if len(pts) < SWEEP_MIN_SAMPLES:
            return
        self._sweep_source_widths = widths

        self._push_undo()
        # an unfinished track is extended; a finished loop is started over
        if self.points and not self.closed:
            self._sweep_base = list(self.points)
            self._sweep_base_widths = list(self.point_widths)
        else:
            self._sweep_base = []
            self._sweep_base_widths = []
        self._sweep_source = pts
        self._rebuild_from_sweep()

    def _rebuild_from_sweep(self):
        """(Re)simplify the stored sweep at the current Detail setting."""
        if not self._sweep_source:
            return
        keep = self._rdp_indices(self._sweep_source, self._detail_epsilon())
        if len(keep) < 2:
            keep = [0, len(self._sweep_source) - 1]
        pts = [self._sweep_source[i] for i in keep]
        # the width painted at each surviving sample rides along with it
        sweep_widths = [self._sweep_source_widths[i] for i in keep]

        base = list(self._sweep_base)
        base_widths = list(self._sweep_base_widths)
        # a sweep that starts on the last waypoint must not duplicate it
        if base and np.hypot(pts[0][0] - base[-1][0],
                             pts[0][1] - base[-1][1]) < 1e-6:
            pts, sweep_widths = pts[1:], sweep_widths[1:]

        points = base + [tuple(p) for p in pts]
        widths = base_widths + [tuple(w) for w in sweep_widths]

        # ending near the track's first waypoint closes the loop
        close = (len(points) >= 3 and
                 np.hypot(points[-1][0] - points[0][0],
                          points[-1][1] - points[0][1]) < SNAP_THRESHOLD)
        if close:
            points[-1] = points[0]
            widths[-1] = widths[0]

        self.points = points
        self.point_widths = widths
        self.closed = close
        self.selected_point_idx = None
        self.preview_point = None
        self._sweep_owns_points = True

        self.canvas.delete('sweep')
        self.save_button.config(state=tk.NORMAL if close else tk.DISABLED)
        self.delete_button.config(state=tk.NORMAL if self.points else tk.DISABLED)
        self.update_display()

    def _detail_epsilon(self):
        """Slider (1 = coarse … 100 = fine) → RDP tolerance in metres."""
        detail = max(1, min(100, self.detail.get()))
        return max(0.02, (101 - detail) / 100.0)

    @staticmethod
    def _rdp_indices(points, epsilon):
        """Ramer-Douglas-Peucker: indices of the points that carry the shape.

        Indices rather than points so per-sample data (the painted width) can
        be carried along. Iterative, so a long sweep cannot blow the recursion
        limit."""
        pts = np.asarray(points, dtype=float)
        n = len(pts)
        if n < 3:
            return list(range(n))

        keep = np.zeros(n, dtype=bool)
        keep[0] = keep[-1] = True
        stack = [(0, n - 1)]
        while stack:
            start, end = stack.pop()
            if end <= start + 1:
                continue
            a, b = pts[start], pts[end]
            seg = b - a
            seg_len = np.hypot(*seg)
            chunk = pts[start + 1:end]
            if seg_len < 1e-12:
                dist = np.hypot(chunk[:, 0] - a[0], chunk[:, 1] - a[1])
            else:
                # perpendicular distance from each point to the chord a→b
                dist = np.abs(np.cross(seg, chunk - a)) / seg_len
            i = int(np.argmax(dist))
            if dist[i] > epsilon:
                split = start + 1 + i
                keep[split] = True
                stack.append((start, split))
                stack.append((split, end))
        return np.flatnonzero(keep).tolist()

    def delete_last_point(self):
        if not self.points:
            return
        self._push_undo()
        self._sweep_owns_points = False
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
        self._push_undo()
        self._sweep_owns_points = False
        self._sweep_source = None
        self._sweep_base = []
        self._sweep_base_widths = []
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
        self._push_undo()
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

    def _sel_width_changed(self, side=None, *_):
        if getattr(self, '_sel_syncing', False):
            return
        if self.selected_point_idx is None:
            return
        try:
            lw = float(self.sel_left_var.get())
            rw = float(self.sel_right_var.get())
        except (ValueError, tk.TclError):
            return

        if self.lock_widths.get() and side is not None:
            # mirror the side the user just moved onto the other one
            self._sel_syncing = True
            if side == 'left':
                rw = lw
                self.sel_right_var.set(round(rw, 4))
            else:
                lw = rw
                self.sel_left_var.set(round(lw, 4))
            self._sel_syncing = False

        idx = self.selected_point_idx
        self.point_widths[idx] = (lw, rw)
        # if closed loop, keep first==last width in sync
        if self.closed and idx == 0:
            self.point_widths[-1] = (lw, rw)
        elif self.closed and idx == len(self.point_widths) - 1:
            self.point_widths[0] = (lw, rw)
        self.update_display()

    def _on_detail_changed(self):
        """Re-simplify the last sweep — unless the user has since hand-edited."""
        if getattr(self, '_sweep_owns_points', False) and self._sweep_source:
            self._rebuild_from_sweep()

    def _move_selected_point(self, rx, ry):
        """Drag the selected waypoint to a new place."""
        idx = self.selected_point_idx
        if idx is None or idx >= len(self.points):
            return
        rx = float(np.clip(rx, 0.0, self.real_width.get()))
        ry = float(np.clip(ry, 0.0, self.real_height.get()))
        self.points[idx] = (rx, ry)
        # a closed loop stores the first point twice — keep the copies together
        if self.closed and len(self.points) > 1:
            if idx == 0:
                self.points[-1] = (rx, ry)
            elif idx == len(self.points) - 1:
                self.points[0] = (rx, ry)
        self._sweep_owns_points = False

    # ── on-canvas width handles ───────────────────────────────────────────────

    def _point_normal(self, idx):
        """Unit normal at waypoint idx, matching generate_polygon's convention."""
        n = len(self.points)
        if n < 2:
            return 0.0, 1.0
        if self.closed and n > 2:
            prev_i, next_i = (idx - 1) % (n - 1), (idx + 1) % (n - 1)
        else:
            prev_i, next_i = max(0, idx - 1), min(n - 1, idx + 1)
        (px, py), (nx, ny) = self.points[prev_i], self.points[next_i]
        dx, dy = nx - px, ny - py
        norm = np.hypot(dx, dy)
        if norm < 1e-9:
            return 0.0, 1.0
        dx, dy = dx / norm, dy / norm
        return -dy, dx          # left-hand normal

    def _width_handle_points(self):
        """Canvas positions of the selected point's two width handles."""
        idx = self.selected_point_idx
        if idx is None or self.mode.get() != 'SELECT' or idx >= len(self.point_widths):
            return None
        lw, rw = self.point_widths[idx]
        px, py = self.points[idx]
        ux, uy = self._point_normal(idx)
        left  = self.real_to_canvas(px + ux * lw, py + uy * lw)
        right = self.real_to_canvas(px - ux * rw, py - uy * rw)
        return left, right

    def _grab_width_handle(self, event):
        handles = self._width_handle_points()
        if handles is None:
            return False
        for name, (hx, hy) in zip(('left', 'right'), handles):
            if np.hypot(event.x - hx, event.y - hy) <= WIDTH_HANDLE_RADIUS_PX:
                self._drag_handle = name
                return True
        return False

    def _drag_width_handle(self, event):
        """Width = how far the cursor sits along the track normal."""
        idx = self.selected_point_idx
        if idx is None:
            return
        rx, ry = self.canvas_to_real(event.x, event.y)
        px, py = self.points[idx]
        ux, uy = self._point_normal(idx)
        dist = abs((rx - px) * ux + (ry - py) * uy)
        width = round(max(MIN_WIDTH, dist), 3)

        lw, rw = self.point_widths[idx]
        if self.lock_widths.get():
            lw = rw = width
        elif self._drag_handle == 'left':
            lw = width
        else:
            rw = width
        self.point_widths[idx] = (lw, rw)
        if self.closed and len(self.point_widths) > 1:
            if idx == 0:
                self.point_widths[-1] = (lw, rw)
            elif idx == len(self.point_widths) - 1:
                self.point_widths[0] = (lw, rw)
        self._sync_sel_vars_from_point(idx)
        self.update_display()

    def _apply_width_to_all(self):
        if self.selected_point_idx is None:
            return
        self._push_undo()
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
        self._set_background(path)
        self.update_display()

    def _set_background(self, path, page=None):
        """Load a floor plan and remember where it came from (for autosave)."""
        if path.lower().endswith('.pdf'):
            self.background_image = self._render_pdf_page(path, page)
        else:
            self.background_image = Image.open(path).convert('RGB')
            self.background_page = 0
        self.background_path = path

    def _rotate_bg(self, degrees):
        self.bg_rotation.set((self.bg_rotation.get() + degrees) % 360)
        self.update_display()

    def _render_pdf_page(self, path, page=None):
        """Render a page of a PDF to a PIL Image using pymupdf.

        `page` is a 0-based index; when given (e.g. restoring a session) the
        page prompt is skipped."""
        import fitz  # pymupdf
        doc = fitz.open(path)
        if doc.page_count == 0:
            return None

        # If multi-page, ask which page
        page_num = 0
        if page is not None:
            page_num = max(0, min(page, doc.page_count - 1))
        elif doc.page_count > 1:
            page_num = self._ask_page_number(doc.page_count)
        self.background_page = page_num

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
        self.cursor_crosshair = []
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        photo = self._compose_photo(cw, ch)
        if photo is not None:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        self._draw_waypoints_and_preview()
        self._draw_width_handles()
        self._draw_obstacles_on_canvas()
        self._draw_scale_bar()
        self._draw_calib_points()
        if self._sweep_canvas:
            self._draw_sweep_trail()    # a redraw mid-sweep must keep the band

    # The picture is expensive; everything above it is cheap canvas items. So
    # rebuild it only when something it depends on actually changed.

    def _track_signature(self, cw, ch):
        return (cw, ch, tuple(self.points), tuple(self.point_widths),
                tuple(self.obstacles), self.closed,
                round(self.boundary_width.get(), 5),
                round(self.real_width.get(), 5),
                round(self.real_height.get(), 5))

    def _bg_signature(self, cw, ch):
        return (cw, ch, id(self.background_image),
                self.bg_rotation.get(), self.bg_opacity.get())

    def _compose_photo(self, cw, ch):
        try:
            sig = (self._track_signature(cw, ch), self._bg_signature(cw, ch))
        except tk.TclError:
            return self.photo          # a field is mid-edit; keep the last frame
        if sig == self._photo_sig and self.photo is not None:
            return self.photo

        img = self._background_layer(cw, ch).copy()
        walls = self._track_layer(cw, ch)
        if walls is not None:
            img.paste((0, 0, 0), (0, 0), walls)   # black through the wall mask
        self.photo = ImageTk.PhotoImage(img)
        self._photo_sig = sig
        return self.photo

    def _background_layer(self, cw, ch):
        """Rotating and resampling the floor plan is slow — do it once."""
        sig = self._bg_signature(cw, ch)
        if sig == self._bg_sig and self._bg_layer is not None:
            return self._bg_layer
        layer = Image.new('RGB', (cw, ch), (255, 255, 255))
        if self.background_image is not None:
            rot = self.bg_rotation.get()
            src = (self.background_image.rotate(-rot, expand=True) if rot
                   else self.background_image)
            bg = src.resize((cw, ch), Image.Resampling.BILINEAR).convert('RGB')
            layer = Image.blend(layer, bg, self.bg_opacity.get() / 100.0)
        self._bg_layer, self._bg_sig = layer, sig
        return layer

    def _track_layer(self, cw, ch):
        """Stencil of the walls, rasterised straight at screen size.

        The metric occupancy grid is only needed by save_grid(); building one
        per frame just to shrink it was most of the cost of a redraw."""
        sx = cw / self.real_width.get()
        sy = ch / self.real_height.get()
        mask = np.zeros((ch, cw), dtype=np.uint8)

        def to_px(poly):
            return np.column_stack([poly[:, 0] * sx, poly[:, 1] * sy]).astype(np.int32)

        x_sp, y_sp, t_new, t_pts, _ = self.generate_spline(include_preview=False)
        if x_sp is not None:
            lw_arr, rw_arr = self._interp_widths(t_new, t_pts, include_preview=False)
            bw = self.boundary_width.get()
            outer = self.generate_polygon(x_sp, y_sp, lw_arr, rw_arr)
            if outer.size > 0:
                cv2.fillPoly(mask, [to_px(outer)], color=1)
            inner = self.generate_polygon(x_sp, y_sp,
                                          np.maximum(lw_arr - bw, 0),
                                          np.maximum(rw_arr - bw, 0))
            if inner.size > 0:
                cv2.fillPoly(mask, [to_px(inner)], color=0)

        # obstacles drawn last so they punch into whatever they overlap
        for ox, oy, r in self.obstacles:
            cv2.ellipse(mask, (int(round(ox * sx)), int(round(oy * sy))),
                        (max(1, int(round(r * sx))), max(1, int(round(r * sy)))),
                        0, 0, 360, color=1, thickness=-1)

        if not mask.any():
            return None
        # a one-channel stencil, not a four-channel image to composite
        return Image.fromarray(mask * 255, 'L')

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
                # show the width the next leg would have, as an outline — drawing
                # it into the raster would mean re-rendering on every mouse move
                w_last = (self.point_widths[-1] if self.point_widths
                          else self._current_default_width())
                w_next = self._current_default_width()
                left, right = self._band_edges(
                    [self.points[-1], self.preview_point], [w_last, w_next])
                if left is not None:
                    for edge in (left, right):
                        self.canvas.create_line(*[c for xy in edge for c in xy],
                                                fill='#1560c0', dash=(3, 3))

    def _draw_width_handles(self):
        """Diamonds on the track edges of the selected point — drag to widen."""
        handles = self._width_handle_points()
        if handles is None:
            return
        idx = self.selected_point_idx
        cx, cy = self.real_to_canvas(*self.points[idx])
        s = WIDTH_HANDLE_RADIUS_PX
        for hx, hy in handles:
            self.canvas.create_line(cx, cy, hx, hy, fill='#0a7', width=1, dash=(2, 2))
            self.canvas.create_polygon(hx, hy - s, hx + s, hy, hx, hy + s, hx - s, hy,
                                       fill='#00c07a', outline='black')

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
        hints = {'TRACK': "Click points or hold and sweep the centerline",
                 'OBSTACLE': "Click to place, right-click to remove",
                 'SELECT': "Drag a point to move it, drag a green handle to widen",
                 'CALIBRATE': "Click 2 points of known distance"}
        scroll_hint = hints[self.mode.get()]
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

    # ── undo / redo ───────────────────────────────────────────────────────────

    def _push_undo(self):
        """Call before changing the design, never after."""
        try:
            snap = self._snapshot()
        except tk.TclError:
            return
        if self._undo_stack and self._undo_stack[-1] == snap:
            return
        self._undo_stack.append(snap)
        del self._undo_stack[:-UNDO_LIMIT]
        self._redo_stack.clear()

    @staticmethod
    def _is_typing(event):
        return isinstance(getattr(event, 'widget', None), (tk.Entry, tk.Spinbox))

    def undo(self, event=None):
        if self._is_typing(event):
            return
        if self._undo_stack:
            self._redo_stack.append(self._snapshot())
            self._restore_state(self._undo_stack.pop())
            self._last_saved_state = None    # the undone state deserves a save
        return "break"

    def redo(self, event=None):
        if self._is_typing(event):
            return
        if self._redo_stack:
            self._undo_stack.append(self._snapshot())
            self._restore_state(self._redo_stack.pop())
            self._last_saved_state = None
        return "break"

    # ── project state (autosave / restore) ────────────────────────────────────

    def _snapshot(self):
        """Everything needed to rebuild the current design, as plain JSON types."""
        return {
            "format": PROJECT_FORMAT,
            "points": [[float(x), float(y)] for x, y in self.points],
            "point_widths": [[float(l), float(r)] for l, r in self.point_widths],
            "obstacles": [[float(x), float(y), float(r)] for x, y, r in self.obstacles],
            "closed": bool(self.closed),
            "resolution": self.resolution.get(),
            "real_width": self.real_width.get(),
            "real_height": self.real_height.get(),
            "left_width": self.left_width.get(),
            "right_width": self.right_width.get(),
            "boundary_width": self.boundary_width.get(),
            "obstacle_radius": self.obstacle_radius.get(),
            "background_path": self.background_path,
            "background_page": self.background_page,
            "bg_opacity": self.bg_opacity.get(),
            "bg_rotation": self.bg_rotation.get(),
        }

    def _is_empty_state(self, state):
        return not state["points"] and not state["obstacles"]

    def _write_project(self, path, state):
        """Write atomically: a crash mid-write must not destroy the last good save."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _restore_state(self, state):
        self.points = [tuple(p) for p in state.get("points", [])]
        self.point_widths = [tuple(w) for w in state.get("point_widths", [])]
        self.obstacles = [tuple(o) for o in state.get("obstacles", [])]
        self.closed = bool(state.get("closed", False))
        self.selected_point_idx = None
        self.preview_point = None
        self._sweep_owns_points = False
        self._sweep_source = None

        for key, var in [("resolution", self.resolution),
                         ("real_width", self.real_width),
                         ("real_height", self.real_height),
                         ("left_width", self.left_width),
                         ("right_width", self.right_width),
                         ("boundary_width", self.boundary_width),
                         ("obstacle_radius", self.obstacle_radius),
                         ("bg_opacity", self.bg_opacity),
                         ("bg_rotation", self.bg_rotation)]:
            if state.get(key) is not None:
                var.set(state[key])

        # widths list must stay parallel to points even if the file is odd
        while len(self.point_widths) < len(self.points):
            self.point_widths.append((self.left_width.get(), self.right_width.get()))
        del self.point_widths[len(self.points):]

        bg = state.get("background_path")
        if bg and bg == self.background_path and self.background_image is not None:
            pass                      # same floor plan already loaded
        elif bg and os.path.exists(bg):
            try:
                self._set_background(bg, page=state.get("background_page", 0))
            except Exception as e:
                print("Could not reload floor plan:", e)
        elif bg:
            self.calib_status.set(f"Floor plan not found: {os.path.basename(bg)}")

        self.save_button.config(state=tk.NORMAL if self.closed else tk.DISABLED)
        self.delete_button.config(state=tk.NORMAL if self.points else tk.DISABLED)
        self._last_saved_state = self._snapshot()
        self.update_display()

    def _autosave_tick(self):
        """Write the design whenever it has changed since the last write."""
        try:
            state = self._snapshot()
        except tk.TclError:
            # a numeric entry is mid-edit and unparseable — try again next tick
            self.root.after(AUTOSAVE_INTERVAL_MS, self._autosave_tick)
            return
        if state != self._last_saved_state and not self._is_empty_state(state):
            state["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
            try:
                self._write_project(AUTOSAVE_FILE, state)
                self._last_saved_state = self._snapshot()
            except OSError as e:
                print("Autosave failed:", e)
        self.root.after(AUTOSAVE_INTERVAL_MS, self._autosave_tick)

    def _on_close(self):
        try:
            state = self._snapshot()
            if not self._is_empty_state(state):
                state["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
                self._write_project(AUTOSAVE_FILE, state)
        except (tk.TclError, OSError) as e:
            print("Final autosave failed:", e)
        self.root.destroy()

    def _offer_restore(self):
        """On launch, offer to reopen whatever the last session was working on."""
        if not AUTOSAVE_FILE.exists():
            return
        try:
            with open(AUTOSAVE_FILE) as f:
                state = json.load(f)
        except (OSError, ValueError) as e:
            print("Could not read autosave:", e)
            return
        if self._is_empty_state(state):
            return

        when = state.get("saved_at", "unknown time").replace("T", " ")
        n_pts = len(state["points"])
        n_obs = len(state["obstacles"])
        if messagebox.askyesno(
                "Restore previous session",
                f"Found an autosaved track from {when}:\n\n"
                f"  {n_pts} waypoints, {n_obs} obstacles\n\n"
                f"Restore it?"):
            self._restore_state(state)

    def open_project(self):
        path = filedialog.askopenfilename(
            title="Open track project",
            initialdir=str(AUTOSAVE_DIR) if AUTOSAVE_DIR.exists() else None,
            filetypes=[("Track project", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path) as f:
                state = json.load(f)
        except (OSError, ValueError) as e:
            messagebox.showerror("Open project", f"Could not read that file:\n{e}")
            return
        if "points" not in state:
            messagebox.showerror("Open project",
                                 "That file is not a track project.")
            return
        self._restore_state(state)

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
        json_file = f"{ts}.json"
        img.save(png_file)
        with open(yaml_file, "w") as f:
            f.write(f'image: "{png_file}"\n'
                    f'resolution: {self.resolution.get()}\n'
                    f'origin: [0.0, 0.0, 0.000000]\n'
                    f'negate: 0\n'
                    f'occupied_thresh: 0.45\n'
                    f'free_thresh: 0.196\n')

        # the editable design alongside the exported grid, so this track can be
        # reopened later via "Open Project…"
        state = self._snapshot()
        state["saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        self._write_project(json_file, state)
        print(f"Saved {png_file}, {yaml_file} and {json_file}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RacetrackDesigner(root)
    root.mainloop()