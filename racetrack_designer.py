import tkinter as tk
import numpy as np
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator
from PIL import Image, ImageTk, ImageDraw
import cv2
import datetime

class RacetrackDesigner:
    def __init__(self, root):
        self.root = root
        self.root.title("Racetrack Designer")
        
        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=800, height=600, bg='white')
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Control panel
        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        # Parameters
        self.resolution = tk.DoubleVar(value=0.06250)
        self.real_width = tk.DoubleVar(value=27.0)
        self.real_height = tk.DoubleVar(value=10.0)
        self.left_width = tk.DoubleVar(value=0.6)
        self.right_width = tk.DoubleVar(value=0.6)
        self.boundary_width = tk.DoubleVar(value=0.1)
        
        self.create_entry(control_frame, "Resolution (m/cell):", self.resolution, 0)
        self.create_entry(control_frame, "Real Width (m):", self.real_width, 1)
        self.create_entry(control_frame, "Real Height (m):", self.real_height, 2)
        self.create_entry(control_frame, "Left Width (m):", self.left_width, 3)
        self.create_entry(control_frame, "Right Width (m):", self.right_width, 4)
        self.create_entry(control_frame, "Boundary Width (m):", self.boundary_width, 5)
        
        # Action buttons
        button_frame = tk.Frame(control_frame)
        button_frame.grid(row=6, columnspan=2, pady=10)
        
        self.save_button = tk.Button(button_frame, text="Save", state=tk.DISABLED, command=self.save_grid)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        self.delete_button = tk.Button(button_frame, text="Delete Last", state=tk.DISABLED, command=self.delete_last_point)
        self.delete_button.pack(side=tk.LEFT, padx=5)
        
        # State variables
        self.points = []
        self.preview_point = None
        self.closed = False
        self.grid_image = None
        self.cursor_crosshair = []
        
        # Event bindings
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Motion>", self.preview_spline)
        self.canvas.bind("<Configure>", self.resize_canvas)
        
        # Parameter change tracking
        for var in [self.resolution, self.real_width, self.real_height,
                    self.left_width, self.right_width, self.boundary_width]:
            var.trace_add('write', self.param_changed)

    def create_entry(self, parent, label, var, row):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky='w')
        entry = tk.Entry(parent, textvariable=var, width=10)
        entry.grid(row=row, column=1, sticky='e')
        return entry

    def real_to_canvas(self, real_x, real_y):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        x = (real_x / self.real_width.get()) * cw
        y = (real_y / self.real_height.get()) * ch
        return x, y

    def canvas_to_real(self, x, y):
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        rx = (x / cw) * self.real_width.get()
        ry = (y / ch) * self.real_height.get()
        return rx, ry

    def add_point(self, event):
        if self.closed:
            return  # Don't add points after loop is closed
            
        rx, ry = self.canvas_to_real(event.x, event.y)
        
        # Check for closure with snapping
        if len(self.points) >= 3:
            first_x, first_y = self.points[0]
            dist = np.hypot(rx - first_x, ry - first_y)
            if dist < 1.0:  # Snap threshold
                rx, ry = first_x, first_y
                self.closed = True
                self.save_button.config(state=tk.NORMAL)
                self.points.append(self.points[0])
                self.delete_button.config(state=tk.NORMAL)
                self.update_display()
                return
                
        self.points.append((rx, ry))
        self.delete_button.config(state=tk.NORMAL)
        self.update_display()

    def delete_last_point(self):
        if not self.points:
            return
            
        if self.closed:
            # Remove the duplicate first point
            self.points.pop()
            self.closed = False
            self.save_button.config(state=tk.DISABLED)
        else:
            self.points.pop()
            
        if not self.points:
            self.delete_button.config(state=tk.DISABLED)
            
        self.update_display()

    def preview_spline(self, event):
        if not self.points:
            self.draw_crosshair(event.x, event.y)
            return
            
        if self.closed:
            self.draw_crosshair(event.x, event.y)
            return
            
        rx, ry = self.canvas_to_real(event.x, event.y)
        
        # Check for closure snapping in preview
        if len(self.points) >= 3 and not self.closed:
            first_x, first_y = self.points[0]
            dist = np.hypot(rx - first_x, ry - first_y)
            if dist < 1.0:  # Snap threshold
                self.preview_point = (first_x, first_y)
            else:
                self.preview_point = (rx, ry)
        else:
            self.preview_point = (rx, ry)
            
        self.draw_crosshair(event.x, event.y)
        self.update_display()

    def draw_crosshair(self, x, y):
        # Remove previous crosshair
        for item in self.cursor_crosshair:
            self.canvas.delete(item)
        self.cursor_crosshair = []
        
        # Draw crosshair (horizontal and vertical lines across entire canvas)
        self.cursor_crosshair.append(self.canvas.create_line(0, y, self.canvas.winfo_width(), y, fill='gray', dash=(2,2)))
        self.cursor_crosshair.append(self.canvas.create_line(x, 0, x, self.canvas.winfo_height(), fill='gray', dash=(2,2)))
        
        # Draw center square
        size = 5
        self.cursor_crosshair.append(self.canvas.create_rectangle(x-size, y-size, x+size, y+size, outline='black', fill='blue'))

    def generate_spline(self):
        if len(self.points) < 2 and not self.preview_point:
            return None, None, False

        points = self.points.copy()
        if self.preview_point and not self.closed:
            points.append(self.preview_point)

        if len(points) < 2:
            return None, None, False

        x = np.array([p[0] for p in points])
        y = np.array([p[1] for p in points])
        closed = self.closed or (len(points) >= 3 and np.allclose(points[-1], points[0]))

        try:
            t = np.linspace(0, 1, len(points)) if closed else np.arange(len(points))
            akima_x = Akima1DInterpolator(t, x)
            akima_y = Akima1DInterpolator(t, y)
            t_new = np.linspace(t.min(), t.max(), 200)
            return akima_x(t_new), akima_y(t_new), closed
        except Exception as e:
            print("Spline error:", e)
            return None, None, False

    def generate_polygon(self, x, y, lw, rw):
        if x is None or len(x) < 2:
            return []

        dx = np.gradient(x)
        dy = np.gradient(y)
        norm = np.hypot(dx, dy)
        dx /= norm
        dy /= norm

        left_x = x - dy * lw
        left_y = y + dx * lw
        right_x = x + dy * rw
        right_y = y - dx * rw

        return np.concatenate([np.column_stack([left_x, left_y]),
                              np.column_stack([right_x[::-1], right_y[::-1]])])

    def update_display(self):
        self.canvas.delete("all")
        
        # Generate grid
        res = 1/self.resolution.get()
        grid_w = int(self.real_width.get() * res)
        grid_h = int(self.real_height.get() * res)
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
        
        # Generate track
        x_spline, y_spline, closed = self.generate_spline()
        if x_spline is not None:
            polygon = self.generate_polygon(x_spline, y_spline, self.left_width.get(), self.right_width.get())
            if polygon.size > 0:
                scaled_poly = (polygon * res).astype(np.int32)
                cv2.fillPoly(grid, [scaled_poly], color=1)
                
                # Generate boundary using generate_polygon but with white
                boundary_poly = self.generate_polygon(x_spline, y_spline, self.left_width.get() - self.boundary_width.get(), self.right_width.get() - self.boundary_width.get())
                if boundary_poly.size > 0:
                    scaled_boundary = (boundary_poly * res).astype(np.int32)
                    cv2.fillPoly(grid, [scaled_boundary], color=0)
        
        # Convert grid to image
        img = Image.fromarray((1 - grid) * 255)
        img = img.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.NEAREST)
        self.photo = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Draw points and preview
        self.draw_points_and_preview()

    def draw_points_and_preview(self):
        for i, (rx, ry) in enumerate(self.points):
            x, y = self.real_to_canvas(rx, ry)
            color = 'green' if i == 0 and len(self.points) > 2 else 'red'
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill=color, outline='black')
            
        if self.preview_point and not self.closed:
            x, y = self.real_to_canvas(*self.preview_point)
            self.canvas.create_oval(x-5, y-5, x+5, y+5, fill='blue')
            if self.points:
                lx, ly = self.real_to_canvas(*self.points[-1])
                self.canvas.create_line(lx, ly, x, y, fill='blue')

    def param_changed(self, *args):
        self.update_display()

    def resize_canvas(self, event):
        self.update_display()

    def save_grid(self):
        # Generate grid (same as in update_display)
        res = 1/self.resolution.get()
        grid_w = int(self.real_width.get() * res)
        grid_h = int(self.real_height.get() * res)
        grid = np.zeros((grid_h, grid_w), dtype=np.uint8)

        x_spline, y_spline, closed = self.generate_spline()
        if x_spline is not None:
            polygon = self.generate_polygon(x_spline, y_spline, self.left_width.get(), self.right_width.get())
            if polygon.size > 0:
                scaled_poly = (polygon * res).astype(np.int32)
                cv2.fillPoly(grid, [scaled_poly], color=1)

                # Generate boundary using generate_polygon but with white
                boundary_poly = self.generate_polygon(x_spline, y_spline, self.left_width.get() - self.boundary_width.get(), self.right_width.get() - self.boundary_width.get())
                if boundary_poly.size > 0:
                    scaled_boundary = (boundary_poly * res).astype(np.int32)
                    cv2.fillPoly(grid, [scaled_boundary], color=0)

        # Create image from grid (PGM is a grayscale format)
        img = Image.fromarray((1 - grid) * 255)

        # Generate timestamp for filename
        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

        png_filename = f"{timestamp}.png"
        yaml_filename = f"{timestamp}.yaml"

        # Save image files
        img.save(png_filename)

        # Create and save YAML with parameters
        yaml_content = f"""image: "{png_filename}"
resolution: {self.resolution.get()}
origin: [0.0, 0.0, 0.000000]
negate: 0
occupied_thresh: 0.45
free_thresh: 0.196
"""

        with open(yaml_filename, "w") as f:
            f.write(yaml_content)

        print(f"Grid saved to {png_filename} and YAML config saved to {yaml_filename}")


if __name__ == "__main__":
    root = tk.Tk()
    app = RacetrackDesigner(root)
    root.mainloop()