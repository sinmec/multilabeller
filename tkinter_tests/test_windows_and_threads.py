import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk


class MyWindow(tk.Toplevel):
    def __init__(self, parent, title, shared_queue):
        super().__init__(parent)
        self.title(title)
        self.label = tk.Label(self, text=f"This is {title}")
        self.label.pack()


class ImageViewerApp:
    def __init__(self, root):
        self.image_rectangle_clean = None
        self.zoomed_image_clean = None
        self.x1 = None
        self.y1 = None
        self.x2 = None
        self.y2 = None
        self.mouse_x2 = None
        self.mouse_y2 = None
        self.mouse_rec_x = None
        self.mouse_rec_y = None
        self.second_window_canvas = None
        self.second_window = None
        self.wheel = None
        self.mouse_x = 0
        self.mouse_y = 0
        self.image = None
        self.image_original = None
        self.zoomed_image = None
        self.zoomed_image_original = None
        self.image_width = 0
        self.image_height = 0
        self.image_aspect_ratio = 0
        self.main_window = root
        self.rectangle_ROI_zoom_count = 30
        self.rectangle_ROI_width = 0
        self.rectangle_ROI_height = 0
        self.rectangle_ROI_zoom = 1.0

        self.initialize_main_window()

        self.initialize_queue()

        self.current_window_canvas = None
        self.lock_zoom = False

    def initialize_main_window(self):
        self.main_window.title("Main")

    def initialize_queue(self):
        self.shared_queue = queue.Queue()

    def display_image_window(self, window_canvas, window_title):
        if window_canvas == self.window_1_canvas:
            image = Image.fromarray(self.image)
            photo = ImageTk.PhotoImage(image=image)

            window_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            window_canvas.photo = photo
            window_canvas.title = window_title
        elif window_canvas == self.window_2_canvas:
            image = Image.fromarray(self.zoomed_image)
            photo = ImageTk.PhotoImage(image=image)

            window_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            window_canvas.photo = photo
            window_canvas.title = window_title

    def run_window_1(self):
        self.window_1 = MyWindow(self.main_window, "window_1", self.shared_queue)
        self.window_1_canvas = tk.Canvas(self.window_1, width=216, height=1024)
        self.window_1_canvas.pack()
        while True:
            self.display_image_window(self.window_1_canvas, "window_1-test")
            self.window_1_canvas.bind("<Motion>", self.on_mouse_motion)
            self.window_1_canvas.bind("<Button-4>", self.on_mouse_wheel)
            self.window_1_canvas.bind("<Button-5>", self.on_mouse_wheel)
            self.window_1_canvas.bind("<Enter>", self.set_focus_to_current_window)
            self.window_1_canvas.bind("<F1>", self.lock_image)
            time.sleep(0.001)

    def run_window_2(self):
        self.window_2 = MyWindow(self.main_window, "window_2", self.shared_queue)
        self.window_2_canvas = tk.Canvas(self.window_2, width=216, height=1024)

        self.window_2_canvas.pack()

        while True:
            self.display_image_window(self.window_2_canvas, "window_2-test")
            self.window_2_canvas.bind("<Motion>", self.on_mouse_motion_second_window)
            self.window_2_canvas.bind("<Enter>", self.set_focus_to_current_window)
            self.window_2_canvas.bind("<space>", self.on_key_press)
            self.window_2_canvas.bind(
                "<Double-Button-1>", self.circle_clone_on_main_window
            )
            self.window_2_canvas.bind(
                "<Double-Button-1>", self.circle_test_second_window, add="+"
            )

            time.sleep(0.001)

    def update_zoomed_image(self):
        x1, x2, y1, y2 = self.zoomed_image_coords
        image_ROI = self.image_original[y1:y2, x1:x2]

        new_size = (
            int(self.rectangle_ROI_zoom * (x2 - x1)),
            int(self.rectangle_ROI_zoom * (y2 - y1)),
        )

        self.zoomed_image = cv2.resize(image_ROI, new_size)
        self.zoomed_image_clean = self.zoomed_image

        self.window_2_canvas.config(width=new_size[0], height=new_size[1])

    def circle_test_second_window(self, event):
        # this function creates a rectangle test on the second window
        circle_radius = int(10 * self.rectangle_ROI_zoom)
        circle_color = (0, 255, 0)
        circle_thickness = int(2 * self.rectangle_ROI_zoom)

        self.zoomed_image = self.zoomed_image_clean

        self.zoomed_image = cv2.circle(
            self.zoomed_image.copy(),
            (self.mouse_x2, self.mouse_y2),
            circle_radius,
            circle_color,
            circle_thickness,
        )

        # self.display_image_second_window(circle)

    def circle_clone_on_main_window(self, event):
        circle_radius = int(15 / self.rectangle_ROI_zoom)
        circle_color = (0, 255, 0)
        circle_thickness = int(4 / self.rectangle_ROI_zoom)

        self.image = self.image_rectangle_clean

        self.image = cv2.circle(
            self.image.copy(),
            (self.mouse_rec_x, self.mouse_rec_y),
            circle_radius,
            circle_color,
            circle_thickness,
        )

    def on_key_press(self, event):
        self.image = cv2.bitwise_not(self.image)
        self.zoomed_image = cv2.bitwise_not(self.zoomed_image)
        # self.update_status(self.window_2_canvas)

    def on_key_press_reset_image(self, event):
        self.image = self.image_original.copy()

    def lock_image(self, event):
        self.lock_zoom = not self.lock_zoom

    def on_mouse_motion(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        if not self.lock_zoom:
            self.update_rectangle_size()
            self.draw_rectangle_ROI()
            self.update_zoomed_image()

    def on_mouse_motion_second_window(self, event):
        self.mouse_x2 = event.x
        self.mouse_y2 = event.y

        self.mouse_rec_x = self.x1 + int(self.mouse_x2 / self.rectangle_ROI_zoom)

        self.mouse_rec_y = self.y1 + int(self.mouse_y2 / self.rectangle_ROI_zoom)

    def set_focus_to_current_window(self, event):
        event.widget.focus_set()

    def get_image_dimensions(self):
        self.image_height, self.image_width, _ = self.image.shape
        self.image_aspect_ratio = self.image_height / self.image_width

        self.zoomed_image_coords = (0, self.image_width, 0, self.image_height)

    def initialize_rectangle_ROI(self):
        self.rectangle_ROI_width = self.image_width // self.rectangle_ROI_zoom_count
        self.rectangle_ROI_height = self.image_height // self.rectangle_ROI_zoom_count

    def update_rectangle_size(self):
        min_rectangle_width = 10
        min_rectangle_height = min_rectangle_width * self.image_aspect_ratio

        max_zoom = float(self.image_height / min_rectangle_height)
        min_zoom = 1.01

        h = 1.0 / max_zoom

        self.rectangle_ROI_zoom = np.clip(
            h * self.rectangle_ROI_zoom_count, min_zoom, max_zoom
        )

        self.rectangle_ROI_width = int(self.image_width / self.rectangle_ROI_zoom)
        self.rectangle_ROI_height = int(self.image_height / self.rectangle_ROI_zoom)

    def draw_rectangle_ROI(self):
        rectangle_color = (0, 255, 0)
        rectangle_width = 2

        self.x1 = int(self.mouse_x - self.rectangle_ROI_width / 2)
        self.y1 = int(self.mouse_y - self.rectangle_ROI_height / 2)
        self.x2 = int(self.mouse_x + self.rectangle_ROI_width / 2)
        self.y2 = int(self.mouse_y + self.rectangle_ROI_height / 2)

        self.x1 = max(2, self.x1)
        self.y1 = max(2, self.y1)
        self.x2 = min(self.image_width - 2, self.x2)
        self.y2 = min(self.image_height + 2, self.y2)

        self.zoomed_image_coords = (self.x1, self.x2, self.y1, self.y2)

        self.image = cv2.rectangle(
            self.image_original.copy(),
            (self.x1, self.y1),
            (self.x2, self.y2),
            rectangle_color,
            rectangle_width,
        )  # TODO: Why do we need this copy?
        self.image_rectangle_clean = self.image.copy()

    def on_mouse_wheel(self, event):
        if os.name == "nt":
            if event.delta > 0:
                self.rectangle_ROI_zoom_count += 1
            elif event.delta < 0:
                self.rectangle_ROI_zoom_count -= 1
        elif os.name == "posix":
            if event.num == 4:
                self.rectangle_ROI_zoom_count += 1
            elif event.num == 5:
                self.rectangle_ROI_zoom_count -= 1

    def load_image_from_file(self):
        file_path = Path(r"out_000001.jpg")
        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image_original = image.copy()
        self.image = image

        self.zoomed_image_original = image.copy()
        self.zoomed_image = image

        self.get_image_dimensions()
        self.initialize_rectangle_ROI()

    def start(self):
        self.load_image_from_file()
        thread1 = threading.Thread(
            target=self.run_window_1,
        )
        thread2 = threading.Thread(
            target=self.run_window_2,
        )
        thread1.start()
        thread2.start()

    def run(self):
        self.main_window.bind("<F11>", self.on_key_press_reset_image)

        self.main_window.mainloop()
        time.sleep(0.1)


if __name__ == "__main__":
    root = tk.Tk()

    app = ImageViewerApp(root)
    app.start()
    app.run()
