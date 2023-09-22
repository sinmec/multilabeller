import math
import os
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageTk




class ImageViewerApp:
    def __init__(self, root):
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

        self.image_navigation_window = root
        self.image_navigation_window.title("OpenCV and Tkinter Image Viewer")

        # self.load_button = ttk.Button(root, text="Load Image", command=self.load_and_display_image)
        # self.load_button.pack(padx=20, pady=10)

        self.image_navigation_canvas = tk.Canvas(root, width=216, height=1024)
        self.image_navigation_canvas.pack()

        self.rectangle_ROI_zoom_count = 30
        self.rectangle_ROI_width = 0
        self.rectangle_ROI_height = 0
        self.rectangle_ROI_zoom = 1.0

    def update_rectangle_size(self):

        min_rectangle_width = 10
        min_rectangle_height = min_rectangle_width * self.image_aspect_ratio


        max_zoom = float(self.image_height / min_rectangle_height)
        min_zoom = 1.01

        h = 1.0 / max_zoom

        self.rectangle_ROI_zoom = np.clip(h * self.rectangle_ROI_zoom_count, min_zoom, max_zoom)

        self.rectangle_ROI_width = int(self.image_width / self.rectangle_ROI_zoom)
        self.rectangle_ROI_height = int(self.image_height / self.rectangle_ROI_zoom)

    def draw_rectangle_ROI(self):

        rectangle_color = (0, 255, 0)
        rectangle_width = 2

        x1 = int(self.mouse_x - self.rectangle_ROI_width / 2)
        y1 = int(self.mouse_y - self.rectangle_ROI_height / 2)
        x2 = int(self.mouse_x + self.rectangle_ROI_width / 2)
        y2 = int(self.mouse_y + self.rectangle_ROI_height / 2)

        x1 = max(2, x1)
        y1 = max(2, y1)
        x2 = min(self.image_width - 2, x2)
        y2 = min(self.image_height + 2, y2)

        self.zoomed_image_coords = (x1, x2, y1, y2)

        self.image = cv2.rectangle(self.image_original.copy(), (x1, y1), (x2, y2), rectangle_color,
                                   rectangle_width)  # TODO: Why do we need this copy?

    def get_image_dimensions(self):
        self.image_height, self.image_width, _ = self.image.shape
        self.image_aspect_ratio = self.image_height / self.image_width

        self.zoomed_image_coords = (0, self.image_width, 0, self.image_height)

    def initialize_rectangle_ROI(self):
        self.rectangle_ROI_width = self.image_width // self.rectangle_ROI_zoom_count
        self.rectangle_ROI_height = self.image_height // self.rectangle_ROI_zoom_count

        a = 2

    def load_image_from_file(self):
        file_path = Path(r"C:\Users\rafaelfc\Data\imgs\out_000001.jpg")
        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image_original = image.copy()
        self.image = image

        self.zoomed_image_original = image.copy()
        self.zoomed_image = image

        self.get_image_dimensions()
        self.initialize_rectangle_ROI()
        self.create_second_window()

    def load_image_from_dialog(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.tif *.tiff")])
        if file_path:
            image = cv2.imread(str(file_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image = image
            self.get_image_dimensions()
            self.initialize_rectangle_ROI()
            self.image_original = image.copy()

    def create_second_window(self):
        if self.second_window is None:
            self.second_window = tk.Toplevel(self.image_navigation_window)
            self.second_window.title("Zoomed")

            self.second_window_canvas = tk.Canvas(self.second_window, width=216, height=1024)
            self.second_window_canvas.pack()

    def update_zoomed_image(self):
        x1, x2, y1, y2 = self.zoomed_image_coords
        image_ROI = self.image_original[y1:y2, x1:x2]

        new_size = (int(self.rectangle_ROI_zoom * (x2-x1)),
                    int(self.rectangle_ROI_zoom * (y2-y1)))

        self.zoomed_image = cv2.resize(image_ROI, new_size)

        self.second_window_canvas.config(width=new_size[0], height=new_size[1])


    def display_image_navigation_window(self):
        image = Image.fromarray(self.image)
        photo = ImageTk.PhotoImage(image=image)

        self.image_navigation_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.image_navigation_canvas.photo = photo

    def display_image_second_window(self):
        image = Image.fromarray(self.zoomed_image)
        photo = ImageTk.PhotoImage(image=image)

        self.second_window_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.second_window_canvas.photo = photo


    def draw_circle_on_screen(self):
        self.image = cv2.circle(self.image_original.copy(), (self.mouse_x, self.mouse_y), 5, (0, 0, 255), -1)
        self.display_image_navigation_window()

    def on_mouse_motion(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y
        self.update_rectangle_size()
        self.draw_rectangle_ROI()
        self.update_zoomed_image()
        self.display_image_navigation_window()
        self.display_image_second_window()

        print('window 1', event.x, event.y)

    def on_mouse_motion_second_window(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

        print('window 2', event.x, event.y)
        # self.update_rectangle_size()
        # self.draw_rectangle_ROI()
        # self.update_zoomed_image()
        # self.display_image_navigation_window()
        # self.display_image_second_window()

    def on_mouse_wheel(self, event):
        if os.name == 'nt':
            if event.delta > 0:
                self.rectangle_ROI_zoom_count += 1
            elif event.delta < 0:
                self.rectangle_ROI_zoom_count -= 1
        elif os.name == 'posix':
            pass





    def run(self):
        self.load_image_from_file()
        self.display_image_navigation_window()
        self.image_navigation_window.mainloop()


if __name__ == "__main__":

    root = tk.Tk()
    app = ImageViewerApp(root)
    app.image_navigation_canvas.bind("<Motion>", app.on_mouse_motion)
    app.image_navigation_canvas.bind("<MouseWheel>", app.on_mouse_wheel)
    if app.second_window_canvas is not None:
        app.second_window_canvas.bind("<Motion>", app.on_mouse_motion_second_window)


    app.run()

