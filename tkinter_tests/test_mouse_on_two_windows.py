import os
import tkinter as tk
from pathlib import Path

import cv2
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

    def get_image_dimensions(self):
        self.image_height, self.image_width, _ = self.image.shape
        self.image_aspect_ratio = self.image_height / self.image_width

        self.zoomed_image_coords = (0, self.image_width, 0, self.image_height)

    def load_image_from_file(self):
        file_path = Path(r"C:\Users\rafaelfc\Data\imgs\out_000001.jpg")
        image = cv2.imread(str(file_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image_original = image.copy()
        self.image = image

        self.zoomed_image_original = image.copy()
        self.zoomed_image = image

        self.get_image_dimensions()
        self.create_second_window()

    def create_second_window(self):
        if self.second_window is None:
            self.second_window = tk.Toplevel(self.image_navigation_window)
            self.second_window.title("Zoomed")

            self.second_window_canvas = tk.Canvas(self.second_window, width=216, height=1024)
            self.second_window_canvas.pack()
            self.second_window_canvas.bind("<Motion>", self.on_mouse_motion_second_window)


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

        self.display_image_navigation_window()

    def on_mouse_motion(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

        print('window 1', event.x, event.y)

    def on_mouse_motion_second_window(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

        print('window 2', event.x, event.y)

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
        self.display_image_second_window()
        self.image_navigation_window.mainloop()


if __name__ == "__main__":

    root = tk.Tk()
    app = ImageViewerApp(root)
    app.image_navigation_canvas.bind("<Motion>", app.on_mouse_motion)
    app.image_navigation_canvas.bind("<MouseWheel>", app.on_mouse_wheel)

    app.run()
