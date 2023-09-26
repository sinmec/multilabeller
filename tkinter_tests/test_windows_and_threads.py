import queue
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
from PIL import Image, ImageTk


class MyWindow(tk.Toplevel):
    def __init__(self, parent, title, shared_queue):
        super().__init__(parent)
        self.title(title)
        self.label = tk.Label(self, text=f"This is {title}")
        self.label.pack()


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

        self.main_window = root

        self.rectangle_ROI_zoom_count = 30
        self.rectangle_ROI_width = 0
        self.rectangle_ROI_height = 0
        self.rectangle_ROI_zoom = 1.0

        self.initialize_main_window()
        self.initialize_queue()

        self.current_window_canvas = None

    def initialize_main_window(self):
        self.main_window.title("Test")

    def initialize_queue(self):
        self.shared_queue = queue.Queue()

    def display_image_window(self, window_canvas, window_title):
        image = Image.fromarray(self.image)
        photo = ImageTk.PhotoImage(image=image)

        window_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        window_canvas.photo = photo
        window_canvas.title = window_title

    def run_window_1(self):
        self.window_1 = MyWindow(self.main_window, 'window_1', self.shared_queue)
        self.window_1_canvas = tk.Canvas(self.window_1, width=216, height=1024)
        self.window_1_canvas.pack()
        while True:
            self.display_image_window(self.window_1_canvas, 'window_1-test')
            self.window_1_canvas.bind("<Motion>", self.on_mouse_motion)
            self.window_1_canvas.bind("<Enter>", self.set_focus_to_current_window)
            time.sleep(0.1)

    def run_window_2(self):
        self.window_2 = MyWindow(self.main_window, 'window_2', self.shared_queue)
        self.window_2_canvas = tk.Canvas(self.window_2, width=216, height=1024)

        self.window_2_canvas.pack()

        while True:
            self.display_image_window(self.window_2_canvas, 'window_2-test')
            self.window_2_canvas.bind("<Motion>", self.on_mouse_motion)
            self.window_2_canvas.bind("<Enter>", self.set_focus_to_current_window)
            self.window_2_canvas.bind("<space>", self.on_key_press)

            time.sleep(0.1)

    # def update_status(self, canvas):
    #     status_bar = tk.Label(canvas, text="Applying bitwise operation...", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    #     status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_key_press(self, event):
        print('barra de espaco')
        self.image = cv2.bitwise_not(self.image)
        # self.update_status(self.window_2_canvas)

    def on_key_press_threshold_image(self, event):
        _, self.image = cv2.threshold(self.image_original, 200, 255,
                                      cv2.THRESH_BINARY_INV)

    def on_key_press_reset_image(self, event):
        self.image = self.image_original.copy()

    def on_mouse_motion(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

    def set_focus_to_current_window(self, event):
        event.widget.focus_set()

    def get_image_dimensions(self):
        self.image_height, self.image_width, _ = self.image.shape
        self.image_aspect_ratio = self.image_height / self.image_width

        self.zoomed_image_coords = (0, self.image_width, 0, self.image_height)

    def initialize_rectangle_ROI(self):
        self.rectangle_ROI_width = self.image_width // self.rectangle_ROI_zoom_count
        self.rectangle_ROI_height = self.image_height // self.rectangle_ROI_zoom_count

    #
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

    def start(self):
        self.load_image_from_file()
        thread1 = threading.Thread(target=self.run_window_1, )
        thread2 = threading.Thread(target=self.run_window_2, )
        thread1.start()
        thread2.start()

    def run(self):

        self.main_window.bind("<F9>", self.on_key_press_threshold_image)
        self.main_window.bind("<F11>", self.on_key_press_reset_image)

        self.main_window.mainloop()
        time.sleep(0.1)


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageViewerApp(root)
    app.start()

    # app.image_navigation_canvas.bind("<Motion>", app.on_mouse_motion)
    # app.image_navigation_canvas.bind("<MouseWheel>", app.on_mouse_wheel)
    # if app.second_window_canvas is not None:
    #     app.second_window_canvas.bind("<Motion>", app.on_mouse_motion_second_window)

    app.run()
