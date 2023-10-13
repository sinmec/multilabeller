import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import yaml

from src.multilabeller.image_manipulator.image_manipulator import ImageManipulator
from src.multilabeller.window.window import Window

if os.name == "nt":
    os_option = "windows"
if os.name == "posix":
    os_option = "linux"


class ImageViewerApp:
    def __init__(self, root):
        self.root_window = root
        self.image_manipulator = None
        self.config = None

        self.navigation_window = None
        self.annotation_window = None

        self.read_config_file()
        self.initialize_main_window()
        self.initialize_queue()

    def read_config_file(self):
        try:
            with open("config.yml", "r") as config_file:
                self.config = yaml.safe_load(config_file)
        except FileExistsError:
            print("Configuration file 'config.yml' was not found.")

    def initialize_main_window(self):
        self.root_window.title(self.config["root_window"]["name"])

    def initialize_queue(self):
        self.shared_queue = queue.Queue()

    def load_image_from_file(self):
        file_path = Path(self.config["test_image"])
        image = cv2.imread(str(file_path), 1)

        self.image_manipulator = ImageManipulator(image, self.config)

    def configure_window(self, window, w, h):
        if w or h:
            _width = w
            _height = h
            window.canvas = tk.Canvas(window, width=_width, height=_height)
            window.canvas.pack()
        else:
            _width = self.image_manipulator.image_original_width
            _height = self.image_manipulator.image_original_height
            window.canvas = tk.Canvas(window, width=_width, height=_height)
            window.canvas.pack()

    def initialize_windows(self):
        self.navigation_window = Window(
            self.root_window,
            self.config["navigation_window"]["title"],
            self.config,
            self.shared_queue,
        )
        self.configure_window(self.navigation_window, None, None)

        self.annotation_window = Window(
            self.root_window,
            self.config["annotation_window"]["title"],
            self.config,
            self.shared_queue,
        )
        self.configure_window(
            self.annotation_window,
            self.config["image_viewer"]["width"],
            self.config["image_viewer"]["height"],
        )

        self.setup_run()

    def setup_run(self):
        def run_navigation_window():
            self.navigation_window.set_image_manipulator(self.image_manipulator)
            while True:
                self.navigation_window.display_image()
                # TODO: Create a window handler. This is unnecessary
                self.navigation_window.canvas.bind(
                    self.config["mouse_motion"][os_option],
                    self.navigation_window.get_mouse_position,
                )

                if os_option == "linux":
                    self.navigation_window.canvas.bind(
                        self.config["mouse_wheel"][os_option]["bind1"],
                        self.navigation_window.modify_ROI_zoom,
                    )
                    self.navigation_window.canvas.bind(
                        self.config["mouse_wheel"][os_option]["bind2"],
                        self.navigation_window.modify_ROI_zoom,
                    )
                elif os_option == "windows":
                    self.navigation_window.canvas.bind(
                        self.config["mouse_wheel"][os_option],
                        self.navigation_window.modify_ROI_zoom,
                    )

                self.navigation_window.bind("<F9>", self.navigation_window.lock_image)

                if not self.navigation_window.annotation_mode:
                    self.navigation_window.draw_ROI()
                time.sleep(0.01)

        self.navigation_window.loop = run_navigation_window

        # TODO: Create Runner
        def run_annotation_window():
            self.annotation_window.set_image_manipulator(self.image_manipulator)
            while True:
                self.annotation_window.display_zoomed_image()
                self.annotation_window.canvas.bind(
                    self.config["mouse_motion"][os_option],
                    self.annotation_window.get_mouse_position,
                )

                if (
                    self.navigation_window.annotation_mode
                ):  # Todo: when this is True, it creates a point, need to fix
                    self.annotation_window.canvas.bind(
                        self.config["left_mouse_click"][os_option],
                        self.annotation_window.store_annotation_point,
                    )

                    self.image_manipulator.draw_annotation_point(
                        self.image_manipulator.zoomed_image,
                        self.annotation_window.point_x,
                        self.annotation_window.point_y,
                    )

                    (
                        self.navigation_window.point_x,
                        self.navigation_window.point_y,
                    ) = self.image_manipulator.translate_points(
                        self.annotation_window.point_x, self.annotation_window.point_y
                    )

                    self.image_manipulator.draw_annotation_point(
                        self.image_manipulator.image,
                        self.navigation_window.point_x,
                        self.navigation_window.point_y,
                    )

                time.sleep(0.01)

        self.annotation_window.loop = run_annotation_window

    # TODO: Think on a smarter solution
    def window_translator(self):
        def translation_function(x1, y1):
            self.mouse_rec_x = self.x1 + int(self.mouse_x2 / self.rectangle_ROI_zoom)

            self.mouse_rec_y = self.y1 + int(self.mouse_y2 / self.rectangle_ROI_zoom)

    def start(self):
        self.load_image_from_file()
        self.initialize_windows()
        thread1 = threading.Thread(target=self.annotation_window.run)
        thread2 = threading.Thread(target=self.navigation_window.run)
        thread1.start()
        thread2.start()

    def run(self):
        self.root_window.mainloop()
        time.sleep(0.1)
