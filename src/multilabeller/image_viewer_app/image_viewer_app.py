import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.multilabeller.image_manipulator.image_manipulator import ImageManipulator
from src.multilabeller.window.window import Window
from src.multilabeller.circle import Circle
from src.multilabeller.contour import Contour

if os.name == "nt":
    os_option = "windows"
if os.name == "posix":
    os_option = "linux"


class ImageViewerApp:
    def __init__(self, root):
        self.contour_confirm = False
        self.contours_list = []
        self.contour_mode = False
        self.clean_manipulator_image = None
        self.j = 0
        self.clean_image = []
        self.root_window = root
        self.image_manipulator = None
        self.config = None
        self.circle_mode = False
        self.navigation_window = None
        self.annotation_window = None
        self.contour_id = -1
        self.id = 0
        self.read_config_file()
        self.initialize_main_window()
        self.initialize_queue()
        self.current_circle = None
        self.current_contour = None


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

            self.annotation_window.bind(self.config["left_mouse_click"][os_option],
                                        self.annotation_window.store_annotation_point)

            self.annotation_window.bind(self.config["left_mouse_click"][os_option],
                                        self.mouse_circle_callback, add="+")

            self.annotation_window.bind('<Key>', self.trigger)

            while True:
                self.navigation_window.display_image(self.image_manipulator.zoomed_image)
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
            self.current_circle = Circle()

            while True:
                self.annotation_window.display_zoomed_image()

                self.annotation_window.canvas.bind(
                    self.config["mouse_motion"][os_option],
                    self.annotation_window.get_mouse_position,
                )

                if self.navigation_window.annotation_mode:
                    if self.circle_mode:
                        if self.current_circle.i != 0:
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

                        self.delete_points()

                        self.create_circle(
                            self.current_circle
                        )

                        self.draw_circle(
                            self.image_manipulator.zoomed_image,
                            self.image_manipulator.image
                        )

                    if self.contour_mode:
                        if self.id != self.contour_id:
                            self.contour_id = self.id
                            self.current_contour = Contour(self.id)

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

                        if self.contour_confirm:
                            self.contours_list.append(self.current_contour)
                            self.id = self.id + 1
                            self.contour_confirm = not self.contour_confirm
                else:
                    self.navigation_window.point_x = None
                    self.navigation_window.point_y = None
                    self.annotation_window.point_x = None
                    self.annotation_window.point_y = None

                time.sleep(0.01)
        self.annotation_window.loop = run_annotation_window


    def trigger(self, event):
        if event.char == 'c':
            self.contour_mode = not self.contour_mode
        elif event.char == 'b':
            self.circle_mode = not self.circle_mode
        elif event.char == ' ':  # spacebar
            self.contour_confirm = not self.contour_confirm

    def mouse_circle_callback(self, event):
        if self.circle_mode:
            (
                self.navigation_window.point_x,
                self.navigation_window.point_y,
            ) = self.image_manipulator.translate_points(
                self.annotation_window.point_x, self.annotation_window.point_y
            )
            # TODO: Need to fix this quick fix above

            self.current_circle.add_circle_points(self.annotation_window.point_x, self.annotation_window.point_y,
                                                  self.navigation_window.point_x, self.navigation_window.point_y
                                                  )

    def create_circle(self, circle):
        if self.current_circle.i == 2:

            # circle on the annotation window

            point_1 = circle.points[0]
            point_2 = circle.points[1]

            self.center = [int((point_1[0] + point_2[0]) / 2), int((point_1[1] + point_2[1]) / 2)]

            self.circle_radius = int(np.sqrt(pow((point_2[0] - self.center[0]), 2) +
                                             pow((point_2[1] - self.center[1]), 2)))

            # circle on the navigation window

            point_1_translated = circle.translated_points[0]
            point_2_translated = circle.translated_points[1]

            self.translated_center = [int((point_1_translated[0] + point_2_translated[0]) / 2),
                                      int((point_1_translated[1] + point_2_translated[1]) / 2)]

            self.translated_circle_radius = int(np.sqrt(pow((point_2_translated[0] - self.translated_center[0]), 2) +
                                            pow((point_2_translated[1] - self.translated_center[1]), 2)))
            # todo: improve this

    def delete_points(self):
        if self.current_circle.i == 0:
            self.clean_image = self.image_manipulator.zoomed_image.copy()
            self.clean_manipulator_image = self.image_manipulator.image.copy()
        elif self.current_circle.i == 2:
            self.image_manipulator.zoomed_image = self.clean_image
            self.image_manipulator.image = self.clean_manipulator_image

    def draw_circle(self, image_annotation, image_manipulator):
        if self.current_circle.i == 2:
            cv2.circle(
                image_annotation, self.center, self.circle_radius, self.current_circle.color, self.current_circle.thickness
            )

            cv2.circle(
                image_manipulator, self.translated_center, self.translated_circle_radius,
                self.current_circle.color, self.current_circle.thickness - 1
            )

            self.current_circle.i = 0
        else:
            pass

    def start(self):
        self.load_image_from_file()
        self.initialize_windows()
        thread1 = threading.Thread(target=self.annotation_window.run)
        thread2 = threading.Thread(target=self.navigation_window.run)
        thread1.start()
        thread2.start()

    def run(self):
        self.root_window.mainloop()
