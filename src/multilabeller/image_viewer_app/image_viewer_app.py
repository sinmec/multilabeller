import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.multilabeller.image_manipulator.image_manipulator import ImageManipulator
from src.multilabeller.window.window import Window
from src.multilabeller.circle import Circle
from src.multilabeller.contour import Contour
from src.multilabeller.selector import Selector
from src.multilabeller.SAM.sam import SegmentAnything

if os.name == "nt":
    os_option = "windows"
if os.name == "posix":
    os_option = "linux"


class ImageViewerApp:
    def __init__(self, root):
        self.image_original = None
        self.zoomed_image_original = None
        self.select_mode = False
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
        self.circle_id = -1
        self.id = 0
        self.read_config_file()
        self.initialize_SAM()
        self.initialize_main_window()
        self.initialize_queue()
        self.current_contour = None
        self.selector = Selector()
        self.selected_contours = []

    def initialize_SAM(self):
        if self.config["SAM"]["device"] == "gpu":
            cuda_available = torch.cuda.is_available()
            assert (
                cuda_available
            ), "PyTorch is having trouble with CUDA. Please change device to 'cpu'"

        if self.config["SAM"]["model"] == "vit_b":
            SAM_model_filename = "sam_vit_b_01ec64.pth"
        elif self.config["SAM"]["model"] == "vit_l":
            SAM_model_filename = "sam_vit_l_0b3195.pth"
        elif self.config["SAM"]["model"] == "vit_h":
            SAM_model_filename = "sam_vit_h_4b8939.pth"
        SAM_model_file = Path("SAM", SAM_model_filename)

        message = (
            f"{os.path.basename(SAM_model_file)} not found at src/multilabeller/SAM folder."
            f"\nPlease download the model file at "
            f"https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints"
        )
        assert os.path.isfile(SAM_model_file), message

        SAM_config = {
            "device": self.config["SAM"]["device"],
            "model": {"name": self.config["SAM"]["model"], "file": SAM_model_file},
        }

        self.SAM = SegmentAnything(SAM_config)

        print("SAM initialized successfully!")

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
            window.status_bar = tk.Label(
                window,
                text="F9 -> Lock Image | C -> Contour Mode | B -> Circle Mode |"
                " V -> Select Mode | Backspace -> Delete Contours |"
                " S -> Auto Segmentation",
                bd=1,
                relief=tk.SUNKEN,
                anchor=tk.W,
            )
            window.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
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

            self.annotation_window.bind(
                self.config["left_mouse_click"][os_option],
                self.annotation_window.store_annotation_point,
            )

            self.annotation_window.bind(
                self.config["left_mouse_click"][os_option],
                self.mouse_circle_callback,
                add="+",
            )

            self.annotation_window.bind(
                self.config["left_mouse_click"][os_option],
                self.mouse_contour_callback,
                add="+",
            )

            self.annotation_window.bind(
                self.config["left_mouse_click"][os_option],
                self.mouse_select_callback,
                add="+",
            )

            self.annotation_window.bind("<Key>", self.trigger)

            while True:
                self.navigation_window.display_image(
                    self.image_manipulator.zoomed_image
                )
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

                self.navigation_window.bind(
                    "<F9>", self.navigation_window.lock_image, add="+"
                )
                self.navigation_window.bind("<F9>", self.save_image, add="+")

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

                if self.navigation_window.annotation_mode:
                    if self.circle_mode:
                        if self.id != self.circle_id:
                            self.circle_id = self.id
                            self.current_circle = Circle(self.id)

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
                            self.annotation_window.point_x,
                            self.annotation_window.point_y,
                        )

                        self.delete_points()

                        self.draw_circle(
                            self.current_circle,
                            self.image_manipulator.zoomed_image,
                            self.image_manipulator.image,
                        )

                    if self.contour_mode:
                        if self.id != self.contour_id:
                            self.contour_id = self.id
                            self.current_contour = Contour(self.id)

                        if self.current_contour.i == 0:
                            self.clean_image = (
                                self.image_manipulator.zoomed_image.copy()
                            )
                            self.clean_manipulator_image = (
                                self.image_manipulator.image.copy()
                            )
                        else:
                            self.image_manipulator.draw_annotation_point(
                                self.image_manipulator.zoomed_image,
                                self.annotation_window.point_x,
                                self.annotation_window.point_y,
                            )

                            self.image_manipulator.draw_annotation_point(
                                self.image_manipulator.image,
                                self.navigation_window.point_x,
                                self.navigation_window.point_y,
                            )

                        (
                            self.navigation_window.point_x,
                            self.navigation_window.point_y,
                        ) = self.image_manipulator.translate_points(
                            self.annotation_window.point_x,
                            self.annotation_window.point_y,
                        )

                        if self.contour_confirm:
                            self.contours_list.append(self.current_contour)
                            self.image_manipulator.zoomed_image = self.clean_image
                            self.image_manipulator.image = self.clean_manipulator_image

                            self.create_contour_lines(
                                self.current_contour,
                                self.image_manipulator.zoomed_image,
                                self.current_contour.points,
                            )

                            self.create_contour_lines(
                                self.current_contour,
                                self.image_manipulator.image,
                                self.current_contour.translated_points,
                            )

                            # print(self.contours_list)
                            self.id = self.id + 1
                            self.contour_confirm = not self.contour_confirm

                    if self.select_mode:
                        if self.selector.i != 0:  # TODO: GAMBIARRA
                            # self.image_manipulator.draw_annotation_point(
                            #     self.image_manipulator.zoomed_image,
                            #     self.annotation_window.point_x,
                            #     self.annotation_window.point_y,
                            # )

                            if (
                                self.annotation_window.point_x
                                and self.annotation_window.point_y
                            ):
                                self.select_contour(
                                    self.annotation_window.point_x,
                                    self.annotation_window.point_y,
                                )

                            self.selector.i = 0

                else:
                    self.navigation_window.point_x = None
                    self.navigation_window.point_y = None
                    self.annotation_window.point_x = None
                    self.annotation_window.point_y = None

                time.sleep(0.01)

        self.annotation_window.loop = run_annotation_window

    def save_image(self, event):
        self.zoomed_image_original = self.image_manipulator.zoomed_image.copy()
        self.image_original = self.image_manipulator.image.copy()

    def mouse_select_callback(self, event):
        if self.select_mode:
            self.selector.point_check()

    def select_contour(self, point_x, point_y):
        for obj in self.contours_list:
            if obj.__class__.__name__ == "Circle":
                dist_to_center = np.sqrt(
                    pow((point_x - obj.center[0]), 2)
                    + pow((point_y - obj.center[1]), 2)
                )

                if dist_to_center <= obj.radius:
                    if obj.color == (255, 0, 0):
                        obj.color = (0, 255, 0)  # verde = selecionado
                        self.selected_contours.append(obj)
                        # print(self.selected_contours)
                    elif obj.color == (0, 255, 0):  # vermelho = deselecionado
                        obj.color = (255, 0, 0)
                        self.selected_contours.remove(obj)
                        # print(self.selected_contours)

                    self.update_circle(
                        obj,
                        self.image_manipulator.zoomed_image,
                        self.image_manipulator.image,
                    )

            if obj.__class__.__name__ == "Contour":
                point = (point_x, point_y)
                cnt_points = np.array([obj.points], np.int32)
                dist = cv2.pointPolygonTest(cnt_points, point, measureDist=True)

                if dist >= 0:
                    if obj.color == (255, 0, 0):
                        obj.color = (0, 255, 0)  # verde = selecionado
                        self.selected_contours.append(obj)
                        # print(self.selected_contours)
                    elif obj.color == (0, 255, 0):  # vermelho = deselecionado
                        obj.color = (255, 0, 0)
                        self.selected_contours.remove(obj)
                        # print(self.selected_contours)

                    self.create_contour_lines(
                        obj, self.image_manipulator.zoomed_image, obj.points
                    )

                    self.create_contour_lines(
                        obj, self.image_manipulator.image, obj.translated_points
                    )

    def trigger(self, event):
        if event.char == "c":
            self.contour_mode = not self.contour_mode
            self.circle_mode = False
            self.select_mode = False
            if self.contour_mode:
                print("Contour mode on")
            else:
                print("Contour mode off")
        elif event.char == "b":
            self.circle_mode = not self.circle_mode
            self.contour_mode = False
            self.select_mode = False
            if self.circle_mode:
                print("Circle mode on")
            else:
                print("Circle mode off")
        elif event.char == "v":
            self.select_mode = not self.select_mode
            self.contour_mode = False
            self.circle_mode = False
            if self.select_mode:
                print("Select mode on")
            else:
                print("Select mode off")
        elif event.char == " ":  # spacebar
            self.contour_confirm = not self.contour_confirm
            print("Contour saved")
        elif event.keysym == "BackSpace":
            self.delete_selected_contours()
        elif event.keysym == "s":
            self.auto_segmentation()

    def auto_segmentation(self):
        print("Applying SAM...")
        self.SAM.apply(self.zoomed_image_original.copy())

        for contour in self.SAM.contours:
            for point in contour.points:
                if point:
                    translated_point = self.image_manipulator.translate_points(
                        point[0], point[1]
                    )
                    contour.add_contour_points(None, translated_point)

            self.contours_list.append(contour)
            self.create_contour_lines(
                contour, self.image_manipulator.zoomed_image, contour.points
            )
            self.create_contour_lines(
                contour, self.image_manipulator.image, contour.translated_points
            )

        print(f"SAM found {len(self.SAM.contours)} contours!")

    def mouse_circle_callback(self, event):
        if self.circle_mode:
            (
                self.navigation_window.point_x,
                self.navigation_window.point_y,
            ) = self.image_manipulator.translate_points(
                self.annotation_window.point_x, self.annotation_window.point_y
            )
            # TODO: Need to fix this gambiarra above

            self.current_circle.add_circle_points(
                self.annotation_window.point_x,
                self.annotation_window.point_y,
                self.navigation_window.point_x,
                self.navigation_window.point_y,
            )

    def delete_selected_contours(self):
        for selected in self.selected_contours:
            if selected in self.contours_list:
                self.contours_list.remove(selected)

        self.recreate_contours()

    def recreate_contours(self):
        self.image_manipulator.zoomed_image = self.zoomed_image_original.copy()
        self.image_manipulator.image = self.image_original.copy()

        for obj in self.contours_list:
            if obj.__class__.__name__ == "Circle":
                self.update_circle(
                    obj,
                    self.image_manipulator.zoomed_image,
                    self.image_manipulator.image,
                )
            elif obj.__class__.__name__ == "Contour":
                self.create_contour_lines(
                    obj, self.image_manipulator.zoomed_image, obj.points
                )
                self.create_contour_lines(
                    obj, self.image_manipulator.image, obj.translated_points
                )

    def mouse_contour_callback(self, event):
        if self.contour_mode:
            (
                self.navigation_window.point_x,
                self.navigation_window.point_y,
            ) = self.image_manipulator.translate_points(
                self.annotation_window.point_x, self.annotation_window.point_y
            )

            # TODO: Need to fix this gambiarra above

            self.current_contour.add_contour_points(
                [self.annotation_window.point_x, self.annotation_window.point_y],
                [self.navigation_window.point_x, self.navigation_window.point_y],
            )

    def create_contour_lines(self, obj, image, points):
        for count, point in enumerate(points):
            cv2.line(image, points[count], points[count - 1], obj.color, obj.thickness)

    def delete_points(self):
        if self.current_circle.i == 0:
            self.clean_image = self.image_manipulator.zoomed_image.copy()
            self.clean_manipulator_image = self.image_manipulator.image.copy()
        elif self.current_circle.i == 2:
            self.image_manipulator.zoomed_image = self.clean_image
            self.image_manipulator.image = self.clean_manipulator_image

    def draw_circle(self, obj, image_annotation, image_manipulator):
        if obj.i == 2:
            cv2.circle(
                image_annotation, obj.center, obj.radius, obj.color, obj.thickness
            )

            cv2.circle(
                image_manipulator,
                obj.translated_center,
                obj.translated_circle_radius,
                obj.color,
                obj.thickness - 1,
            )

            obj.i = 0

            self.add_circle_to_list(obj)

    def update_circle(self, obj, image_annotation, image_manipulator):
        cv2.circle(image_annotation, obj.center, obj.radius, obj.color, obj.thickness)

        cv2.circle(
            image_manipulator,
            obj.translated_center,
            obj.translated_circle_radius,
            obj.color,
            obj.thickness - 1,
        )

    def add_circle_to_list(self, obj):
        self.contours_list.append(obj)
        # print(self.contours_list)
        self.id += 1

    def start(self):
        self.load_image_from_file()
        self.initialize_windows()
        thread1 = threading.Thread(target=self.annotation_window.run)
        thread2 = threading.Thread(target=self.navigation_window.run)
        thread1.start()
        thread2.start()

    def run(self):
        self.root_window.mainloop()
