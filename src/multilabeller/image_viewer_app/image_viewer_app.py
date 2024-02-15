import os
import queue
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import torch
import yaml

from src.multilabeller.SAM.sam import SegmentAnything
from src.multilabeller.circle import Circle
from src.multilabeller.drawed_contour import DrawedContour
from src.multilabeller.image_manipulator.image_manipulator import ImageManipulator
from src.multilabeller.selector import Selector
from src.multilabeller.window.window import Window

if os.name == "nt":
    os_option = "windows"
if os.name == "posix":
    os_option = "linux"


class ImageViewerApp:
    def __init__(self, root, contour_collection):
        self.contour_collection = contour_collection

        self.root_window = root
        self.image_manipulator = None
        self.config = None
        self.navigation_window = None
        self.annotation_window = None

        self.selector = Selector()
        self.read_config_file()
        self.initialize_SAM()
        self.initialize_main_window()
        self.initialize_queue()

        self.operation_mode = None

        self.current_drawed_contour = None
        self.current_circle = None

        self.annotation_objects = []

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

    def create_annotation_window_text(self):
        text = ""
        text += f"{self.config['shortcuts']['annotation_mode']} -> Lock Image | "
        text += f"{self.config['shortcuts']['drawed_contour_mode']} ->  Drawed Contour Mode | "
        text += f"{self.config['shortcuts']['save_drawed_contour']} ->  Save Drawed Contour | "
        text += f"{self.config['shortcuts']['circle_mode']} ->  Circle Mode | "
        text += f"{self.config['shortcuts']['selection_mode']} ->  Select Mode | "
        text += f"{self.config['shortcuts']['delete_contour']} ->  Delete Contours | "
        text += f"{self.config['shortcuts']['apply_SAM']} ->  Auto Segmentation"

        return text

    def configure_window(self, window, w, h):
        if w or h:
            _width = w
            _height = h
            window.canvas = tk.Canvas(window, width=_width, height=_height)
            window.status_bar = tk.Label(
                window,
                text=self.create_annotation_window_text(),
                bd=1,
                relief=tk.SUNKEN,
                anchor=tk.W,
            )
            window.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
            window.canvas.pack()
        else:
            _width = self.image_manipulator.navigation_image_width
            _height = self.image_manipulator.navigation_image_height
            window.canvas = tk.Canvas(window, width=_width, height=_height)
            window.canvas.pack()

    def initialize_windows(self):
        self.navigation_window = Window(
            self.root_window,
            self.config["navigation_window"]["title"],
            self.config,
            self.shared_queue,
            self.contour_collection,
        )
        self.configure_window(self.navigation_window, None, None)

        self.annotation_window = Window(
            self.root_window,
            self.config["annotation_window"]["title"],
            self.config,
            self.shared_queue,
            self.contour_collection,
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
                self.navigation_window.display_navigation_image(
                    self.image_manipulator.annotation_image
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
                    self.config["shortcuts"]["annotation_mode"],
                    self.navigation_window.lock_annotation_image,
                )

                if not self.navigation_window.annotation_mode:
                    self.navigation_window.draw_ROI((0, 255, 0))
                else:
                    self.navigation_window.image_manipulator.draw_rectangle_ROI(
                        self.navigation_window.last_mouse_event_x,
                        self.navigation_window.last_mouse_event_y,
                        (255, 0, 0),
                    )

                time.sleep(0.01)

        self.navigation_window.loop = run_navigation_window

        def run_annotation_window():
            self.annotation_window.set_image_manipulator(self.image_manipulator)

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
                self.mouse_select_callback,
                add="+",
            )

            self.annotation_window.bind(
                self.config["left_mouse_click"][os_option],
                self.mouse_contour_callback,
                add="+",
            )

            self.annotation_window.bind(
                self.config["shortcuts"]["annotation_mode"],
                self.navigation_window.lock_annotation_image,
                add="+",
            )

            self.annotation_window.bind("<Key>", self.trigger, add="+")

            while True:

                self.annotation_window.display_annotation_image()

                self.annotation_window.canvas.bind(
                    self.config["mouse_motion"][os_option],
                    self.annotation_window.get_mouse_position,
                )

                if self.navigation_window.annotation_mode:
                    if self.operation_mode == "circle":
                        if self.current_circle is None:
                            self.current_circle = Circle()
                            self.annotation_objects.append(self.current_circle)
                            self.contour_collection.items = self.annotation_objects
                            self.current_circle = self.annotation_objects[-1]
                            print("soy circulito")
                        if self.current_circle.finished:
                            self.current_circle = None

                    if self.operation_mode == "drawed_contour":
                        if self.current_drawed_contour is None:
                            self.current_drawed_contour = DrawedContour()
                            self.annotation_objects.append(self.current_drawed_contour)
                            self.contour_collection.items = self.annotation_objects
                            self.current_drawed_contour = self.annotation_objects[-1]

                    if self.operation_mode == "selection":
                        if self.selector.valid:
                            self.select_contour()

                else:
                    self.navigation_window.point_x = None
                    self.navigation_window.point_y = None
                    self.annotation_window.point_x = None
                    self.annotation_window.point_y = None

                time.sleep(0.01)

        self.annotation_window.loop = run_annotation_window

    def mouse_select_callback(self, event):
        if self.operation_mode == "selection":
            self.selector.update_point(
                self.annotation_window.point_x, self.annotation_window.point_y
            )

    def select_contour(self):
        for obj in self.annotation_objects:
            if not obj.valid:
                continue
            if obj.in_progress:
                continue

            point = (self.selector.point_x, self.selector.point_y)
            cnt_points = obj.annotation_window_contour
            dist = cv2.pointPolygonTest(cnt_points, point, measureDist=True)
            if dist >= 0:
                obj.toggle_color()
                obj.toggle_selection()
                self.selector.valid = False

    def trigger(self, event):

        if event.char == self.config["shortcuts"]["drawed_contour_mode"]:
            print("Drawed contour mode")
            self.operation_mode = "drawed_contour"

        elif event.char == self.config["shortcuts"]["circle_mode"]:
            print("Circle mode")
            self.operation_mode = "circle"

        elif event.char == self.config["shortcuts"]["selection_mode"]:
            print("Selection mode")
            self.operation_mode = "selection"
        else:
            print(f"Please chose a valid option!")

        if self.operation_mode == "drawed_contour":
            if event.char == self.config["shortcuts"]["save_drawed_contour"]:
                self.save_drawed_contour()
                # TODO #5: Add ellipse
                # TODO #6: Check for SAM

        if self.operation_mode == "selection":
            if event.keysym == self.config["shortcuts"]["delete_contour"]:
                self.invalidate_selected_contours()

    def save_drawed_contour(self):
        if len(self.current_drawed_contour.points_annotation_window) < 3:
            print("Invalid number of drawed contour points!")
            return
        self.current_drawed_contour.translate_from_annotation_to_navigation_windows(
            self.image_manipulator
        )
        self.current_drawed_contour.to_cv2_contour()
        self.current_drawed_contour.in_progress = False
        self.current_drawed_contour.finished = True
        self.current_drawed_contour = None

    def auto_segmentation(self):
        print("Applying SAM...")
        self.SAM.apply(self.zoomed_image_original.copy())

        for contour in self.SAM.contours:
            for point in contour.points_annotation_window:
                if point:
                    translated_point = self.image_manipulator.translate_from_annotation_to_navigation_windows(
                        point[0], point[1]
                    )
                    contour.add_contour_points(None, translated_point)

            self.contours_list.append(contour)
            self.create_contour_lines(
                contour,
                self.image_manipulator.annotation_image,
                contour.points_annotation_window,
            )
            self.create_contour_lines(
                contour,
                self.image_manipulator.navigation_image,
                contour.points_navigation_window,
            )

        print(f"SAM found {len(self.SAM.contours)} contours!")

    def mouse_circle_callback(self, event):
        if self.operation_mode == "circle":
            self.current_circle.add_circle_points(
                self.annotation_window.point_x,
                self.annotation_window.point_y,
                self.image_manipulator,
            )

    def invalidate_selected_contours(self):
        N_invalid_contours = 0
        for obj in self.annotation_objects:
            if obj.selected:
                if obj.valid:
                    obj.valid = False
                    N_invalid_contours += 1
        print(f"A total of {N_invalid_contours} were removed (invalidated)!")

    def mouse_contour_callback(self, event):
        if self.operation_mode == "drawed_contour":
            self.current_drawed_contour.add_contour_points(
                [self.annotation_window.point_x, self.annotation_window.point_y],
            )

    def start(self):
        self.load_image_from_file()
        self.initialize_windows()
        thread1 = threading.Thread(target=self.annotation_window.run)
        thread2 = threading.Thread(target=self.navigation_window.run)
        thread1.start()
        thread2.start()

    def run(self):
        self.root_window.mainloop()
