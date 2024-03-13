import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog
import h5py
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from src.multilabeller.SAM.sam import SegmentAnything
from src.multilabeller.circle import Circle
from src.multilabeller.drawed_contour import DrawedContour
from src.multilabeller.ellipse import Ellipse
from src.multilabeller.image_manipulator.image_manipulator import ImageManipulator
from src.multilabeller.selector import Selector
from src.multilabeller.window.window import Window

if os.name == "nt":
    os_option = "windows"
if os.name == "posix":
    os_option = "linux"


class ImageViewerApp:
    def __init__(self, root, contour_collection):
        self.previous_img_button = None
        self.next_img_button = None
        self.image_files_path = None
        self.file_path = None
        self.load_file_button = None
        self.number_of_exports = 0
        self.contour_collection = contour_collection
        self.output_path = None

        self.root_window = root
        self.export_button = None
        self.image_manipulator = None
        self.config = None
        self.file_index = None
        self.navigation_window = None
        self.annotation_window = None

        self.selector = Selector()
        self.read_config_file()
        self.initialize_SAM()

        self.initialize_queue()
        self.initialize_main_window()

        self.operation_mode = None

        self.current_drawed_contour = None
        self.current_circle = None
        self.current_ellipse = None

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

    def load_file_index_file(self, file_path):
        try:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)
        except FileExistsError:
            print("Configuration file 'file_index.yml' was not found.")

    def save_file_index(self, file_path, data):
        with open(file_path, 'w') as file:
            yaml.dump(data, file)

    def initialize_main_window(self):
        self.root_window.title(self.config["root_window"]["name"])
        self.root_window.geometry("260x140")
        file_index = self.load_file_index_file("file_index.yml")

        if file_index["program_running"]:
            self.open_directory()
        else:
            load_file_button = tk.Button(
                self.root_window, text="Open images directory", command=self.open_directory
            )
            load_file_button.pack()

    def initialize_buttons(self):
        export_button = tk.Button(
            self.root_window, text="Export Contours", command=self.export_contours
        )

        frame_arrows = tk.Frame(self.root_window)
        frame_arrows.pack(pady=10)

        next_img_button = tk.Button(
            frame_arrows, text="Next image >", command=self.next_image_button
        )
        previous_img_button = tk.Button(
            frame_arrows, text="< Previous image", command=self.previous_image_button
        )

        previous_img_button.pack(side=tk.LEFT, padx=5, pady=10)
        next_img_button.pack(side=tk.LEFT, padx=5, pady=10)

        frame_arrows.place(relx=0.5, rely=0.5, anchor=tk.N)

        export_button.pack()

    def next_image_button(self):
        file_index = self.load_file_index_file("file_index.yml")

        if file_index["file_index"] != len(self.image_files):
            file_index["file_index"] += 1
        else:
            file_index["file_index"] = 0

        self.save_file_index("file_index.yml", file_index)

        self.reset_program()

    def previous_image_button(self):
        file_index = self.load_file_index_file("file_index.yml")

        if file_index["file_index"] != 0:
            file_index["file_index"] -= 1
        else:
            file_index["file_index"] = len(self.image_files)

        self.save_file_index("file_index.yml", file_index)

        self.reset_program()

    def reset_program(self):
        file_index = self.load_file_index_file("file_index.yml")
        file_index["program_running"] = True
        self.save_file_index("file_index.yml", file_index)
        self.stop()
        os.system('python main.py')

    def copy_h5_contents(self, src_path, dest_path):
        with h5py.File(src_path, "r") as src_file:
            with h5py.File(dest_path, "w") as dest_file:
                for item in src_file.keys():
                    src_file.copy(item, dest_file)

    def export_contours(self):
        print("Started contours exporting...")

        image_folder_name = self.image_files_path.stem

        self.output_path = Path(f"{self.config['output_path']}", image_folder_name)
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.number_of_exports == 0:
            N_h5_files = 0
            for _file in self.output_path.iterdir():
                if not str(_file).endswith('.h5'):
                    continue
                if not image_folder_name in str(_file):
                    continue

                N_h5_files += 1
            self.number_of_exports += N_h5_files

        h5_file_name = f"{image_folder_name}_{self.number_of_exports:06d}.h5"
        h5_file_path = Path(self.output_path, h5_file_name)

        h5_file = h5py.File(h5_file_path, "w")
        h5_file.attrs['date'] = datetime.now().strftime('%Y_%m_%d_%H_%M')

        current_image = self.image_files[self.file_index]
        img_group = h5_file.create_group(f"{current_image.name}")
        img_group.create_dataset(
            "img", data=np.array(cv2.imread(str(current_image), 1))
        )
        contours_group = img_group.create_group("contours")

        n_contours = 0
        for item in range(len(self.contour_collection.items)):
            contour = self.contour_collection.items[item]
            if not contour.valid:
                continue
            if contour.navigation_window_contour is None:
                continue
            cv2_contours = contour.navigation_window_contour
            contours_group.create_dataset(f"cnt_{n_contours:06d}", data=np.array(cv2_contours))
            n_contours += 1

        h5_file.close()


        self.number_of_exports += 1
        self.merge_exported_files()

        print(f"A total of {n_contours} contours from {current_image} successfully exported to {self.output_path}")

    def merge_exported_files(self):

        image_folder_name = self.image_files_path.stem
        output_path = Path(f"{self.config['output_path']}")

        h5_file_name = f"{image_folder_name}.h5"
        h5_file_path = Path(output_path, h5_file_name)

        h5_file = h5py.File(h5_file_path, "w")
        h5_file.attrs['date'] = datetime.now().strftime('%Y_%m_%d_%H_%M')

        # 1. merge images
        n_contours = 0
        for _file in self.output_path.iterdir():
            tpm_h5_file = h5py.File(_file, "r")
            for _img in tpm_h5_file.keys():
                if not _img in h5_file:
                    img_group = h5_file.create_group(f"{_img}")
                    img_group.create_dataset(
                        "img", data=np.array(tpm_h5_file[_img]['img'][...], dtype=np.uint8)
                    )

                    # TODO: Check if it's necessary
                    if not 'contours' in img_group.keys():
                        contours_group = img_group.create_group("contours")

                contours_group = h5_file[_img]['contours']
                for cnt in tpm_h5_file[_img]['contours']:
                    contour = tpm_h5_file[_img]['contours'][cnt][...]
                    contours_group.create_dataset(f"cnt_{n_contours:06d}", data=contour)
                    n_contours += 1

            tpm_h5_file.close()

        h5_file.close()

    def initialize_queue(self):
        self.shared_queue = queue.Queue()

    def open_directory(self):
        file_index = self.load_file_index_file("file_index.yml")

        if not file_index["program_running"]:
            while True:
                self.image_files_path = Path(filedialog.askdirectory())
                file_index["current_path"] = str(self.image_files_path)

                self.save_file_index("file_index.yml", file_index)

                if self.image_files_path:
                    self.choose_images()
                    break
                else:
                    print("Please select a valid folder.")
        else:
            self.image_files_path = Path(f'{file_index["current_path"]}')
            self.choose_images()

    def choose_images(self):
        self.image_files = []
        for file in self.image_files_path.iterdir():
            IMAGE_FILE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".TIFF"]
            if not file.suffix in IMAGE_FILE_EXTENSIONS:
                continue
            self.image_files.append(file)
        self.image_files.sort()
        if len(self.image_files) == 0:
            print("Please select a valid folder")
            exit()  # TODO: How do we kill the program?

        self.select_image()
        self.initialize_buttons()

    def select_image(self):
        file_index = self.load_file_index_file("file_index.yml")
        print(f"Initializing {file_index['file_index'] + 1}° image")
        self.load_image_from_file(self.image_files[file_index["file_index"]])

    def load_image_from_file(self, img_path):
        file_path = Path(img_path)
        image = cv2.imread(str(file_path), 1)

        if self.annotation_window or self.navigation_window:
            if (
                self.annotation_window.winfo_exists()
                or self.navigation_window.winfo_exists()
            ):
                self.annotation_window.destroy()
                self.navigation_window.destroy()

        self.image_manipulator = ImageManipulator(image, self.config)

        self.reinitialize_context()

        self.start()

    def reinitialize_context(self):
        self.contour_collection.items = []
        self.annotation_objects = []
        self.current_drawed_contour = None
        self.current_circle = None
        self.current_ellipse = None


    def create_annotation_window_text(self):
        text = ""
        text += f"{self.config['shortcuts']['annotation_mode']}: Lock Image | "
        text += f"{self.config['shortcuts']['circle_mode']}: Circle Mode | "
        text += f"{self.config['shortcuts']['ellipse_mode']}: Ellipse Mode | "
        text += (
            f"{self.config['shortcuts']['drawed_contour_mode']}: Drawed Contour Mode | "
        )
        text += f"Space: Save Drawed Contour | "
        text += f"{self.config['shortcuts']['selection_mode']}: Select Mode | "
        text += f"{self.config['shortcuts']['delete_contour']}: Delete Contours | "
        text += f"{self.config['shortcuts']['apply_SAM']}: Auto Segmentation"

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
                self.mouse_ellipse_callback,
                add="+",
            )

            self.annotation_window.bind(
                self.config["mouse_wheel"][os_option],
                self.mouse_configure_ellipse_minor_axis_callback,
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
                "<MouseWheel>",
                self.mouse_ellipse_axes_callback,
                add="+",
            )

            self.annotation_window.bind(
                self.config["shortcuts"]["annotation_mode"],
                self.navigation_window.lock_annotation_image,
                add="+",
            )

            self.annotation_window.bind("<Key>", self.shortcut_selector, add="+")

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
                        if self.current_circle.finished:
                            self.current_circle = None

                    if self.operation_mode == "drawed_contour":
                        if self.current_drawed_contour is None:
                            self.current_drawed_contour = DrawedContour()
                            self.annotation_objects.append(self.current_drawed_contour)
                            self.contour_collection.items = self.annotation_objects
                            self.current_drawed_contour = self.annotation_objects[-1]

                    if self.operation_mode == "ellipse":
                        if self.current_ellipse is None:
                            self.current_ellipse = Ellipse()
                            self.annotation_objects.append(self.current_ellipse)
                            self.contour_collection.items = self.annotation_objects
                            self.current_ellipse = self.annotation_objects[-1]
                        if self.current_ellipse.in_configuration:
                            self.current_ellipse.configure_ellipse_parameters()
                            self.current_ellipse.create_minor_axis_annotation_points()
                        if self.current_ellipse.finished:
                            self.current_ellipse = None

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

    def mouse_configure_ellipse_minor_axis_callback(self, event):
        if self.operation_mode == "ellipse":
            if self.current_ellipse.in_configuration:
                if os.name == "nt":
                    if event.delta > 0:
                        self.current_ellipse.minor_axis += 1
                    elif event.delta < 0:
                        if self.current_ellipse.minor_axis > 0:
                            self.current_ellipse.minor_axis -= 1
                if os.name == "posix":
                    if event.num == 4:
                        self.current_ellipse.minor_axis += 1
                    elif event.num == 5:
                        if self.current_ellipse.minor_axis > 0:
                            self.current_ellipse.minor_axis -= 1

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

    def shortcut_selector(self, event):
        if event.char == self.config["shortcuts"]["drawed_contour_mode"]:
            print("Drawed contour mode")
            self.operation_mode = "drawed_contour"
        elif event.char == self.config["shortcuts"]["circle_mode"]:
            print("Circle mode")
            self.operation_mode = "circle"
        elif event.char == self.config["shortcuts"]["selection_mode"]:
            print("Selection mode")
            self.operation_mode = "selection"
        elif event.keysym == self.config["shortcuts"]["apply_SAM"]:
            print("SAM mode")
            self.auto_segmentation()
        elif event.keysym == self.config["shortcuts"]["ellipse_mode"]:
            print("Ellipse mode")
            self.operation_mode = "ellipse"
        elif event.char == self.config["shortcuts"]["save_contour"]:
            if self.operation_mode == "drawed_contour":
                self.save_drawed_contour()
            elif self.operation_mode == "ellipse":
                self.save_ellipse_contour()
        elif event.keysym == self.config["shortcuts"]["delete_contour"]:
            if self.operation_mode == "selection":
                self.invalidate_selected_contours()
        else:
            print(f"Please chose a valid option!")

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

    def save_ellipse_contour(self):
        if self.current_ellipse.points_annotation_window[-1] == None:
            print("Invalid number of ellipse points!")
            return

        self.current_ellipse.translate_from_annotation_to_navigation_windows(
            self.image_manipulator
        )
        self.current_ellipse.translate_from_annotation_to_navigation_windows(
            self.image_manipulator
        )
        self.current_ellipse.translate_from_navigation_to_annotation_windows(
            self.image_manipulator
        )
        self.current_ellipse.to_cv2_contour()
        self.current_ellipse.in_progress = False
        self.current_ellipse.in_configuration = False
        self.current_ellipse.finished = True
        self.current_ellipse = None

    def auto_segmentation(self):
        print("Applying SAM...")
        self.SAM.image_manipulator = self.image_manipulator
        self.SAM.apply(self.image_manipulator.annotation_image_buffer.copy())
        print("SAM contour detection finished!")
        print(f"SAM found {len(self.SAM.contours)} contours!")

        for SAM_contour in self.SAM.contours:
            self.annotation_objects.append(SAM_contour)
            self.contour_collection.items = self.annotation_objects

    # TODO: Merge with ellipse's one
    def mouse_circle_callback(self, event):
        if self.operation_mode == "circle":
            self.current_circle.add_points(
                self.annotation_window.point_x,
                self.annotation_window.point_y,
                self.image_manipulator,
            )

    def mouse_ellipse_callback(self, event):
        if self.operation_mode == "ellipse":
            self.current_ellipse.add_points(
                self.annotation_window.point_x,
                self.annotation_window.point_y,
                self.image_manipulator,
            )
            if self.current_ellipse.finished:
                self.current_ellipse = None

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
            self.current_drawed_contour.add_points(
                [self.annotation_window.point_x, self.annotation_window.point_y],
            )

    def mouse_ellipse_axes_callback(self, event):
        if self.operation_mode == "ellipse":
            if self.current_ellipse.in_configuration:
                if os.name == "nt":
                    if event.delta > 0:
                        self.current_ellipse.minor_axis += 1
                    elif event.delta < 0:
                        self.current_ellipse.minor_axis -= 1
                if os.name == "posix":
                    if event.num == 4:
                        self.current_ellipse.minor_axis += 1
                    elif event.num == 5:
                        self.current_ellipse.minor_axis -= 1
                self.current_ellipse.minor_axis = max(
                    self.current_ellipse.minor_axis, 0
                )
                self.current_ellipse.minor_axis = min(
                    self.current_ellipse.minor_axis, self.current_ellipse.major_axis
                )

    def start(self):
        self.initialize_windows()
        thread2 = threading.Thread(target=self.navigation_window.run)
        thread1 = threading.Thread(target=self.annotation_window.run)
        thread1.start()
        thread2.start()

    def stop(self):
        self.root_window.destroy()

    def run(self):
        self.root_window.mainloop()
