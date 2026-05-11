import os
import queue
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
from src.multilabeller.circle import Circle, WheelCircle
from src.multilabeller.drawed_contour import DrawedContour
from src.multilabeller.ellipse import Ellipse
from src.multilabeller.image_manipulator.image_manipulator import ImageManipulator
from src.multilabeller.selector import Selector
from src.multilabeller.window.window import Window

if os.name == "nt":
    os_option = "windows"
if os.name == "posix":
    os_option = "linux"

WINDOW_REFRESH_INTERVAL_MS = 33


class ImageViewerApp:
    def __init__(self, root, contour_collection):
        self.previous_img_button = None
        self.next_img_button = None
        self.file_index = 0
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
        self.navigation_window = None

        self.windows_initialized = False
        self.navigation_loop_running = False
        self.annotation_loop_running = False

        self.annotation_window = None

        self.selector = Selector()
        self.read_config_file()
        self.initialize_SAM()
        self.initialize_main_window()
        self.initialize_queue()

        self.operation_mode = None

        self.current_drawed_contour = None
        self.current_circle = None
        self.current_wheel_circle = None
        self.current_ellipse = None

        self.last_wheel_circle_radius = 10

        self.annotation_objects = []

        self.h5_mode = False
        self.h5_file_path = None
        self.h5_images = []
        self.h5_image_raw = {}
        self.h5_contour_raw = {}

        self.buttons_initialized = False
        self.contours_visible = True

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
        self.root_window.geometry("260x170")
        self.root_window.protocol("WM_DELETE_WINDOW", self.close_application)

        load_file_button = tk.Button(
            self.root_window, text="Open images directory", command=self.open_directory
        )
        load_file_button.pack()

        load_h5_button = tk.Button(
            self.root_window, text="Open .h5 file", command=self.open_h5_file
        )
        load_h5_button.pack()

    def initialize_buttons(self):
        if self.buttons_initialized:
            return
        self.buttons_initialized = True

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
        if self.h5_mode:
            self.save_current_h5_contours()
            if self.file_index < len(self.h5_images) - 1:
                self.file_index += 1
            else:
                self.file_index = 0
            self.select_h5_image()
        else:
            if self.file_index != len(self.image_files):
                self.file_index += 1
            else:
                self.file_index = 0
            self.select_image()

    def previous_image_button(self):
        if self.h5_mode:
            self.save_current_h5_contours()
            if self.file_index != 0:
                self.file_index -= 1
            else:
                self.file_index = len(self.h5_images) - 1
            self.select_h5_image()
        else:
            if self.file_index != 0:
                self.file_index -= 1
            else:
                self.file_index = len(self.image_files)
            self.select_image()

    def copy_h5_contents(self, src_path, dest_path):
        with h5py.File(src_path, "r") as src_file:
            with h5py.File(dest_path, "w") as dest_file:
                for item in src_file.keys():
                    src_file.copy(item, dest_file)

    def export_contours(self):
        if self.h5_mode:
            self.export_h5_contours()
            return

        print("Started contours exporting...")

        image_folder_name = self.image_files_path.stem

        self.output_path = Path(f"{self.config['output_path']}", image_folder_name)
        self.output_path.mkdir(parents=True, exist_ok=True)

        if self.number_of_exports == 0:
            N_h5_files = 0
            for _file in self.output_path.iterdir():
                if not str(_file).endswith(".h5"):
                    continue
                if not image_folder_name in str(_file):
                    continue

                N_h5_files += 1
            self.number_of_exports += N_h5_files

        h5_file_name = f"{image_folder_name}_{self.number_of_exports:06d}.h5"
        h5_file_path = Path(self.output_path, h5_file_name)

        h5_file = h5py.File(h5_file_path, "w")
        h5_file.attrs["date"] = datetime.now().strftime("%Y_%m_%d_%H_%M")

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
            contours_group.create_dataset(
                f"cnt_{n_contours:06d}", data=np.array(cv2_contours)
            )
            n_contours += 1

        h5_file.close()

        self.number_of_exports += 1
        self.merge_exported_files()

        print(
            f"A total of {n_contours} contours from {current_image} successfully exported to {self.output_path}"
        )

    def merge_exported_files(self):

        image_folder_name = self.image_files_path.stem
        output_path = Path(f"{self.config['output_path']}")

        h5_file_name = f"{image_folder_name}.h5"
        h5_file_path = Path(output_path, h5_file_name)

        h5_file = h5py.File(h5_file_path, "w")
        h5_file.attrs["date"] = datetime.now().strftime("%Y_%m_%d_%H_%M")

        # 1. merge images
        n_contours = 0
        for _file in self.output_path.iterdir():
            tpm_h5_file = h5py.File(_file, "r")
            for _img in tpm_h5_file.keys():
                if not _img in h5_file:
                    img_group = h5_file.create_group(f"{_img}")
                    img_group.create_dataset(
                        "img",
                        data=np.array(tpm_h5_file[_img]["img"][...], dtype=np.uint8),
                    )

                    # TODO: Check if it's necessary
                    if not "contours" in img_group.keys():
                        contours_group = img_group.create_group("contours")

                contours_group = h5_file[_img]["contours"]
                for cnt in tpm_h5_file[_img]["contours"]:
                    contour = tpm_h5_file[_img]["contours"][cnt][...]
                    contours_group.create_dataset(f"cnt_{n_contours:06d}", data=contour)
                    n_contours += 1

            tpm_h5_file.close()

        h5_file.close()

    def initialize_queue(self):
        self.shared_queue = queue.Queue()

    def open_h5_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")]
        )
        if file_path:
            self.load_from_h5(Path(file_path))

    def load_from_h5(self, h5_path):
        self.h5_file_path = h5_path
        self.h5_mode = True
        self.h5_images = []
        self.h5_image_raw = {}
        self.h5_contour_raw = {}

        with h5py.File(h5_path, "r") as h5:
            for img_key in h5.keys():
                self.h5_images.append(img_key)

        if not self.h5_images:
            print("No images found in the h5 file.")
            return

        self.file_index = 0
        self.select_h5_image()
        self.initialize_buttons()

    def _load_h5_image_data(self, img_key):
        """Load image and contours for *img_key* from the h5 file into the cache."""
        with h5py.File(self.h5_file_path, "r") as h5:
            self.h5_image_raw[img_key] = np.array(
                h5[img_key]["img"][...], dtype=np.uint8
            )
            contours = []
            for cnt_key in sorted(h5[img_key]["contours"].keys()):
                contours.append(
                    np.array(h5[img_key]["contours"][cnt_key][...], dtype=np.int32)
                )
            self.h5_contour_raw[img_key] = contours

    def select_h5_image(self):
        img_key = self.h5_images[self.file_index]

        if img_key not in self.h5_image_raw:
            self._load_h5_image_data(img_key)

        image = self.h5_image_raw[img_key]

        self.image_manipulator = ImageManipulator(image, self.config)
        self.reinitialize_context()

        for cnt_array in self.h5_contour_raw[img_key]:
            contour = DrawedContour()
            contour.points_image = cnt_array[:, 0, :].tolist()
            contour.points_navigation_window = contour.points_image.copy()
            contour.navigation_window_contour = cnt_array
            contour.in_progress = False
            contour.finished = True
            self.annotation_objects.append(contour)

        self.contour_collection.items = self.annotation_objects

        if (
            not self.windows_initialized
            or self.navigation_window is None
            or self.annotation_window is None
        ):
            self.start()
            self.windows_initialized = True
        else:
            self.refresh_windows_for_new_image()

    def save_current_h5_contours(self):
        if not self.h5_mode or not self.h5_images:
            return
        img_key = self.h5_images[self.file_index]
        saved = []
        for idx, obj in enumerate(self.annotation_objects):
            if not obj.valid or not obj.finished:
                continue

            if obj.navigation_window_contour is None:
                valid_pts = [pt for pt in obj.points_image if pt is not None]
                if not valid_pts:
                    print(
                        f"Warning: skipping object #{idx + 1} ({obj.__class__.__name__}) "
                        "because it has no image points or contour geometry to export."
                    )
                    continue
                obj.update_window_points_from_image_points(self.image_manipulator)
                obj.to_cv2_contour()

            if obj.navigation_window_contour is None:
                print(
                    f"Warning: skipping object #{idx + 1} ({obj.__class__.__name__}) "
                    "because it has no contour geometry to export."
                )
                continue

            cnt = np.array(obj.navigation_window_contour, dtype=np.int32)
            saved.append(cnt)
        self.h5_contour_raw[img_key] = saved

    def export_h5_contours(self):
        print("Started h5 export...")
        self.save_current_h5_contours()

        default_name = self.h5_file_path.stem + "_modified.h5"
        output_file = filedialog.asksaveasfilename(
            defaultextension=".h5",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            initialfile=default_name,
            initialdir=str(self.h5_file_path.parent),
        )
        if not output_file:
            print("Export cancelled.")
            return

        output_path = Path(output_file)
        n_total = 0
        with h5py.File(output_path, "w") as h5_out:
            h5_out.attrs["date"] = datetime.now().strftime("%Y_%m_%d_%H_%M")
            with h5py.File(self.h5_file_path, "r") as h5_src:
                for img_key in self.h5_images:
                    img_group = h5_out.create_group(img_key)
                    if img_key in self.h5_image_raw:
                        img_group.create_dataset("img", data=self.h5_image_raw[img_key])
                    else:
                        img_group.create_dataset(
                            "img",
                            data=np.array(h5_src[img_key]["img"][...], dtype=np.uint8),
                        )
                    contours_group = img_group.create_group("contours")
                    if img_key in self.h5_contour_raw:
                        contours = self.h5_contour_raw[img_key]
                    else:
                        contours = [
                            np.array(
                                h5_src[img_key]["contours"][cnt_key][...],
                                dtype=np.int32,
                            )
                            for cnt_key in sorted(h5_src[img_key]["contours"].keys())
                        ]
                    for i, cnt in enumerate(contours):
                        contours_group.create_dataset(f"cnt_{i:06d}", data=cnt)
                    n_total += len(contours)
        print(
            f"Exported {n_total} contours across {len(self.h5_images)} images to {output_path}"
        )

    def open_directory(self):
        while True:
            self.image_files_path = Path(filedialog.askdirectory())
            if self.image_files_path:
                self.choose_images()
                break
            else:
                print("Please select a valid folder.")

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
        self.load_image_from_file(self.image_files[self.file_index])

    def refresh_windows_for_new_image(self):
        if self.navigation_window is None or self.annotation_window is None:
            self.windows_initialized = False
            return

        self.navigation_window.set_image_manipulator(self.image_manipulator)
        self.annotation_window.set_image_manipulator(self.image_manipulator)
        self.navigation_window.annotation_mode = False

        navigation_width = self.image_manipulator.navigation_image_width
        navigation_height = self.image_manipulator.navigation_image_height
        canvas_width, canvas_height = self.get_navigation_canvas_size(
            navigation_width, navigation_height
        )
        self.navigation_window.canvas.configure(
            width=canvas_width,
            height=canvas_height,
            scrollregion=(0, 0, navigation_width, navigation_height),
        )

        self.navigation_window.canvas.xview_moveto(0)
        self.navigation_window.canvas.yview_moveto(0)

        self.navigation_window.point_x = None
        self.navigation_window.point_y = None
        self.navigation_window.last_mouse_event_x = None
        self.navigation_window.last_mouse_event_y = None

        self.annotation_window.point_x = None
        self.annotation_window.point_y = None

        self.selector.valid = False

    def load_image_from_file(self, img_path):
        file_path = Path(img_path)
        image = cv2.imread(str(file_path), 1)

        self.image_manipulator = ImageManipulator(image, self.config)

        self.reinitialize_context()

        if (
            not self.windows_initialized
            or self.navigation_window is None
            or self.annotation_window is None
        ):
            self.start()
            self.windows_initialized = True
        else:
            self.refresh_windows_for_new_image()

    def stop_windows(self):
        self.navigation_loop_running = False
        self.annotation_loop_running = False
        self.windows_initialized = False

        for window in (self.annotation_window, self.navigation_window):
            if window is None:
                continue
            try:
                if window.winfo_exists():
                    window.destroy()
            except tk.TclError:
                pass

        self.annotation_window = None
        self.navigation_window = None

    def close_application(self):
        self.stop_windows()
        self.root_window.destroy()

    def reinitialize_context(self):
        self.contour_collection.items = []
        self.annotation_objects = []
        self.current_drawed_contour = None
        self.current_circle = None
        self.current_wheel_circle = None
        self.current_ellipse = None

    def create_annotation_window_text(self):
        text = ""
        text += f"{self.config['shortcuts']['annotation_mode']}: Lock Image | "
        text += f"{self.config['shortcuts']['circle_mode']}: Circle Mode | "
        text += f"{self.config['shortcuts']['wheel_circle_mode']}: Wheel Circle Mode | "
        text += f"{self.config['shortcuts']['ellipse_mode']}: Ellipse Mode | "
        text += (
            f"{self.config['shortcuts']['drawed_contour_mode']}: Drawed Contour Mode | "
        )
        text += f"Space: Save Drawed Contour | "
        text += f"{self.config['shortcuts']['selection_mode']}: Select Mode | "
        text += f"{self.config['shortcuts']['delete_contour']}: Delete Contours | "
        text += (
            f"{self.config['shortcuts']['contour_visibility_mode']}: Toggle Contours | "
        )
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
            canvas_width, canvas_height = self.get_navigation_canvas_size(
                _width, _height
            )

            window.canvas_frame = tk.Frame(window)
            window.canvas_frame.pack(fill=tk.BOTH, expand=True)

            window.canvas = tk.Canvas(
                window.canvas_frame,
                width=canvas_width,
                height=canvas_height,
                scrollregion=(0, 0, _width, _height),
            )
            window.vertical_scrollbar = tk.Scrollbar(
                window.canvas_frame,
                orient=tk.VERTICAL,
                command=window.canvas.yview,
            )
            window.horizontal_scrollbar = tk.Scrollbar(
                window.canvas_frame,
                orient=tk.HORIZONTAL,
                command=window.canvas.xview,
            )
            window.canvas.configure(
                xscrollcommand=window.horizontal_scrollbar.set,
                yscrollcommand=window.vertical_scrollbar.set,
            )

            window.canvas.grid(row=0, column=0, sticky="nsew")
            window.vertical_scrollbar.grid(row=0, column=1, sticky="ns")
            window.horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
            window.canvas_frame.rowconfigure(0, weight=1)
            window.canvas_frame.columnconfigure(0, weight=1)

    def get_navigation_canvas_size(self, image_width, image_height):
        max_width = max(400, self.root_window.winfo_screenwidth() - 100)
        max_height = max(300, self.root_window.winfo_screenheight() - 140)
        return min(image_width, max_width), min(image_height, max_height)

    def initialize_windows(self):
        self.navigation_window = Window(
            self.root_window,
            self.config["navigation_window"]["title"],
            self.config,
            self.shared_queue,
            self.contour_collection,
        )
        self.configure_window(self.navigation_window, None, None)
        self.navigation_window.contours_visible = self.contours_visible

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
        self.annotation_window.contours_visible = self.contours_visible

        self.setup_run()

    def setup_run(self):
        self.setup_navigation_bindings()
        self.setup_annotation_bindings()

    def setup_navigation_bindings(self):
        self.navigation_window.set_image_manipulator(self.image_manipulator)

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

    def setup_annotation_bindings(self):
        self.annotation_window.set_image_manipulator(self.image_manipulator)

        self.annotation_window.canvas.bind(
            self.config["mouse_motion"][os_option],
            self.annotation_window.get_mouse_position,
        )

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
            self.mouse_wheel_circle_callback,
            add="+",
        )

        self.annotation_window.bind(
            self.config["left_mouse_click"][os_option],
            self.mouse_ellipse_callback,
            add="+",
        )

        if os.name == "nt":
            self.annotation_window.bind(
                "<MouseWheel>",
                self.mouse_configure_ellipse_minor_axis_callback,
                add="+",
            )
            self.annotation_window.bind(
                "<MouseWheel>",
                self.mouse_configure_wheel_circle_radius_callback,
                add="+",
            )
        else:
            self.annotation_window.bind(
                self.config["mouse_wheel"]["linux"]["bind1"],
                self.mouse_configure_ellipse_minor_axis_callback,
                add="+",
            )
            self.annotation_window.bind(
                self.config["mouse_wheel"]["linux"]["bind2"],
                self.mouse_configure_ellipse_minor_axis_callback,
                add="+",
            )
            self.annotation_window.bind(
                self.config["mouse_wheel"]["linux"]["bind1"],
                self.mouse_configure_wheel_circle_radius_callback,
                add="+",
            )
            self.annotation_window.bind(
                self.config["mouse_wheel"]["linux"]["bind2"],
                self.mouse_configure_wheel_circle_radius_callback,
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

    def ensure_current_circle(self):
        if self.current_circle is None:
            self.current_circle = Circle()
            self.annotation_objects.append(self.current_circle)
            self.contour_collection.items = self.annotation_objects
            self.current_circle = self.annotation_objects[-1]

    def ensure_current_wheel_circle(self):
        if self.current_wheel_circle is None:
            self.current_wheel_circle = WheelCircle()
            self.current_wheel_circle.radius_annotation_window = (
                self.last_wheel_circle_radius
            )
            self.annotation_objects.append(self.current_wheel_circle)
            self.contour_collection.items = self.annotation_objects
            self.current_wheel_circle = self.annotation_objects[-1]

    def ensure_current_drawed_contour(self):
        if self.current_drawed_contour is None:
            self.current_drawed_contour = DrawedContour()
            self.annotation_objects.append(self.current_drawed_contour)
            self.contour_collection.items = self.annotation_objects
            self.current_drawed_contour = self.annotation_objects[-1]

    def ensure_current_ellipse(self):
        if self.current_ellipse is None:
            self.current_ellipse = Ellipse()
            self.annotation_objects.append(self.current_ellipse)
            self.contour_collection.items = self.annotation_objects
            self.current_ellipse = self.annotation_objects[-1]

    def mouse_select_callback(self, event):
        if self.operation_mode == "selection":
            self.selector.update_point(
                self.annotation_window.point_x, self.annotation_window.point_y
            )

    def mouse_configure_ellipse_minor_axis_callback(self, event):
        if self.operation_mode == "ellipse":
            self.ensure_current_ellipse()
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
            self.ensure_current_drawed_contour()
        elif event.char == self.config["shortcuts"]["circle_mode"]:
            print("Circle mode")
            self.operation_mode = "circle"
            self.ensure_current_circle()
        elif event.char == self.config["shortcuts"]["wheel_circle_mode"]:
            print("Wheel circle mode")
            self.operation_mode = "wheel_circle"
            self.ensure_current_wheel_circle()
        elif event.char == self.config["shortcuts"]["selection_mode"]:
            print("Selection mode")
            self.operation_mode = "selection"
        elif event.keysym == self.config["shortcuts"]["apply_SAM"]:
            print("SAM mode")
            self.auto_segmentation()
        elif event.keysym == self.config["shortcuts"]["ellipse_mode"]:
            print("Ellipse mode")
            self.operation_mode = "ellipse"
            self.ensure_current_ellipse()
        elif event.char == self.config["shortcuts"]["save_contour"]:
            if self.operation_mode == "drawed_contour":
                self.save_drawed_contour()
            elif self.operation_mode == "ellipse":
                self.save_ellipse_contour()
            elif self.operation_mode == "wheel_circle":
                self.save_wheel_circle_contour()
        elif event.keysym == self.config["shortcuts"]["delete_contour"]:
            if self.operation_mode == "selection":
                self.invalidate_selected_contours()
        elif event.char == self.config["shortcuts"]["contour_visibility_mode"]:
            self.toggle_contour_visibility()
        else:
            print(f"Please chose a valid option!")

    def toggle_contour_visibility(self):
        self.contours_visible = not self.contours_visible
        if self.navigation_window is not None:
            self.navigation_window.contours_visible = self.contours_visible
        if self.annotation_window is not None:
            self.annotation_window.contours_visible = self.contours_visible

        if self.image_manipulator is not None:
            if self.navigation_window is not None:
                self.navigation_window.set_image_manipulator(self.image_manipulator)
                self.navigation_window.display_navigation_image(
                    self.image_manipulator.annotation_image
                )
            if self.annotation_window is not None:
                self.annotation_window.set_image_manipulator(self.image_manipulator)
                self.annotation_window.display_annotation_image()

        if self.contours_visible:
            print("Contours visible")
        else:
            print("Contours hidden")

    def save_drawed_contour(self):
        if self.current_drawed_contour is None:
            print("No active drawed contour to save!")
            return
        if len(self.current_drawed_contour.points_annotation_window) < 3:
            print("Invalid number of drawed contour points!")
            return

        self.current_drawed_contour.translate_from_annotation_to_image(
            self.image_manipulator
        )
        self.current_drawed_contour.update_window_points_from_image_points(
            self.image_manipulator
        )
        self.current_drawed_contour.to_cv2_contour()

        self.current_drawed_contour.in_progress = False
        self.current_drawed_contour.finished = True
        self.current_drawed_contour = None

    def save_wheel_circle_contour(self):
        if self.current_wheel_circle is None:
            print("No active wheel circle to save!")
            return
        if self.current_wheel_circle.points_annotation_window[0] is None:
            print("Invalid circle center point!")
            return

        self.last_wheel_circle_radius = (
            self.current_wheel_circle.radius_annotation_window
        )

        self.current_wheel_circle.translate_from_annotation_to_image(
            self.image_manipulator
        )
        self.current_wheel_circle.update_window_points_from_image_points(
            self.image_manipulator
        )
        self.current_wheel_circle.to_cv2_contour()

        self.current_wheel_circle.in_progress = False
        self.current_wheel_circle.in_configuration = False
        self.current_wheel_circle.finished = True
        self.current_wheel_circle = None

    def save_ellipse_contour(self):
        if self.current_ellipse is None:
            print("No active ellipse to save!")
            return
        if self.current_ellipse.points_annotation_window[-1] is None:
            print("Invalid number of ellipse points!")
            return

        self.current_ellipse.translate_from_annotation_to_image(self.image_manipulator)
        self.current_ellipse.update_window_points_from_image_points(
            self.image_manipulator
        )
        self.current_ellipse.to_cv2_contour()

        self.current_ellipse.in_progress = False
        self.current_ellipse.in_configuration = False
        self.current_ellipse.finished = True
        self.current_ellipse = None

    def auto_segmentation(self):
        print("Applying SAM...")
        self.SAM.contours = []
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
            self.ensure_current_circle()
            self.current_circle.add_points(
                self.annotation_window.point_x,
                self.annotation_window.point_y,
                self.image_manipulator,
            )

    def mouse_wheel_circle_callback(self, event):
        if self.operation_mode == "wheel_circle":
            self.ensure_current_wheel_circle()
            self.current_wheel_circle.add_points(
                self.annotation_window.point_x,
                self.annotation_window.point_y,
                self.image_manipulator,
            )

    def mouse_ellipse_callback(self, event):
        if self.operation_mode == "ellipse":
            self.ensure_current_ellipse()
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
            self.ensure_current_drawed_contour()
            self.current_drawed_contour.add_points(
                [self.annotation_window.point_x, self.annotation_window.point_y],
            )

    def mouse_ellipse_axes_callback(self, event):
        if self.operation_mode == "ellipse":
            self.ensure_current_ellipse()
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

    def mouse_configure_wheel_circle_radius_callback(self, event):
        if self.operation_mode == "wheel_circle":
            if self.current_wheel_circle is None:
                return

            if self.current_wheel_circle.in_configuration:
                if os.name == "nt":
                    if event.delta > 0:
                        self.current_wheel_circle.radius_annotation_window += 1
                    elif event.delta < 0:
                        self.current_wheel_circle.radius_annotation_window -= 1
                if os.name == "posix":
                    if event.num == 4:
                        self.current_wheel_circle.radius_annotation_window += 1
                    elif event.num == 5:
                        self.current_wheel_circle.radius_annotation_window -= 1

                self.current_wheel_circle.radius_annotation_window = max(
                    self.current_wheel_circle.radius_annotation_window,
                    1,
                )
                self.last_wheel_circle_radius = (
                    self.current_wheel_circle.radius_annotation_window
                )
                self.current_wheel_circle.configure_circle_parameters()

    def start(self):
        self.initialize_windows()
        self.start_navigation_loop()
        self.start_annotation_loop()

    def start_navigation_loop(self):
        if self.navigation_loop_running:
            return

        self.navigation_loop_running = True
        self.update_navigation_window()

    def update_navigation_window(self):
        if not self.navigation_loop_running:
            return

        if self.image_manipulator is None:
            self.root_window.after(
                WINDOW_REFRESH_INTERVAL_MS, self.update_navigation_window
            )
            return

        if self.navigation_window is None:
            self.root_window.after(
                WINDOW_REFRESH_INTERVAL_MS, self.update_navigation_window
            )
            return

        self.navigation_window.set_image_manipulator(self.image_manipulator)

        if not self.navigation_window.annotation_mode:
            self.navigation_window.draw_ROI((0, 255, 0))
        elif (
            self.navigation_window.last_mouse_event_x is not None
            and self.navigation_window.last_mouse_event_y is not None
        ):
            self.navigation_window.image_manipulator.draw_rectangle_ROI(
                self.navigation_window.last_mouse_event_x,
                self.navigation_window.last_mouse_event_y,
                (255, 0, 0),
            )

        self.navigation_window.display_navigation_image(
            self.image_manipulator.annotation_image
        )

        self.root_window.after(
            WINDOW_REFRESH_INTERVAL_MS, self.update_navigation_window
        )

    def start_annotation_loop(self):
        if self.annotation_loop_running:
            return

        self.annotation_loop_running = True
        self.update_annotation_window()

    def update_annotation_window(self):
        if not self.annotation_loop_running:
            return

        if self.image_manipulator is None:
            self.root_window.after(
                WINDOW_REFRESH_INTERVAL_MS, self.update_annotation_window
            )
            return

        if self.annotation_window is None:
            self.root_window.after(
                WINDOW_REFRESH_INTERVAL_MS, self.update_annotation_window
            )
            return

        self.annotation_window.set_image_manipulator(self.image_manipulator)
        self.annotation_window.display_annotation_image()

        if self.navigation_window.annotation_mode:
            if self.operation_mode == "circle":
                self.ensure_current_circle()
                if self.current_circle is not None and self.current_circle.finished:
                    self.current_circle = None

            if self.operation_mode == "wheel_circle":
                self.ensure_current_wheel_circle()
                if (
                    self.current_wheel_circle is not None
                    and self.current_wheel_circle.in_configuration
                ):
                    self.current_wheel_circle.configure_circle_parameters()
                if (
                    self.current_wheel_circle is not None
                    and self.current_wheel_circle.finished
                ):
                    self.current_wheel_circle = None

            if self.operation_mode == "drawed_contour":
                self.ensure_current_drawed_contour()

            if self.operation_mode == "ellipse":
                self.ensure_current_ellipse()
                if (
                    self.current_ellipse is not None
                    and self.current_ellipse.in_configuration
                ):
                    self.current_ellipse.configure_ellipse_parameters()
                    self.current_ellipse.create_minor_axis_annotation_points()
                if self.current_ellipse is not None and self.current_ellipse.finished:
                    self.current_ellipse = None

            if self.operation_mode == "selection":
                if self.selector.valid:
                    self.select_contour()

        else:
            self.navigation_window.point_x = None
            self.navigation_window.point_y = None
            self.annotation_window.point_x = None
            self.annotation_window.point_y = None

        self.root_window.after(
            WINDOW_REFRESH_INTERVAL_MS, self.update_annotation_window
        )

    def run(self):
        self.root_window.mainloop()
