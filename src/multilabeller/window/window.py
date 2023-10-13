import os
import tkinter as tk

from PIL import Image, ImageTk


class Window(tk.Toplevel):
    def __init__(self, parent, title, config, shared_queue):
        super().__init__(parent)

        self.title_string = title
        self.title(title)
        self.label = tk.Label(self, text=f"{title}")
        self.label.pack()

        self.shared_queue = shared_queue
        self.canvas = None
        self.loop = None
        self.config = config

        self.image_manipulator = None

        self.mouse_y = 0
        self.mouse_x = 0

        self.point_x = None
        self.point_y = None

        self.annotation_mode = False

    def run(self):
        if self.loop is None:
            print(f"Warning: No run action defined in window {self.title_string}.")
            return
        self.loop()

    def set_image_manipulator(self, image_manipulator):
        self.image_manipulator = image_manipulator

    # TODO: Think on a solution to have only a single 'display_image'
    def display_image(self):
        if self.image_manipulator is None:
            print(
                f"Warning: No image manipulator in window {self.title_string} was defined."
            )
            return
        image = Image.fromarray(self.image_manipulator.image)
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.photo = photo

    # TODO: Think on a solution to have only a single 'display_image';
    #       We have an unnecessary 'display_zoomed_image' here!
    def display_zoomed_image(self):
        if self.image_manipulator is None:
            print(
                f"Warning: No image manipulator in window {self.title_string} was defined."
            )
            return
        image = Image.fromarray(self.image_manipulator.zoomed_image)
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.photo = photo

    def get_mouse_position(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

    def draw_ROI(self):
        self.image_manipulator.update_rectangle_size()
        self.image_manipulator.draw_rectangle_ROI(self.mouse_x, self.mouse_y)
        self.image_manipulator.update_zoomed_image()

    def modify_ROI_zoom(self, event):
        if os.name == "nt":
            if event.delta > 0:
                self.image_manipulator.rectangle_ROI_zoom_count += (
                    1  # TODO: Add step as option
                )
            elif event.delta < 0:
                self.image_manipulator.rectangle_ROI_zoom_count -= (
                    1  # TODO: Add step as option
                )
        if os.name == "posix":
            if event.num == 4:
                self.image_manipulator.rectangle_ROI_zoom_count += 1
            elif event.num == 5:
                self.image_manipulator.rectangle_ROI_zoom_count -= 1

    def lock_image(self, event):
        self.annotation_mode = not self.annotation_mode  # TODO: Think of a better name

    def store_annotation_point(self, event):
        self.point_x = self.mouse_x
        self.point_y = self.mouse_y
