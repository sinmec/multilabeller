import os
import tkinter as tk

import cv2
from PIL import Image, ImageTk


class Window(tk.Toplevel):
    def __init__(self, parent, title, config, shared_queue, contour_collection):
        super().__init__(parent)

        self.original_annotation_image = None
        self.title_string = title
        self.title(title)
        self.label = tk.Label(self, text=f"{title}")
        self.label.pack()

        self.shared_queue = shared_queue
        self.canvas = None
        self.loop = None
        self.config = config

        self.contour_collection = contour_collection

        self.image_manipulator = None

        self.mouse_y = 0
        self.mouse_x = 0

        self.last_mouse_event_x = None
        self.last_mouse_event_y = None

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

    def display_navigation_image(self, image):
        if self.image_manipulator is None:
            print(
                f"Warning: No image manipulator in window {self.title_string} was defined."
            )
            return

        self.draw_navigation_window_objects()

        image = Image.fromarray(self.image_manipulator.navigation_image)
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.photo = photo

    def draw_annotation_window_objects(self):
        image_copy = self.image_manipulator.annotation_image_buffer.copy()

        for annotation_object in self.contour_collection.items:
            if annotation_object.__class__.__name__ == "Ellipse":
                if annotation_object.in_configuration:
                    cv2.drawContours(
                        image_copy,
                        [annotation_object.ellipse_contour],
                        -1,
                        annotation_object.color,
                        1,
                    )

        for annotation_object in self.contour_collection.items:
            if annotation_object.finished:
                annotation_object.translate_from_navigation_to_annotation_windows(
                    self.image_manipulator
                )
                annotation_object.to_cv2_contour()

        for annotation_object in self.contour_collection.items:
            if annotation_object is None:
                continue
            if not annotation_object.valid:
                continue

            if annotation_object.in_progress:
                if len(annotation_object.points_annotation_window) > 0:
                    for point in annotation_object.points_annotation_window:
                        if point is not None:
                            circle_radius = 4
                            circle_color = (0, 255, 0)
                            circle_thickness = -1

                            point_x, point_y = point

                            cv2.circle(
                                image_copy,
                                (point_x, point_y),
                                circle_radius,
                                circle_color,
                                circle_thickness,
                            )

            elif annotation_object.finished:

                cv2.drawContours(
                    image_copy,
                    [annotation_object.annotation_window_contour],
                    -1,
                    annotation_object.color,
                    annotation_object.thickness,
                )

        self.image_manipulator.annotation_image = image_copy

    def draw_navigation_window_objects(self):
        image_copy = self.image_manipulator.navigation_image_buffer.copy()

        for annotation_object in self.contour_collection.items:
            if annotation_object is None:
                continue
            if not annotation_object.valid:
                continue
            if annotation_object.in_progress:
                continue

            annotation_object.translate_from_navigation_to_annotation_windows(
                self.image_manipulator
            )
            annotation_object.to_cv2_contour()

            cv2.drawContours(
                image_copy,
                [annotation_object.navigation_window_contour],
                -1,
                annotation_object.color,
                annotation_object.thickness,
            )

        self.image_manipulator.navigation_image = image_copy

    def display_annotation_image(self):
        if self.image_manipulator is None:
            print(
                f"Warning: No image manipulator in window {self.title_string} was defined."
            )
            return

        self.draw_annotation_window_objects()

        image = Image.fromarray(self.image_manipulator.annotation_image)
        photo = ImageTk.PhotoImage(image=image)

        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.photo = photo

    def get_mouse_position(self, event):
        self.mouse_x = event.x
        self.mouse_y = event.y

    def draw_ROI(self, color):
        self.image_manipulator.update_rectangle_size()
        self.image_manipulator.draw_rectangle_ROI(self.mouse_x, self.mouse_y, color)
        self.image_manipulator.update_annotation_image()

        self.last_mouse_event_x = self.mouse_x
        self.last_mouse_event_y = self.mouse_y

    def modify_ROI_zoom(self, event):
        if os.name == "nt":
            if event.delta > 0:
                self.image_manipulator.rectangle_ROI_zoom_count += self.config[
                    "mouse_wheel"
                ]["step_sensibility"]
            elif event.delta < 0:
                self.image_manipulator.rectangle_ROI_zoom_count -= self.config[
                    "mouse_wheel"
                ]["step_sensibility"]
        if os.name == "posix":
            if event.num == 4:
                self.image_manipulator.rectangle_ROI_zoom_count += self.config[
                    "mouse_wheel"
                ]["step_sensibility"]
            elif event.num == 5:
                self.image_manipulator.rectangle_ROI_zoom_count -= self.config[
                    "mouse_wheel"
                ]["step_sensibility"]

    def lock_annotation_image(self, event):
        self.annotation_mode = not self.annotation_mode
        self.original_annotation_image = self.image_manipulator.annotation_image.copy()

    def store_annotation_point(self, event):
        self.point_x = self.mouse_x
        self.point_y = self.mouse_y
