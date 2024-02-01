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
from src.multilabeller.select import Select
from src.multilabeller.SAM.sam import SegmentAnything
from src.multilabeller.elipse import Elipse

if os.name == "nt":
    os_option = "windows"
if os.name == "posix":
    os_option = "linux"


class ImageViewerApp:
    def __init__(self, root):
        self.k = 0
        self.ellipse_confirm = False
        self.elipse_id = -1
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
        self.elipse_mode = False
        self.navigation_window = None
        self.annotation_window = None
        self.contour_id = -1
        self.circle_id = -1
        self.id = 0
        self.read_config_file()
        self.initialize_main_window()
        self.initialize_queue()
        self.current_contour = None
        self.select = Select()
        self.selected_contours = []
        self.segmentation = None
        self.segmentation_id = 0
        self.contours_to_recreate = []
        self.duplication = False

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

            window.status_bar = tk.Label(window, text=self.config["tk_label"]["text"],
                                         bd=1, relief=tk.SUNKEN, anchor=tk.W)
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

            self.annotation_window.bind(self.config["left_mouse_click"][os_option],
                                        self.annotation_window.store_annotation_point)

            self.annotation_window.bind(self.config["left_mouse_click"][os_option],
                                        self.mouse_circle_callback, add="+")

            self.annotation_window.bind(self.config["left_mouse_click"][os_option],
                                        self.mouse_contour_callback, add="+")

            self.annotation_window.bind(self.config["left_mouse_click"][os_option],
                                        self.mouse_select_callback, add="+")

            self.annotation_window.bind(self.config["left_mouse_click"][os_option],
                                        self.mouse_ellipse_callback, add="+")

            self.annotation_window.bind(self.config["left_mouse_click"][os_option],
                                        self.draw_point_mouse_callback, add="+")

            self.annotation_window.bind('<Key>', self.trigger)

            if os_option == "linux":
                self.annotation_window.bind(
                    self.config["mouse_wheel"][os_option]["bind1"],
                    self.y_ellipse_axis_callback, add="+",
                )
                self.annotation_window.bind(
                    self.config["mouse_wheel"][os_option]["bind2"],
                    self.y_ellipse_axis_callback, add="+",
                )

            elif os_option == "windows":
                self.annotation_window.bind(
                    self.config["mouse_wheel"][os_option],
                    self.y_ellipse_axis_callback, add="+",
                )

            while True:
                self.navigation_window.display_image(self.image_manipulator.zoomed_image)
                self.navigation_window.canvas.bind(
                    self.config["mouse_motion"][os_option],
                    self.navigation_window.get_mouse_position,
                )

                if os_option == "linux":
                    self.navigation_window.canvas.bind(
                        self.config["mouse_wheel"][os_option]["bind1"],
                        self.navigation_window.modify_ROI_zoom, add="+",
                    )
                    self.navigation_window.canvas.bind(
                        self.config["mouse_wheel"][os_option]["bind2"],
                        self.navigation_window.modify_ROI_zoom, add="+",
                    )

                elif os_option == "windows":
                    self.navigation_window.canvas.bind(
                        self.config["mouse_wheel"][os_option],
                        self.navigation_window.modify_ROI_zoom,
                    )

                self.navigation_window.bind("<F9>", self.navigation_window.lock_image, add="+")
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

                if self.contours_list:
                    self.recreate_contours_original_image()
                    self.recreate_contours_zoomed_image()

                if self.navigation_window.annotation_mode:
                    if self.contours_list:
                        self.recreate_contours_original_image()

                    if self.circle_mode:  # CIRCLE
                        if self.id != self.circle_id:
                            self.circle_id = self.id
                            self.current_circle = Circle(self.id,
                                                         (self.image_manipulator.rectangle_ROI_zoom,
                                                          self.image_manipulator.zoomed_image_coordinates)
                                                         )

                        self.delete_points(self.current_circle.i)

                        self.draw_circle(
                            self.current_circle,
                            self.image_manipulator.zoomed_image,
                            self.image_manipulator.image
                        )

                    if self.contour_mode:  # CONTOUR
                        if self.id != self.contour_id:
                            self.contour_id = self.id
                            self.current_contour = Contour(self.id,
                                                           (self.image_manipulator.rectangle_ROI_zoom,
                                                            self.image_manipulator.zoomed_image_coordinates)
                                                           )

                        if self.current_contour.i == 0:
                            self.clean_image = self.image_manipulator.zoomed_image.copy()
                            self.clean_manipulator_image = self.image_manipulator.image.copy()

                        if self.contour_confirm:
                            self.contours_list.append(self.current_contour)
                            self.image_manipulator.zoomed_image = self.clean_image
                            self.image_manipulator.image = self.clean_manipulator_image

                            self.create_contour_lines(self.current_contour,
                                                      self.image_manipulator.zoomed_image,
                                                      self.current_contour.points)

                            self.create_contour_lines(self.current_contour,
                                                      self.image_manipulator.image,
                                                      self.current_contour.translated_points)

                            # print(self.contours_list)
                            self.id = self.id + 1
                            self.contour_confirm = not self.contour_confirm

                    if self.elipse_mode:  # ELLIPSE
                        if self.id != self.elipse_id:
                            self.id = self.elipse_id
                            self.current_elipse = Elipse(self.elipse_id,
                                                         (self.image_manipulator.rectangle_ROI_zoom,
                                                          self.image_manipulator.zoomed_image_coordinates))

                        if self.current_elipse.i == 2:
                            self.image_manipulator.zoomed_image = self.clean_image.copy()
                            self.image_manipulator.image = self.clean_manipulator_image.copy()
                            self.create_ellipse(self.current_elipse)

                        if self.contour_confirm:
                            self.contours_list.append(self.current_elipse)
                            self.id += 1
                            self.contour_confirm = not self.contour_confirm

                    if self.select_mode:  # SELECT
                        if self.select.i != 0:

                            if self.annotation_window.point_x and self.annotation_window.point_y:
                                self.select_contour(self.annotation_window.point_x, self.annotation_window.point_y)

                            self.select.i = 0

                else:
                    self.navigation_window.point_x = None
                    self.navigation_window.point_y = None
                    self.annotation_window.point_x = None
                    self.annotation_window.point_y = None

                time.sleep(0.01)

        self.annotation_window.loop = run_annotation_window

    def ellipse_confirm_callback(self, event):
        self.ellipse_confirm = True

    def draw_point_mouse_callback(self, event):
        if self.elipse_mode:
            if self.current_elipse.i < 2:

                if self.current_elipse.i == 0:
                    self.clean_image = self.image_manipulator.zoomed_image.copy()
                    self.clean_manipulator_image = self.image_manipulator.image.copy()
                    print('salvou a imagem')
                self.draw_points_functions()

                if self.current_elipse.i == 1:
                    self.current_elipse.create_initial_ellipse()
                self.current_elipse.increase_ellipse_counter()

        if self.contour_mode:
            if self.current_contour.i != 0:
                self.draw_points_functions()
        if self.circle_mode:
            if self.current_circle.i == 1:
                self.draw_points_functions()

    def draw_points_functions(self):
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
            self.annotation_window.point_x, self.annotation_window.point_y
        )

    def save_image(self, event):
        self.zoomed_image_original = self.image_manipulator.zoomed_image.copy()
        self.image_original = self.image_manipulator.image.copy()

    def mouse_select_callback(self, event):
        if self.select_mode:
            self.select.point_check()

    def select_contour(self, point_x, point_y):

        for obj in self.contours_list:

            if obj.__class__.__name__ == 'Elipse':
                x_ang = ((point_x - obj.x_axis) * np.cos(obj.angle)) - ((point_y - obj.y_axis) * np.sin(obj.angle))
                y_ang = ((point_x - obj.x_axis) * np.sin(obj.angle)) - ((point_y - obj.y_axis) * np.cos(obj.angle))

                ellipse_equation = (
                        (pow(((point_x - obj.center[0]) * np.cos(obj.angle)) +
                             ((point_y - obj.center[1]) * np.sin(obj.angle)), 2)

                         / pow((max((obj.x_axis), (obj.y_axis))), 2))

                        +

                        (pow(((point_x - obj.center[0]) * np.sin(obj.angle)) -
                             ((point_y - obj.center[1]) * np.cos(obj.angle)), 2)

                         / pow((min((obj.x_axis), (obj.y_axis))), 2))
                )
                print(f'{ellipse_equation}\n')

                if ellipse_equation <= 1:
                    if obj.color == (255, 0, 0):
                        obj.color = (0, 255, 0)  # verde = selecionado
                        self.selected_contours.append(obj)

                    elif obj.color == (0, 255, 0):  # vermelho = deselecionado
                        obj.color = (255, 0, 0)
                        self.selected_contours.remove(obj)

                    self.create_ellipse(obj)

            if obj.__class__.__name__ == 'Circle':

                dist_to_center = np.sqrt(pow((point_x - obj.center[0]), 2) +
                                         pow((point_y - obj.center[1]), 2))

                if dist_to_center <= obj.radius:
                    if obj.color == (255, 0, 0):
                        obj.color = (0, 255, 0)  # verde = selecionado
                        self.selected_contours.append(obj)
                        # print(self.selected_contours)
                    elif obj.color == (0, 255, 0):  # vermelho = deselecionado
                        obj.color = (255, 0, 0)
                        self.selected_contours.remove(obj)
                        # print(self.selected_contours)

                    self.update_circle(obj,
                                       self.image_manipulator.zoomed_image,
                                       self.image_manipulator.image)

            if obj.__class__.__name__ == 'Contour':
                point = (point_x, point_y)
                cnt_points = np.array([obj.points], np.int32)
                dist = cv2.pointPolygonTest(cnt_points, point, measureDist=True)

                if dist >= 0:
                    if obj.color == (255, 0, 0):
                        obj.color = (0, 255, 0)  # verde = selecionado
                        self.selected_contours.append(obj)
                    elif obj.color == (0, 255, 0):  # vermelho = deselecionado

                        obj.color = (255, 0, 0)
                        self.selected_contours.remove(obj)

                    self.create_contour_lines(obj,
                                              self.image_manipulator.zoomed_image,
                                              obj.points)

                    self.create_contour_lines(obj,
                                              self.image_manipulator.image,
                                              obj.translated_points)

    def create_ellipse_original_image(self, obj):
        translated_axes_lenght = ((int(obj.translated_x_axis / 2)),
                                  int(
                                      (obj.y_axis / self.image_manipulator.rectangle_ROI_zoom) *
                                      self.image_manipulator.image_original_width / (
                                          self.config["image_viewer"]["width"])
                                  ))

        # manipulator image
        cv2.ellipse(self.image_manipulator.image, obj.translated_center, translated_axes_lenght,
                    obj.translated_angle, 0, 360, obj.color, obj.thickness)

    def create_ellipse(self, obj):
        axes_lenght = ((int(obj.x_axis / 2)), obj.y_axis)

        translated_axes_lenght = ((int(obj.translated_x_axis / 2)),
                                  int(
                                      (obj.y_axis / self.image_manipulator.rectangle_ROI_zoom) *
                                      self.image_manipulator.image_original_width / (
                                          self.config["image_viewer"]["width"])
                                  ))

        # zoomed image
        cv2.ellipse(self.image_manipulator.zoomed_image, obj.center, axes_lenght,
                    obj.angle, 0, 360, obj.color, obj.thickness)

        # manipulator image
        cv2.ellipse(self.image_manipulator.image, obj.translated_center, translated_axes_lenght,
                    obj.translated_angle, 0, 360, obj.color, obj.thickness)

    def y_ellipse_axis_callback(self, event):
        if self.elipse_mode and self.current_elipse.i == 2:
            if os.name == "nt":
                if event.delta > 0:
                    self.current_elipse.define_y_axis(1)
                elif event.delta < 0:
                    if self.current_elipse.y_axis > 0:
                        self.current_elipse.define_y_axis(-1)
            if os.name == "posix":
                if event.num == 4:
                    self.current_elipse.define_y_axis(1)
                elif event.num == 5:
                    if self.current_elipse.y_axis > 0:
                        self.current_elipse.define_y_axis(-1)

    def trigger(self, event):
        if event.char == 'c':
            self.contour_mode = not self.contour_mode
            self.circle_mode = False
            self.select_mode = False
            self.elipse_mode = False
            if self.contour_mode:
                print('Contour mode on')
                self.id = 0
            else:
                print('Contour mode off')
        elif event.char == 'b':
            self.circle_mode = not self.circle_mode
            self.contour_mode = False
            self.select_mode = False
            self.elipse_mode = False
            if self.circle_mode:
                print('Circle mode on')
                self.id = 0
            else:
                print('Circle mode off')
        elif event.char == 'v':
            self.select_mode = not self.select_mode
            self.contour_mode = False
            self.circle_mode = False
            self.elipse_mode = False
            if self.select_mode:
                print('Select mode on')
            else:
                print('Select mode off')
        elif event.char == ' ':  # spacebar
            self.contour_confirm = not self.contour_confirm
            print('Contour saved')
        elif event.keysym == 'BackSpace':
            self.delete_selected_contours()
        elif event.keysym == 's' and self.segmentation_id == 0:
            self.auto_segmentation()
            self.segmentation_id += 1
        elif event.keysym == 'e':
            self.elipse_mode = not self.elipse_mode
            if self.elipse_mode:
                print('Elipse mode on')
                self.id = 0
            else:
                print('Elipse mode off')
            self.contour_mode = False
            self.circle_mode = False
            self.select_mode = False

    def auto_segmentation(self):
        print('started Auto Segmentation')
        self.segmentation = SegmentAnything(self.zoomed_image_original.copy())

        for contour in self.segmentation.contours:
            for point in contour.points:
                print(point)
                if point:
                    translated_point = self.image_manipulator.translate_points(point[0], point[1])
                    contour.add_contour_points(None, translated_point)

            self.contours_list.append(contour)

        for contour in self.segmentation.contours:
            self.create_contour_lines(contour, self.image_manipulator.zoomed_image, contour.points)
            self.create_contour_lines(contour, self.image_manipulator.image, contour.translated_points)

    def mouse_ellipse_callback(self, event):
        if self.elipse_mode:
            (
                self.navigation_window.point_x,
                self.navigation_window.point_y,
            ) = self.image_manipulator.translate_points(
                self.annotation_window.point_x, self.annotation_window.point_y
            )

            self.current_elipse.add_elipse_points(self.annotation_window.point_x, self.annotation_window.point_y,
                                                  self.navigation_window.point_x, self.navigation_window.point_y
                                                  )

    def mouse_circle_callback(self, event):
        if self.circle_mode:
            (
                self.navigation_window.point_x,
                self.navigation_window.point_y,
            ) = self.image_manipulator.translate_points(
                self.annotation_window.point_x, self.annotation_window.point_y
            )
            # TODO: Need to fix this gambiarra above

            self.current_circle.add_circle_points(self.annotation_window.point_x, self.annotation_window.point_y,
                                                  self.navigation_window.point_x, self.navigation_window.point_y
                                                  )

    def translate_point_different_ROIs(self, point, old_image_settings, new_image_settings):

        # image_settings = [ROI, image_coordinates]

        # this function exists to solve the problem of the display of the contours on the zoomed image
        # after unlock the image
        # TODO: I THINK THE ONLY PROBLEM NOW IS IN THIS FUNCTION

        old_ROI = old_image_settings[0]
        new_ROI = new_image_settings[0]
        delta_ROI = new_ROI - old_ROI

        old_x1_rec, old_x2_rec, old_y1_rec, old_y2_rec = old_image_settings[1]
        new_x1_rec, new_x2_rec, new_y1_rec, new_y2_rec = new_image_settings[1]

        delta_x_rec = (new_x2_rec - new_x1_rec) - (old_x2_rec - old_x1_rec)
        delta_y_rec = (new_y2_rec - new_y1_rec) - (old_y2_rec - old_y1_rec)

        point_x1 = point[0]
        point_y1 = point[1]

        new_point = (delta_ROI * (point_x1 - delta_x_rec), (delta_ROI * (point_y1 - delta_y_rec)))

        return new_point

    def delete_selected_contours(self):
        for selected in self.selected_contours:
            if selected in self.contours_list:
                self.contours_list.remove(selected)

        self.recreate_contours()

    def recreate_contours_zoomed_image(self):
        for obj in self.contours_list:
            if obj.__class__.__name__ == "Circle":
                # todo: pegar pontos de quando o objeto foi criado e traduzir para a nova ROI da zoomed image
                point1 = obj.points[0]
                point2 = obj.points[1]

                new_point_1 = self.translate_point_different_ROIs(point1, (obj.ROI[0], obj.ROI[1]),
                                                                     (self.image_manipulator.rectangle_ROI_zoom,
                                                                      self.image_manipulator.zoomed_image_coordinates))

                new_point_2 = self.translate_point_different_ROIs(point2, (obj.ROI[0], obj.ROI[1]),
                                                                     (self.image_manipulator.rectangle_ROI_zoom,
                                                                      self.image_manipulator.zoomed_image_coordinates))


                circulo = Circle(self.k, 0)

                self.k += 1

                circulo.update_circle_points_zoomed_image(new_point_1, new_point_2)

                self.update_circle_zoomed_image(circulo, self.image_manipulator.zoomed_image)

            elif obj.__class__.__name__ == "Contour":
                # todo: pegar pontos de quando o objeto foi criado e traduzir para a nova ROI da zoomed image
                a = 2
            elif obj.__class__.__name__ == "Elipse":
                # todo: pegar pontos de quando o objeto foi criado e traduzir para a nova ROI da zoomed image
                a = 2

    def recreate_contours_original_image(self):
        # self.image_manipulator.image = self.image_original.copy()

        for obj in self.contours_list:
            if obj.__class__.__name__ == "Circle":
                self.update_circle_original_image(obj, self.image_manipulator.image)
            elif obj.__class__.__name__ == "Contour":
                self.create_contour_lines(obj, self.image_manipulator.image, obj.translated_points)
            elif obj.__class__.__name__ == "Elipse":
                self.create_ellipse_original_image(obj)

    def recreate_contours(self):
        self.image_manipulator.zoomed_image = self.zoomed_image_original.copy()
        self.image_manipulator.image = self.image_original.copy()

        for obj in self.contours_list:
            if obj.__class__.__name__ == "Circle":
                self.update_circle(obj, self.image_manipulator.zoomed_image,
                                   self.image_manipulator.image)
            elif obj.__class__.__name__ == "Contour":
                self.create_contour_lines(obj, self.image_manipulator.zoomed_image, obj.points)
                self.create_contour_lines(obj, self.image_manipulator.image, obj.translated_points)
            elif obj.__class__.__name__ == "Elipse":
                self.create_ellipse(obj)

    def mouse_contour_callback(self, event):
        if self.contour_mode:
            (
                self.navigation_window.point_x,
                self.navigation_window.point_y,
            ) = self.image_manipulator.translate_points(
                self.annotation_window.point_x, self.annotation_window.point_y
            )

            # TODO: Need to fix this gambiarra above

            self.current_contour.add_contour_points([self.annotation_window.point_x, self.annotation_window.point_y],
                                                    [self.navigation_window.point_x, self.navigation_window.point_y])

    def create_contour_lines(self, obj, image, points):
        for count, point in enumerate(points):
            cv2.line(image, points[count], points[count - 1],
                     obj.color, obj.thickness)

    def delete_points(self, i):
        if i == 0:
            self.clean_image = self.image_manipulator.zoomed_image.copy()
            self.clean_manipulator_image = self.image_manipulator.image.copy()
        elif i == 2:
            self.image_manipulator.zoomed_image = self.clean_image
            self.image_manipulator.image = self.clean_manipulator_image

    def draw_circle(self, obj, image_annotation, image_manipulator):
        if obj.i == 2:
            cv2.circle(
                image_annotation, obj.center, obj.radius, obj.color, obj.thickness
            )

            cv2.circle(
                image_manipulator, obj.translated_center, obj.translated_circle_radius,
                obj.color, obj.thickness - 1
            )

            obj.i = 0

            self.add_circle_to_list(obj)

    def update_circle(self, obj, image_annotation, image_manipulator):
        cv2.circle(
            image_annotation, obj.center, obj.radius, obj.color, obj.thickness
        )

        cv2.circle(
            image_manipulator, obj.translated_center, obj.translated_circle_radius,
            obj.color, obj.thickness - 1
        )

    def update_circle_zoomed_image(self, obj, image_annotation):
        cv2.circle(
            image_annotation, obj.center, obj.radius, obj.color, obj.thickness
        )

    def update_circle_original_image(self, obj, image):
        cv2.circle(
            image, obj.translated_center, obj.translated_circle_radius,
            obj.color, obj.thickness - 1
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
