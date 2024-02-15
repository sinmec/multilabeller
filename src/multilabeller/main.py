import tkinter as tk

from image_viewer_app.image_viewer_app import ImageViewerApp
from src.contour_collection import ContourCollection

root = tk.Tk()
contour_collection = ContourCollection()
app = ImageViewerApp(root, contour_collection)
app.start()
app.run()
