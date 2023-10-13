import tkinter as tk

from image_viewer_app.image_viewer_app import ImageViewerApp

root = tk.Tk()
app = ImageViewerApp(root)
app.start()
app.run()
