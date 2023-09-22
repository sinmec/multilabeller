import tkinter as tk
import threading
import time
import queue


class MyWindow(tk.Toplevel):
    def __init__(self, parent, title, shared_queue):
        super().__init__(parent)
        self.title(title)
        self.label = tk.Label(self, text=f"This is {title}")
        self.label.pack()

        # Allow both width and height resizing
        self.resizable(True, True)

        # Bind the mouse enter event
        self.label.bind("<Enter>", self.on_hover)

        # Bind the spacebar key event
        self.bind("<Key>", self.on_key_press)

        # Shared queue for communication
        self.shared_queue = shared_queue

    def on_hover(self, event):
        # Get the window title and print the message
        window_title = self.title()
        print(f"I am at window {window_title}")

    def on_key_press(self, event):
        # Check if the spacebar key was pressed and print the message only in Window 1
        if event.keysym == "space" and self.title() == "Window 1":
            print("I hit spacebar")
            # Put a message in the shared queue when spacebar is pressed in Window 1
            self.shared_queue.put("Spacebar pressed in Window 1")


def thread1_function(shared_queue):
    # Simulate some work for Thread 1
    time.sleep(3)
    window1 = MyWindow(root, "Window 1", shared_queue)


def thread2_function(shared_queue):
    # Simulate some work for Thread 2
    time.sleep(2)
    window2 = MyWindow(root, "Window 2", shared_queue)

    # Counter for spacebar presses
    counter = 0

    while True:
        try:
            # Try to get messages from the shared queue
            message = shared_queue.get(block=False)
            if message == "Spacebar pressed in Window 1":
                counter += 1
                print(f"Spacebar pressed {counter} times in Window 1")
        except queue.Empty:
            # Queue is empty, sleep briefly to avoid busy-waiting
            time.sleep(0.1)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Main Window")

    # Shared queue for communication between threads
    shared_queue = queue.Queue()

    # Create and start two separate threads
    thread1 = threading.Thread(target=thread1_function, args=(shared_queue,))
    thread2 = threading.Thread(target=thread2_function, args=(shared_queue,))
    thread1.start()
    thread2.start()

    root.mainloop()
