import threading
import time
import queue

def thread1_function(shared_queue):
    A = 1
    dict = {'function': 'thread1',
            'value': 0}
    while True:
        dict['value'] = A
        shared_queue.put(dict)
        time.sleep(0.1)
def thread2_function(shared_queue):
    A = 2
    dict = {'function': 'thread2',
            'value': 0}
    while True:
        dict['value'] = A
        shared_queue.put(dict)
        time.sleep(0.1)

if __name__ == "__main__":

    # Shared queue for communication between threads
    shared_queue = queue.Queue()

    # Create and start two separate threads
    thread1 = threading.Thread(target=thread1_function, args=(shared_queue,))
    thread2 = threading.Thread(target=thread2_function, args=(shared_queue,))
    thread1.start()
    thread2.start()

    while True:
        A = shared_queue.get()
        if A is not None:
            if A['function'] == 'thread1':
                print(f'A from thread1_function is {A}')
            if A['function'] == 'thread2':
                print(f'A from thread2_function is {A}')


