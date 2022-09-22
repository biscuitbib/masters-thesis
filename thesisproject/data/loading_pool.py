from threading import Lock, Thread
from queue import Queue
from time import sleep

def _load_func(load_queue, results_queue):
    while True:
        to_load = load_queue.get()
        try:
            to_load.load()
        finally:
            print("added to loading pool")
            results_queue.put(to_load)
            load_queue.task_done()

def _gather_loaded(output_queue, put_function):
    while True:
        # Wait for studies in the output queue
        image_pair = output_queue.get(block=True)
        put_function(image_pair)
        output_queue.task_done()
        print("thread putting into loaded queue")

class LoadingPool:
    """
    Implements a multithreading loading queue
    """
    def __init__(self,
                 put_function=lambda _elem: None,
                 n_threads=16,
                 max_queue_size=50):
        # Setup load thread pool
        self._load_queue = Queue(maxsize=max_queue_size)
        self._output_queue = Queue(maxsize=max_queue_size)
        self.thread_lock = Lock()

        args = (self._load_queue, self._output_queue)
        self.pool = []
        for i in range(n_threads):
            p = Thread(target=_load_func, args=args, daemon=True, name=f"load_{i}")
            p.start()
            self.pool.append(p)

        # Prepare gathering thread
        self._put_function = put_function
        self.gather_loaded_thread = Thread(
            target=_gather_loaded,
            args=(self._output_queue,
                  self._put_function),
            daemon=True,
            name="gather")
        self.gather_loaded_thread.start()

    @property
    def qsize(self):
        """ Returns the qsize of the load queue """
        return self._load_queue.qsize

    @property
    def maxsize(self):
        """ Returns the maxsize of the load queue """
        return self._load_queue.maxsize

    def join(self):
        """ Join on all queues """
        self._load_queue.join()
        self._output_queue.join()

    def add_image_to_load_queue(self, image_pair):
        if self.qsize() == self.maxsize:
            "Sleep until empty"
            print("loading pool queue full")
            while self.qsize() > 1:
                sleep(1)
        self._load_queue.put(image_pair)
