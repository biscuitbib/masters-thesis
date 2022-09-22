from threading import Lock, Thread
from queue import Queue
from time import sleep

def _load_func(load_queue, results_queue):
    while True:
        to_load = load_queue.get()
        try:
            to_load.load()
        finally:
            results_queue.put(to_load)
            load_queue.task_done()

def _gather_loaded(output_queue, put_function):
    while True:
        # Wait for studies in the output queue
        image_pair = output_queue.get(block=True)
        put_function(image_pair)
        output_queue.task_done()

class LoadingPool:
    """
    Implements a multithreading loading queue
    """
    def __init__(self,
                 n_threads=5,
                 max_queue_size=50,):
        # Setup load thread pool
        self._load_queue = Queue(maxsize=max_queue_size)
        self._output_queue = Queue(maxsize=max_queue_size)
        self.thread_lock = Lock()

        args = (self._load_queue, self._output_queue)
        self.pool = []
        for _ in range(n_threads):
            p = Thread(target=_load_func, args=args, daemon=True)
            p.start()
            self.pool.append(p)

        # Prepare gathering thread
        self._put_function = lambda: None
        self.gather_loaded_thread = Thread(target=_gather_loaded,
                                           args=(self._output_queue,
                                                 self._put_function),
                                           daemon=True)
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
            self.logger.warn("Loading queue seems about to block! "
                             "(max_size={}, current={}). "
                             "Sleeping until loading queue is empty "
                             "again.".format(self.maxsize,
                                             self.qsize()))
            while self.qsize() > 1:
                sleep(1)
        self._load_queue.put(image_pair)

    def register_put_function(self, load_put_function):
        with self.thread_lock:
            self._put_function = load_put_function

    def de_register_put_function(self):
        with self.thread_lock:
            del self._put_function
