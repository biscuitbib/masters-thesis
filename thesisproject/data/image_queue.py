import numpy as np
from queue import Empty, Queue
from contextlib import contextmanager

from thesisproject.data.image_pair import ImagePair
from thesisproject.data.loading_pool import LoadingPool

class ImageQueue():
    def __init__(self, dataset, queue_length=16, max_access=10):
        self.dataset = dataset

        self.queue_length = min(queue_length, len(self.dataset))
        self.max_access = max_access

        self.non_loaded = Queue(maxsize=len(dataset))
        self.loaded = Queue(maxsize=queue_length)

        # Fill non-loaded queue in random order
        inds = np.arange(len(dataset))
        np.random.shuffle(inds)
        for i in inds:
            self.non_loaded.put(self.dataset.images[i])

        self.loading_pool = LoadingPool()
        self.loading_pool.register_put_function(self._add_to_loaded)

        self._load_queue_full()

    def _add_to_load_queue(self):
        image = self.non_loaded.get_nowait()
        self.loading_pool.add_image_to_load_queue(image)

    def _load_queue_full(self):
        for _ in range(self.queue_length):
            self._add_to_load_queue()

    def __len__(self):
        return len(self.dataset)

    @contextmanager
    def get_random_image(self):
        timeout_s = 5
        try:
            image_pair, n_access = self.loaded.get(timeout=timeout_s)
        except Empty:
            raise StopIteration

        try:
            yield image_pair
        finally:
            self._release_image(image_pair, n_access)

    def _release_image(self, image_pair: ImagePair, n_access):
        """
        Adds image pair to back of queue if not accessed more than max access amount, else adds new image pair to queue.
        """
        if n_access >= self.max_access:
            image_pair.unload()
            self.non_loaded.put(image_pair)
            self._add_to_loaded()
        else:
            self.loaded.put((image_pair, n_access + 1))
