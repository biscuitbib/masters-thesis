import numpy as np
from queue import Empty, Queue
from contextlib import contextmanager
from time import sleep


from thesisproject.data.image_pair import ImagePair
from thesisproject.data.loading_pool import LoadingPool

class ImageQueue():
    def __init__(self, dataset, queue_length=16, max_access=50):
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

        self.loading_pool = LoadingPool(put_function=self._add_image_to_load_queue)
        #self.loading_pool.register_put_function(self._add_image_to_load_queue)
        
        # Increment counters to random off-set points for the first images
        self.max_offset = int(self.max_access * 0.75)
        self.n_offset = self.queue_length

        self._load_queue_full()

    def _add_image_to_loading_pool(self):
        image = self.non_loaded.get_nowait()
        self.loading_pool.add_image_to_load_queue(image)

    def _add_image_to_load_queue(self, image):
        if self.n_offset >= 0:
            offset = np.random.randint(0, self.max_offset)
            self.n_offset -= 1
        else:
            offset = 0
            
        self.loaded.put((image, offset))

    def _load_queue_full(self):
        for _ in range(self.queue_length):
            self._add_image_to_loading_pool()
        self.loading_pool.join()

    def __len__(self):
        return len(self.dataset)

    @contextmanager
    def get_random_image(self):
        print("loaded size: ", self.loaded.qsize())
        if self.loaded.qsize() < self.queue_length // 2:
            sleep(2)
        timeout_s = 5
        try:
            image_pair, n_access = self.loaded.get(timeout=timeout_s)
        except Empty:
            raise Empty
        try:
            yield image_pair
        finally:
            self._release_image(image_pair, n_access)

    def _release_image(self, image_pair, n_access):
        """
        Adds image pair to back of queue if not accessed more than max access amount, else adds new image pair to queue.
        """
        if n_access >= self.max_access:
            image_pair.unload()
            self.non_loaded.put(image_pair)
            self._add_image_to_loading_pool()
        else:
            self.loaded.put((image_pair, n_access + 1))
