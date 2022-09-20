import numpy as np
from torch.utils.data import IterableDataset
from queue import Empty, Queue

from thesisproject.data.image_pair import ImagePair

class ImageQueue(IterableDataset):
    def __init__(self, dataset, queue_length=16, max_access=10, max_iterations=10000):
        self.dataset = dataset

        self.queue_length = min(queue_length, len(self.dataset))
        self.max_access = max_access
        self._len = len(dataset)

        self.non_loaded = Queue(maxsize=len(dataset))
        self.loaded = Queue(maxsize=queue_length)

        # Fill non-loaded queue in random order
        inds = np.arange(len(dataset))
        np.random.shuffle(inds)
        for i in inds:
            self.non_loaded.put(self.dataset.images[i])

        self._load_queue_full()


    def _add_to_loaded(self):
        image = self.non_loaded.get_nowait()
        self.loaded.put((image, 0))

    def _load_queue_full(self):
        for _ in range(self.queue_length):
            self._add_to_loaded()

    def __len__(self):
        return self._len

    def __iter__(self):
        return self

    def __next__(self):
        try:
            image_pair, n_access = self.loaded.get()
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

    def set_len(self, new_len):
        self._len = new_len
