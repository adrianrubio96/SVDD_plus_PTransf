from .torchvision_dataset import TorchvisionDataset
import torchvision.datasets as tvds
import torchvision.transforms as transforms


class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str):
        super().__init__(root)
        self.train_set = tvds.MNIST(root=self.root, train=True, download=True)
        self.test_set = tvds.MNIST(root=self.root, train=False, download=True)
