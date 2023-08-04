from torch import nn, optim
from matplotlib import pyplot as plt
from collections import defaultdict

from .get_device import get_device, plot_examples

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class Runner(object):
    def __init__(self, model, max_epochs=None, precision="32-true"):
        self.model = model
        self.dataset = model.dataset
        self.trainer = Trainer(callbacks= ModelSummary(max_depth=10), max_epochs=max_epochs or model.max_epochs, precision= precision)
        self.incorrect_preds = None
        self.grad_cam = None
    
    def run(self):
        return self.trainer.fit(self.model)
    
    def get_incorrect_preds(self):
        self.incorrect_preds = defaultdict(list)
        incorrect_images = list()
        processed = 0
        results = self.trainer.predict(self.model, self.model.predict_dataloader())

        




