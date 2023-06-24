import numpy as np
import os
import pickle
import sys
import torchvision
import torch
import matplotlib.pyplot as plt
sys.path.append('/content/mixed-segdec-net-comind2021')
sys.path.append('/content/mixed-segdec-net-comind2021/data')
from data.dataset import Dataset
from config import Config
from os import listdir
from os.path import isfile, join
from PIL import Image

class MagneticDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(MagneticDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_contents(self):
        datasetPath = os.path.join(self.cfg.DATASET_PATH, self.kind)
        pos_samples, neg_samples = [], []
        #Per ogni cartella nella cartella del dataset se è free li metto in neg altrimenti in pos
        categorie = listdir(datasetPath)
        for categoria in categorie:
          percorsoCategoria = os.path.join(datasetPath, categoria)
          percorsoImmagini = os.path.join(percorsoCategoria, "Imgs")
          immagini = [f for f in listdir(percorsoImmagini) if isfile(join(percorsoImmagini, f))]
          for immagine in immagini:
            if(".png" in immagine): continue
            nomeImmagine = immagine[:-4]
            percorsoImmagine = os.path.join(percorsoImmagini, immagine)
            percorsoMaschera = os.path.join(percorsoImmagini, nomeImmagine + ".png")
            image = self.read_img_resize(percorsoImmagine, self.grayscale, self.image_size)
            if "Free" not in categoria:
                seg_mask, _ = self.read_label_resize(percorsoMaschera, self.image_size, dilate=self.cfg.DILATE)
                image = self.to_tensor(image)
                seg_loss_mask = self.distance_transform(seg_mask, self.cfg.WEIGHTED_SEG_LOSS_MAX, self.cfg.WEIGHTED_SEG_LOSS_P)
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                seg_loss_mask = self.to_tensor(self.downsize(seg_loss_mask))
                pos_samples.append((image, seg_mask, seg_loss_mask, True, percorsoImmagine, percorsoMaschera, nomeImmagine))
            else:
                seg_mask = np.zeros_like(image)
                image = self.to_tensor(image)
                seg_loss_mask = self.to_tensor(self.downsize(np.ones_like(seg_mask)))
                seg_mask = self.to_tensor(self.downsize(seg_mask))
                neg_samples.append((image, seg_mask, seg_loss_mask, True, percorsoImmagine, percorsoMaschera, nomeImmagine))
        
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        self.num_pos = len(pos_samples)
        self.num_neg = len(neg_samples)
        self.len = len(pos_samples) + len(neg_samples)
        print(self.num_pos)
        print(self.num_neg)
        #Per il momento disabilitata self.init_extra()
        self.init_extra()