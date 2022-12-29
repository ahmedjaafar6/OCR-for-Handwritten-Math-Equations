# from CNN_Baseline import CNN
from DCCNN_best import DCCNN as CNN
# from DCCNN_paper import DCCNN as CNN
import torch
from utils import get_aida_batch, standardize_image, get_handwritten_values
import numpy as np
from main import Parser, results_overlay
import matplotlib.pyplot as plt


def testAIDA():
    images = []
    border_list = []
    batch = get_aida_batch(1)
    for i in range(1):
        next(batch)
    images.append(next(batch))

    # Test with AIDA
    model = CNN()
    # model.load_state_dict(torch.load("CNN_Baseline.pt"))
    # model.load_state_dict(torch.load("DCCNN.pt"))
    model.load_state_dict(torch.load("DCCNN_best.pt"))
    model.eval()

    img_predictions = []
    for img in images:
        all_slices, borders = Parser(img, 0.005, show=False)
        border_list.append(borders)
        preds = []
        for slic in all_slices:
            slic = slic.astype(np.float32)
            slic = standardize_image(
                slic, square=True, resize=True, invert=True, to_black_and_white=True)
            slic = np.reshape(slic, [1, 1, slic.shape[0], slic.shape[1]])
            slic = torch.tensor(slic).float()
            with torch.no_grad():
                yPred = torch.argmax(model(slic), dim=1).numpy()
            yPred_val = get_handwritten_values(yPred)[0]
            preds.append(yPred_val)
        img_predictions.append(np.array(preds))
    return images, img_predictions, border_list


if __name__ == "__main__":
    images, img_predictions, borders = testAIDA()

    results_overlay(images, img_predictions, borders)
