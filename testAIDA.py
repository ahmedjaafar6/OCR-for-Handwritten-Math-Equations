from CNN_Baseline import CNN
import torch
from utils import get_aida_batch, standardize_image, get_handwritten_values
import numpy as np
from main import Parser, results_overlay
import matplotlib.pyplot as plt


def testAIDA():
    images = []
    border_list = []
    batch = get_aida_batch(2)
    for i in range(2):
        images.append(next(batch))

    # Test with AIDA
    model = CNN()
    model.load_state_dict(torch.load("CNN_Baseline.pt"))
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
            # plt.imshow(slic[0][0], cmap="gray")
            # plt.show()
        img_predictions.append(np.array(preds))
    return images, img_predictions, border_list


if __name__ == "__main__":
    images, img_predictions, borders = testAIDA()

    results_overlay(images, img_predictions, borders)
