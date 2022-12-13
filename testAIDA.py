from CNN_Baseline import CNN
import torch
from utils import get_aida_batch, standardize_image, get_handwritten_values
import numpy as np
from main import Parser, results_overlay


def testAIDA():
    images = []
    batch = get_aida_batch(2)
    for i in range(10):
        images.append(next(batch))
    images = np.array(images)

    # Test with AIDA
    model = CNN()
    model.load_state_dict(torch.load("CNN_Baseline.pt"))
    model.eval()

    img_predictions = []
    for img in images:
        all_slices = Parser(img, 0.005, show=False)
        preds = []
        for slic in all_slices:
            slic = slic.astype(np.float32)
            slic = standardize_image(slic, square=True, resize=True)
            slic = np.reshape(slic, [1, 1, slic.shape[0], slic.shape[1]])
            slic = torch.tensor(slic).float()
            yPred = torch.argmax(model(slic), dim=1).numpy()
            yPred_val = get_handwritten_values(yPred)[0]
            preds.append(yPred_val)
        img_predictions.append(np.array(preds))
    img_predictions = np.array(img_predictions)
    return images, img_predictions


if __name__ == "__main__":
    print(testAIDA())
