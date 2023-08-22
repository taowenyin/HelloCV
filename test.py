import torch
import matplotlib.pyplot as plt

from torch import nn
from PIL import Image
from torchvision import transforms
from pytorch_wavelets import DWTForward, DWTInverse
from torchvision.utils import save_image


if __name__ == "__main__":
    # image = torch.randn(6, 3, 128, 128)
    image = Image.open("data/norain-1x2.png")
    toTensor = transforms.Compose([transforms.ToTensor()])
    toPIL = transforms.Compose([transforms.ToPILImage()])
    image = toTensor(image).unsqueeze(0)

    xfm = DWTForward(J=1, wave="haar", mode="zero")
    ifm = DWTInverse(wave="haar", mode="zero")

    # YLL [B, C, H / 2, W / 2]
    # YH [B, C, 3, H / 2, W / 2]
    YLL, YH = xfm(image)
    YHL = YH[0][:, :, 0, :, :]
    YLH = YH[0][:, :, 1, :, :]
    YHH = YH[0][:, :, 2, :, :]

    Y = ifm((YLL, YH))
    # Y = toPIL(Y.squeeze(0))

    # YLL = toPIL(YLL.squeeze(0))
    # YHL = toPIL(YHL.squeeze(0))
    # YLH = toPIL(YLH.squeeze(0))
    # YHH = toPIL(YHH.squeeze(0))

    save_image(YLL, "LL.png")
    save_image(YHL, "YHL.png")
    save_image(YLH, "YLH.png")
    save_image(YHH, "YHH.png")
    save_image(Y, "Y.png")

    # plt.subplot(2, 3, 1)
    # plt.title("LL")
    # plt.imshow(YLL)

    # plt.subplot(2, 3, 2)
    # plt.title("HL")
    # plt.imshow(YHL)

    # plt.subplot(2, 3, 3)
    # plt.title("LL")
    # plt.imshow(YLH)

    # plt.subplot(2, 3, 4)
    # plt.title("LL")
    # plt.imshow(YHH)

    # plt.subplot(2, 3, 5)
    # plt.title("Y")
    # plt.imshow(Y)

    # plt.show()
