import copy
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from net.blip import blip_decoder


class BLIP(nn.Module):
    def __init__(
        self,
        num_beam: int = 3,
        max_length: int = 20,
        min_length: int = 5,
        pretrained: str = 'ckpt/model_base_caption_capfilt_large.pth',
        image_size: int = 384,
        vit: str = 'base',
    ):
        super().__init__()

        self.num_beam = num_beam
        self.max_length = max_length
        self.min_length = min_length

        self.model = blip_decoder(
            pretrained=pretrained,
            image_size=image_size,
            vit=vit
        )
        # self.model_train = copy.deepcopy(self.model)

        self.transform = transforms.Compose([transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC)])

    def forward(self, x: torch.Tensor, embeds=None):
        im = self.transform(x)

        # 不训练
        self.model.eval()
        with torch.no_grad():
            caption, image_embeds = self.model.generate(im, sample=True, num_beams=self.num_beam, max_length=self.max_length, min_length=self.min_length, embeds=embeds)

        return caption, image_embeds

        # 训练
        # caption_train, image_embeds_train = self.model_train.generate(im, sample=True, num_beams=self.num_beam, max_length=self.max_length, min_length=self.min_length)
        #
        # self.model.eval()
        # with torch.no_grad():
        #     caption, image_embeds = self.model.generate(im, sample=True, num_beams=self.num_beam, max_length=self.max_length, min_length=self.min_length)
        #
        # return caption, image_embeds, caption_train, image_embeds_train

        # caption, image_embeds = self.model.generate(im, sample=True, num_beams=self.num_beam, max_length=self.max_length, min_length=self.min_length)
        # return caption, image_embeds

def load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert('RGB')

    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


if __name__ == '__main__':
    predictor = BLIP().to('cuda')
    im = load_image('2317073.jpg', 384, 'cuda')
    print(predictor.forward(im))