import open_clip
import torch
import torch.nn.functional as F
from attributee import Attributee, String
from attributee.object import class_fullname, import_class
from torch import nn


class AttributeeModule(Attributee, nn.Module):
    """
    Provides a base torch Module class that supports serialization of hyperparameters with
    Attributee.
    """

    def __init__(self, *args, **kwargs):
        nn.Module.__init__(self)
        super().__init__(*args, **kwargs)

    def save_module(self, path: str, **kwargs):
        import datetime, os
        import torch
        dirname = os.path.abspath(os.path.dirname(path))
        os.makedirs(dirname, exist_ok=True)

        with open(path, "wb") as handle:
            data = dict(config=self.dump(), state=self.state_dict(), type=class_fullname(self))
            data.update(kwargs)
            data["timestamp"] = datetime.datetime.now().isoformat()
            torch.save(data, handle)

    @classmethod
    def load_module(cls, path: str, strict: bool = True, metadata: bool = False, cuda=True) -> "AttributeeModule":
        import torch
        with open(path, "rb") as handle:
            if cuda:
                data = torch.load(handle, weights_only=True)
            else:
                data = torch.load(handle, weights_only=True, map_location='cpu')
            module_class = import_class(data["type"])
            assert issubclass(module_class, cls) or not strict
            module = module_class(**data["config"])
            module.load_state_dict(data["state"], strict=strict)
            if metadata:
                del data["state"]
                return module, data
            return module

    def load(self, path: str, strict: bool = True):
        import torch
        with open(path, "rb") as handle:
            data = torch.load(handle, weights_only=True)
            if strict:
                module_class = import_class(data["type"])
                assert isinstance(self, module_class)
            self.load_state_dict(data["state"], strict=strict)


class SBIRModel(AttributeeModule):

    def create_sketch_transforms(self):
        # By default, the sketch transforms are the same as the image transforms.
        return self.create_image_transforms()

    def create_image_transforms(self):
        raise NotImplementedError("SBIRModel is not implemented. Use CLIP_SBIRModel or SiameseSBIRModel instead.")

    def forward(self, data):
        raise NotImplementedError("SBIRModel is not implemented. Use CLIP_SBIRModel or SiameseSBIRModel instead.")

    def use_amp(self):
        """
        Returns True if the model supports automatic mixed precision (AMP) training.
        """
        return False


def _parse_clip_model(model: str) -> (str, str, int):
    size = 224
    # Map the model name to the architecture
    if model == "convnext_base":
        arch = "convnext_base"
        pretrain = "laion400m_s13b_b51k"
    elif model == "convnext_small":
        arch = "convnext_small"
        pretrain = "laion400m_s13b_b51k"
    elif model == "convnext_tiny":
        arch = "convnext_tiny"
        pretrain = "laion400m_s13b_b51k"
    elif model == "convnext_large":
        arch = "convnext_large_d"
        pretrain = "laion2B-s26B-b102K-augreg"
        size = 256
    elif model == "vit_base":
        arch = "ViT-B-16"
        pretrain = "laion2b_s34b_b88k"
    else:
        raise ValueError(
            f"Unknown model name: {model}. Please specify a valid model name (convnext_base, convnext_large, vit_base, vit_large).")

    return arch, pretrain, size


class CLIP_SBIRModel(SBIRModel):
    """
    CLIP_SBIRModel is a wrapper around the CLIP model for sketch-photo retrieval.
    It uses the open_clip library to create a model and preprocess function.
    """

    model = String(default="convnext_base", description="Pretrained model name")

    def __init__(self, *args, **kwargs):
        super(CLIP_SBIRModel, self).__init__(*args, **kwargs)
        arch, pretrain, self.size = _parse_clip_model(self.model)

        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrain)
        self.embedding_net = model.visual

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        res = self.embedding_net(data)
        res = F.normalize(res)
        return res

    def create_image_transforms(self):
        from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB
        from torchvision.transforms import InterpolationMode
        import torch
        transforms = Compose([
            ToImage(),
            ToDtype(torch.uint8, scale=True),
            RGB(),
            Resize((self.size, self.size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        return transforms

    def use_amp(self) -> bool:
        """
        Returns True if the model supports automatic mixed precision (AMP) training.
        """
        return True


# TODO
class SiameseSBIRModel(SBIRModel):
    """
    
    """

    arch = String(default="convnext_base", description="Model architecture")
    init = String(default="imagenet", description="Initialization method")

    def __init__(self, *args, **kwargs):
        from torch.nn import AdaptiveAvgPool2d
        super(SiameseSBIRModel, self).__init__(*args, **kwargs)

        # Get a torchvision model from a string for different architectures, e.g., "resnet18", "resnet50", etc.
        if self.arch == "convnext_base":
            from torchvision.models import convnext_base
            from torchvision.models import ConvNeXt_Base_Weights
            self.embedding_net = convnext_base(
                weights=ConvNeXt_Base_Weights.DEFAULT if self.init == "imagenet" else None).features
            self.num_features = 1024
            self.pool = AdaptiveAvgPool2d(1)
        elif self.arch == "convnext_large":
            from torchvision.models import convnext_large
            from torchvision.models import ConvNeXt_Large_Weights
            self.embedding_net = convnext_large(
                weights=ConvNeXt_Large_Weights.DEFAULT if self.init == "imagenet" else None).features
            self.num_features = 1536
            self.pool = AdaptiveAvgPool2d(1)
        elif self.arch == "convnext_small":
            from torchvision.models import convnext_small
            from torchvision.models import ConvNeXt_Small_Weights
            self.embedding_net = convnext_small(
                weights=ConvNeXt_Small_Weights.DEFAULT if self.init == "imagenet" else None).features
            self.num_features = 768
            self.pool = AdaptiveAvgPool2d(1)
        elif self.arch == "vit_base":
            from torchvision.models import vit_b_16
            from torchvision.models import ViT_B_16_Weights
            self.embedding_net = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if self.init == "imagenet" else None)
            # Get the last layer of the model
            self.embedding_net.heads = nn.Identity()
            self.num_features = 768
            self.pool = nn.Identity()
        elif self.arch == "swin_base":
            from torchvision.models import swin_b
            from torchvision.models import Swin_B_Weights
            self.embedding_net = swin_b(weights=Swin_B_Weights.DEFAULT if self.init == "imagenet" else None).features
            self.num_features = 1024
            self.pool = AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")

        # If the pretrained model is an initialization, we initialize the weights.
        if self.init == "xavier":
            for m in self.embedding_net.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.InstanceNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        elif self.init == "kaiming":
            for m in self.embedding_net.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.GroupNorm):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.InstanceNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        else:
            pass

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        res = self.embedding_net(data)
        res = self.pool(res)
        res = res.view(-1, self.num_features)
        res = F.normalize(res)
        return res

    def create_image_transforms(self):
        from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB
        from torchvision.transforms import InterpolationMode
        import torch
        transforms = Compose([
            Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            RGB(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transforms
