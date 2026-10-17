"""Encoder (ImageNet ResNet-50 with partial fine-tuning) and whole-image features."""
import numpy as np
import torch
import torch.nn.functional as F

from .data import resize, to_tensor


def build_model(train_layers="4", pretrained=True):
    """ResNet-50 without the classifier (2048-d pooled output).

    Only the stages listed in train_layers (e.g. "4" or "3,4") receive gradients; the others keep
    their ImageNet weights and BatchNorm statistics (see set_train).
    """
    import torchvision
    try:                                    # torchvision >= 0.13
        net = torchvision.models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
    except TypeError:                       # older torchvision
        net = torchvision.models.resnet50(pretrained=pretrained)
    net.fc = torch.nn.Identity()
    net._train_layers = [s for s in str(train_layers).split(",") if s]
    for name, p in net.named_parameters():
        p.requires_grad = name.startswith(tuple(f"layer{i}" for i in net._train_layers))
    return net


def set_train(net, on):
    """Training mode for the fine-tuned stages only; frozen stages stay in eval mode."""
    net.train(on)
    if on:
        frozen = [net.conv1, net.bn1] + [getattr(net, f"layer{i}") for i in range(1, 5)
                                         if str(i) not in net._train_layers]
        for mod in frozen:
            mod.eval()


@torch.no_grad()
def image_features(net, images, files, device, bs=96):
    """Feature of a whole image = normalised mean of the normalised features of a 2 x 3 grid of
    square crops (side 0.5 of the short image side, resized to 224 px). Returns len(files) x 2048."""
    set_train(net, False)
    crops, owner = [], []
    for i, f in enumerate(files):
        im = images[f]
        H, W = im.shape[:2]
        c = int(0.5 * min(H, W))
        for y in np.linspace(0, H - c, 2).astype(int):
            for x in np.linspace(0, W - c, 3).astype(int):
                crops.append(resize(im[y:y + c, x:x + c], 224))
                owner.append(i)
    out = []
    use_amp = str(device).startswith("cuda")
    for s in range(0, len(crops), bs):
        x = torch.stack([to_tensor(c) for c in crops[s:s + bs]]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out.append(F.normalize(net(x).float(), dim=-1).cpu())
    z = torch.cat(out)
    feat = torch.zeros(len(files), z.shape[1]).index_add_(0, torch.tensor(owner), z)
    return F.normalize(feat, dim=-1)
