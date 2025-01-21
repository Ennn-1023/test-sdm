import argparse, os, sys, glob
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


def make_batch(image, mask, device):
    # image: the masked&preprocessed image
    # mask: the mask
    image = np.array(Image.open(image).convert("RGB").resize((512, 512)))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L").resize((512, 512)))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = (1-mask)*image

    batch = {"mask": mask, "masked_image": masked_image}
    for k in batch:
        batch[k] = batch[k].to(device=device)
        batch[k] = batch[k]*2.0-1.0
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to model weights",
    )
    parser.add_argument(
        "--inmask",
        type=str,
        nargs="?",
        help="dir containing mask (`example.png` with same name as image)",
    )
    parser.add_argument(
        "--inimage",
        type=str,
        nargs="?",
        help="dir containing image (`example.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    opt = parser.parse_args()

    masks = sorted(os.path.join(opt.inmask, f) for f in os.listdir(opt.inmask))
    images = sorted(os.path.join(opt.inimage, f) for f in os.listdir(opt.inimage))
    print(masks)
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load(opt.config)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(opt.weights)["state_dict"],
                          strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    with torch.no_grad():
        with model.ema_scope():
            for image, mask in tqdm(zip(images, masks)):
                outpath = os.path.join(opt.outdir, os.path.split(image)[1])
                batch = make_batch(image, mask, device=device)

                # encode masked image and concat downsampled mask
                c = model.get_first_stage_encoding(model.encode_first_stage(batch["masked_image"]))

                cc = torch.nn.functional.interpolate(batch["mask"],
                                                     size=c.shape[-2:])
                c = torch.cat((c, cc), dim=1)

                shape = (c.shape[1]-1,)+c.shape[2:]
                samples_ddim, _ = sampler.sample(S=opt.steps,
                                                 conditioning=c,
                                                 batch_size=c.shape[0],
                                                 shape=shape,
                                                 verbose=False)
                x_samples_ddim = model.decode_first_stage(samples_ddim)

                image = torch.clamp((batch["masked_image"]+1.0)/2.0,
                                    min=0.0, max=1.0)
                mask = torch.clamp((batch["mask"]+1.0)/2.0,
                                   min=0.0, max=1.0)
                predicted_image = torch.clamp((x_samples_ddim+1.0)/2.0,
                                              min=0.0, max=1.0)

                inpainted = (1-mask)*image+mask*predicted_image
                inpainted = inpainted.cpu().numpy().transpose(0,2,3,1)[0]*255
                Image.fromarray(inpainted.astype(np.uint8)).save(outpath)
