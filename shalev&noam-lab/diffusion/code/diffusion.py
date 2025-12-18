import argparse
from pathlib import Path
import random

import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

PROMPTS = {
    "rain": {
        "prompt": (
            "photorealistic dashcam RAW photo, real-world severe weather, "
            "violent rainstorm, torrential rain, extreme downpour, intense rain streaks, "
            "large raindrops and smear on windshield, heavy water splashes, strong tire spray, "
            "standing water and puddles on road, wet asphalt mirror reflections, "
            "dark storm clouds, gloomy overcast sky, very low visibility, atmospheric haze, "
            "motion blur in rain only, cinematic realism, natural colors, "
            "same scene, preserve the exact vehicles and road layout, no object relocation"
        ),
        "negative": (
            "cartoon, anime, illustration, CGI, render, text, watermark, logo, "
            "extra cars, duplicated vehicles, missing vehicles, changed vehicle shape, "
            "warped road, bent lanes, deformed objects, melted surfaces, geometry change, "
            "sunny, clear sky, dry road, fantasy, surreal"
        ),
    },
    "snow": {
        "prompt": (
            "photorealistic dashcam RAW photo, real-world severe winter storm, "
            "heavy snowstorm, thick snowfall, dense snowflakes close to camera, "
            "blowing snow, strong wind gusts, swirling snow, icy slush on asphalt, "
            "snow buildup on road edges, frozen mist, cold gray atmosphere, "
            "very low visibility, reduced contrast, realistic winter lighting, "
            "cinematic realism, natural colors, "
            "same scene, preserve the exact vehicles and road layout, no object relocation"
        ),
        "negative": (
            "whiteout blank image, pure white frame, cartoon, anime, illustration, CGI, render, "
            "text, watermark, logo, summer, green grass, tropical, sunny, "
            "extra cars, duplicated vehicles, missing vehicles, changed vehicle shape, "
            "warped road, deformed objects, geometry change"
        ),
    },
    "fog": {
        "prompt": (
            "photorealistic dashcam RAW photo, real-world extreme dense fog, "
            "very thick gray haze, heavy mist, smoke-like fog in the air (NOT fire smoke), "
            "uniform volumetric fog layer, strong atmospheric perspective, "
            "major contrast reduction, desaturated colors, soft edges, "
            "distant objects almost disappear, very low visibility, realistic optics, "
            "cinematic realism, natural colors, "
            "same scene, preserve the exact vehicles and road layout, no object relocation"
        ),
        "negative": (
            "fire, flames, burning, black smoke, explosion, chimney smoke, "
            "cartoon, anime, illustration, CGI, render, text, watermark, logo, "
            "extra cars, duplicated vehicles, missing vehicles, changed vehicle shape, "
            "warped road, deformed objects, geometry change, night, pitch black"
        ),
    },
}

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def list_images(in_dir: Path):
    return sorted([p for p in in_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def pad_to_multiple_of_8(img: Image.Image):
    w, h = img.size
    new_w = ((w + 7) // 8) * 8
    new_h = ((h + 7) // 8) * 8
    if new_w == w and new_h == h:
        return img, (0, 0, w, h)
    padded = Image.new("RGB", (new_w, new_h))
    padded.paste(img, (0, 0))
    return padded, (0, 0, w, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)

    
    ap.add_argument("--model_id", default="SG161222/RealVisXL_V4.0")

    
    ap.add_argument("--strength", type=float, default=0.45)
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=6.5)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--split", default="train")
    ap.add_argument("--only", default="", help="comma-separated: rain,snow,fog")

    ap.add_argument("--cpu_offload", action="store_true")
    ap.add_argument("--force_fp32", action="store_true")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)

    images = list_images(in_dir)
    if not images:
        raise RuntimeError(f"No images found in: {in_dir}")

    if args.shuffle:
        random.seed(args.seed)
        random.shuffle(images)

    if args.limit and args.limit > 0:
        images = images[:args.limit]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.force_fp32 or device != "cuda":
        dtype = torch.float32
        variant = None
    else:
        dtype = torch.float16
        variant = "fp16"

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant=variant,
    ).to(device)

    if device == "cuda":
        pipe.enable_attention_slicing()
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

       
        if hasattr(pipe, "upcast_vae"):
            pipe.upcast_vae()
        else:
            pipe.vae.to(dtype=torch.float32)

        if args.cpu_offload:
            try:
                import accelerate  
                pipe.enable_model_cpu_offload()
            except Exception:
                print("cpu_offload requested but 'accelerate' not available. Install: pip install accelerate")

    selected = PROMPTS
    if args.only.strip():
        allowed = {x.strip() for x in args.only.split(",") if x.strip()}
        selected = {k: v for k, v in PROMPTS.items() if k in allowed}
        if not selected:
            raise RuntimeError(f"--only had no valid keys. Use one of: {list(PROMPTS.keys())}")

    for subtype, pp in selected.items():
        out_dir = out_root / subtype / "images" / args.split
        out_dir.mkdir(parents=True, exist_ok=True)

        for i, img_path in enumerate(images):
            init_img = Image.open(img_path).convert("RGB")
            padded, crop_box = pad_to_multiple_of_8(init_img)

            gen = torch.Generator(device=device).manual_seed(args.seed + i)

            with torch.inference_mode():
                out = pipe(
                    prompt=pp["prompt"],
                    negative_prompt=pp["negative"],
                    image=padded,
                    strength=args.strength,
                    guidance_scale=args.cfg,
                    num_inference_steps=args.steps,
                    generator=gen,
                ).images[0]

            out.crop(crop_box).save(out_dir / img_path.name)

            if (i + 1) % 10 == 0:
                print(f"[{subtype}] {i+1}/{len(images)}")

    print(f"Finished! Images saved to: {out_root}")


if __name__ == "__main__":
    main()
