import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="ldmvfi-vqflow-f32-c256-concat_max.ckpt",
        help="Path to the full LDMVFI checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="vqflow-extracted.ckpt",
        help="Path to save the extracted first-stage checkpoint.",
    )
    parser.add_argument(
        "--prefix",
        default="first_stage_model.",
        help="State-dict prefix to extract.",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    extracted = {}
    for key, value in state_dict.items():
        if key.startswith(args.prefix):
            extracted[key[len(args.prefix):]] = value

    if not extracted:
        raise ValueError(f"No parameters found with prefix {args.prefix!r} in {args.input}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out = {"state_dict": extracted}
    torch.save(out, args.output)

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"prefix={args.prefix}")
    print(f"num_extracted_keys={len(extracted)}")


if __name__ == "__main__":
    main()
