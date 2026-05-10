#!/usr/bin/env python3
"""Merge individual Multilabeller .h5 exports into one .h5 file.

Expected input structure per file:
  <image_name>/img
  <image_name>/contours/cnt_XXXXXX
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import h5py


def merge_h5_files(input_dir: Path, output_file: Path) -> tuple[int, int]:
    h5_files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".h5"
    )

    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {input_dir}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_contours = 0

    with h5py.File(output_file, "w") as h5_out:
        h5_out.attrs["date"] = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as h5_in:
                for img_key in h5_in.keys():
                    if img_key not in h5_out:
                        img_group = h5_out.create_group(img_key)
                        img_group.create_dataset("img", data=h5_in[img_key]["img"][...])
                        img_group.create_group("contours")
                        n_images += 1
                    else:
                        src_img = h5_in[img_key]["img"]
                        dst_img = h5_out[img_key]["img"]
                        if src_img.shape != dst_img.shape or src_img.dtype != dst_img.dtype:
                            print(
                                f"Warning: duplicate image key '{img_key}' in {h5_path} has "
                                "different image shape/dtype; keeping first image and merging "
                                "only contours."
                            )
                        else:
                            print(
                                f"Warning: duplicate image key '{img_key}' found in {h5_path}; "
                                "keeping first image data and merging contours."
                            )

                    contours_group = h5_out[img_key]["contours"]
                    next_index = len(contours_group)

                    if "contours" not in h5_in[img_key]:
                        continue

                    for cnt_key in sorted(h5_in[img_key]["contours"].keys()):
                        contour = h5_in[img_key]["contours"][cnt_key][...]
                        contours_group.create_dataset(
                            f"cnt_{next_index:06d}", data=contour
                        )
                        next_index += 1
                        n_contours += 1

    return n_images, n_contours


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge individual .h5 files into a single .h5 file."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing individual .h5 files.",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to merged output .h5 file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.exists() or not args.input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {args.input_dir}")

    n_images, n_contours = merge_h5_files(args.input_dir, args.output_file)
    print(
        f"Merged {n_images} images and {n_contours} contours into {args.output_file}"
    )


if __name__ == "__main__":
    main()
