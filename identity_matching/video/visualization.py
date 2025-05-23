
import os
import PIL


def create_pair_images(identities_images: {int: [PIL.Image]}, pairs: [(int, int)]) -> [(PIL.Image, PIL.Image, PIL.Image)]:
    """
    Creates a list of paired images based on identity images and pairs.

    :param identities_images: Dictionary of identity images.
    :param pairs: List of pairs of identity indices.
    :return: List of combined paired images as PIL Image objects.
    """

    paired_images = []
    for i, pair in enumerate(pairs):
        img1 = next((img for img in identities_images[pair[0]] if img), None)
        if not img1:
            print(f"Visualization error, pair {i}")
            continue

        img2 = next((img for img in identities_images[pair[1]] if img), None)
        if not img2:
            print(f"Visualization error, pair {i}")
            continue

        # adjust by the smaller height
        min_height = min(img1.height, img2.height)
        if img1.height != min_height:
            aspect_ratio1 = img1.width / img1.height
            img1 = img1.resize((int(min_height * aspect_ratio1), min_height), PIL.Image.LANCZOS)
        if img2.height != min_height:
            aspect_ratio2 = img2.width / img2.height
            img2 = img2.resize((int(min_height * aspect_ratio2), min_height), PIL.Image.LANCZOS)

        dst_width = img1.width + img2.width
        dst = PIL.Image.new("RGB", (dst_width, min_height))
        dst.paste(img1, (0, 0))
        dst.paste(img2, (img1.width, 0))

        paired_images.append((dst, img1, img2))

    return paired_images


def save_pair_images(paired_images: [(PIL.Image, PIL.Image, PIL.Image)], output_folder: str) -> None:
    """
    Saves paired images to subfolders within the specified output folder.

    :param paired_images: List of paired images as PIL Image objects.
    :param output_folder: Path to the output folder where paired images will be saved.
    """

    for i, (img_merged, img_0, img_1) in enumerate(paired_images):
        subfolder_path = os.path.join(output_folder, f"pair_{i}")
        os.makedirs(subfolder_path, exist_ok=True)
        img_merged.save(os.path.join(subfolder_path, "merged.jpg"))
        img_0.save(os.path.join(subfolder_path, "img0.jpg"))
        img_1.save(os.path.join(subfolder_path, "img1.jpg"))


def create_and_save_pair_images(identities_images: {int: [PIL.Image]}, pairs: [(int, int)], output_folder: str) -> None:
    """
    Creates and saves paired images based on identity images and pairs.

    :param identities_images: Dictionary of identity images.
    :param pairs: List of pairs of identity indices.
    :param output_folder: Path to the output folder where paired images will be saved.
    """

    paired_images = create_pair_images(identities_images, pairs)
    save_pair_images(paired_images, output_folder)


if __name__ == "__main__":
    pass
