from PIL import Image


def translate_image(generator, input_image_path, output_image_path):
    """
    Translate an input image using the trained Pix2Pix generator.
    """
    image = Image.open(input_image_path)
    # Perform translation using the generator
    image.save(output_image_path)
