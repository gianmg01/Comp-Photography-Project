
# Alexie Linardatos
# 100746621

import math
import argparse
import PySimpleGUI as sg
import os
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
from scipy.ndimage import gaussian_filter as gf


matplotlib.use('TkAgg')

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def load_image(filename):
    image = cv2.imread(filename)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image is not None else None

def display_images_with_textbox(original_image, image_name, image_width, image_height, ):
    original_data = np_im_to_data(original_image)

    original_height = original_image.shape[0]
    original_width = original_image.shape[1]

   
    layout = [
        [sg.Graph(
            canvas_size=(original_width, original_height),
            graph_bottom_left=(0, 0),
            graph_top_right=(original_width, original_height),
            key='-IMAGES-',
            background_color='white',
            change_submits=True,
            drag_submits=True)],
       #  [sg.Multiline(size=(original_width, original_height), key='-TEXTBOX-', font=('Helvetica', 12))],

            [sg.Multiline(size=(30, 5), key='textbox')],
        [sg.Text(f'Image Name: {image_name}', size=(30, 1)),
         sg.Text(f'Original Size: {image_width}x{image_height}', size=(20, 1))],
        [sg.Button('Load Image', size=(15, 1)), sg.Button('Apply', size=(15, 1)), sg.Button('Save Text', size=(15, 1))],
    ]

    window = sg.Window('Image Viewer', layout, finalize=True)
    window['-IMAGES-'].draw_image(data=original_data, location=(0, original_height))

    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break
        elif event == 'Apply':
            pass
            # You can access the content of the text box using window['-TEXTBOX-'].get()
            # Convert the text to image or apply your logic here
           
        if event == "Load Image":
            load_filename = sg.popup_get_file('Load Image', file_types=(("Image Files", "*.png;*.jpg;*.jpeg;*.bmp"),))
            if load_filename:
                loaded_image = load_image(load_filename)
                if loaded_image is not None:
                    original_image = loaded_image.copy()
                    original_data = np_im_to_data(original_image)
                    window['-IMAGES-'].draw_image(data=original_data, location=(0, original_height))
                    #window['-TEXTBOX-'].update(value='')  # Clear the text box when loading a new image
        elif event == 'Save Text':
            pass
            # new_filename = sg.popup_get_text('Enter a new name for the image:')
            # if new_filename:
            #     if not new_filename.endswith('.txt'):
            #         new_filename += '.png'
            #     new_file_path = os.path.join(os.getcwd(), new_filename)
            #     with open(new_file_path, 'wb') as new_file:
            #         new_file.write(np_im_to_data(original_image))
            #     sg.popup(f'Image saved as {new_filename}')


    window.close()


def main():
    parser = argparse.ArgumentParser(description='A simple image viewer.')
    parser.add_argument('file', action='store', help='Image file.')
    args = parser.parse_args()

    print(f'Loading {args.file} ... ', end='')
    im = cv2.imread(args.file, cv2.COLOR_BGR2RGB)
    print(f'{im.shape}')

    h = 280
    print(f'Resizing the image to have height ' + str(h) + '...', end='')
    hh, w = im.shape[:2]
    r = h / float(hh)
    dim = (int(w * r), h)
    im = cv2.resize(im, dim, interpolation=cv2.INTER_LINEAR)
    print(f'{im.shape}')

    display_images_with_textbox(im, args.file, w, h)

if __name__ == '__main__':
    main()