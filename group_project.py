import math
import argparse
import PySimpleGUI as sg
import os
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
from easyocr import Reader

def findAndRemoveWord(image, word):
    img_copy = np.copy(image)
    langs = ["en"]
    reader = Reader(langs)
    results = reader.readtext(image, width_ths=0)

    for (bbox, text, prob) in results:
        if text.lower() == word.lower():
            (top_left, top_right, bottom_right, bottom_left) = bbox
            tl = (int(top_left[0]), int(top_left[1]))
            br = (int(bottom_right[0]), int(bottom_right[1]))
            
            cv2.rectangle(img_copy, tl, br, (255, 255, 255), -1)  # Fill the region with white

    return img_copy

def resizeImage(image, scale_factor):
    h, w, _ = image.shape
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    resized_image = cv2.resize(image, (new_w, new_h))
    return resized_image

def getText(image, show_boxes=True):
    langs = ["en"]
    reader = Reader(langs)
    results = reader.readtext(image, width_ths=20)
    
    toReturn = ""

    for (bbox, text, prob) in results:
        toReturn += text + "\n"
        
        if show_boxes:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            tl = (int(top_left[0]), int(top_left[1]))
            br = (int(bottom_right[0]), int(bottom_right[1]))
            
            cv2.rectangle(image, tl, br, (0, 0, 255), 2)
            cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)

    return toReturn

def np_im_to_data(im):
    array = np.array(im, dtype=np.uint8)
    im = Image.fromarray(array)
    with BytesIO() as output:
        im.save(output, format='PNG')
        data = output.getvalue()
    return data

def transformImg(img, add_boxes=True):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w = img.shape[1]
    h = img.shape[0]
    min_val = 900
    for i in range(0, h):
        for j in range(0, w):
            if img_gray[i, j] < min_val:
                min_val = img_gray[i, j]
    for i in range(0, h):
        for j in range(0, w):
            if img_gray[i, j] <= (min_val + 20):
                img[i, j] = 0, 0, 0
            else:
                img[i, j] = 255, 255, 255
    
    if add_boxes:
        # Add boxes only if specified
        getText(img, add_boxes)
    
    return img

def gaussianBlur(img, kernel_size=(5, 5), sigmaX=0):
    blurred_img = cv2.GaussianBlur(img, kernel_size, sigmaX)
    return blurred_img

def normalizeBrightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    value = 0.5
    hsv[:,:,2] = hsv[:,:,2] * value
    normalized_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return normalized_img

def main():
    layout = [
        [sg.Button("Load Image"), sg.Button("Save Image"), sg.Button("Quit")], 
        [sg.Button("Increase Size"), sg.Button("Decrease Size")],
        [sg.Button("Clean and Get Text"), sg.Button("Show Boxes"), sg.Button("Hide Boxes")],
        [sg.InputText("", key="-WORD-"), sg.Button("Find and Remove")],
        [sg.Multiline("", size=(60, 5), key="-TEXTBOX-", disabled=True)],
        [sg.Image(filename="", key="-IMAGE-")],  
    ]

    window = sg.Window("Image Viewer", layout, resizable=True)

    image_data = None
    show_boxes = True
    scale_factor = 1.0

    while True:
        event, values = window.read()

        if event == sg.WIN_CLOSED or event == "Quit":
            break
        elif event == "Load Image":
            image_path = sg.popup_get_file("Select an image file", file_types=(("Image Files", "*.png;*.jpg;*.jpeg"),))
            if image_path:
                image_data = load_image(image_path)
                update_image(window, image_data)
        elif event == "Save Image" and image_data is not None:
            save_path = sg.popup_get_file("Save the image as", save_as=True, default_extension=".png", file_types=(("PNG Files", "*.png"),))
            if save_path:
                save_image(save_path, image_data)
        elif event == "Clean and Get Text" and image_data is not None:
            # Normalize brightness before transforming the image
            normalized_data = normalizeBrightness(image_data)
            update_image(window, normalized_data)
            # Transform the image before extracting text
            transformed_data = transformImg(normalized_data, add_boxes=False)
            blurred_data = gaussianBlur(transformed_data)
            update_image(window, blurred_data)
            text_result = getText(transformed_data, False)
            window["-TEXTBOX-"].update(value=text_result)
        elif event == "Show Boxes" and image_data is not None:
            show_boxes = True
            # Draw boxes around words
            img_with_boxes = np.copy(transformed_data)
            getText(img_with_boxes, show_boxes)
            update_image(window, img_with_boxes)
        elif event == "Hide Boxes" and image_data is not None:
            show_boxes = False
            # Remove boxes from the image
            img_without_boxes = np.copy(transformed_data)
            getText(img_without_boxes, show_boxes)
            update_image(window, img_without_boxes)
        elif event == "Find and Remove" and image_data is not None:
            word_to_remove = values["-WORD-"]
            if word_to_remove.strip() != "":
                # Find and remove the specified word
                transformed_data = findAndRemoveWord(transformed_data, word_to_remove)
                update_image(window, transformed_data)
        elif event == "Increase Size" and image_data is not None:
            scale_factor += 0.1
            transformed_data = resizeImage(image_data, scale_factor)
            update_image(window, transformed_data)
        elif event == "Decrease Size" and image_data is not None:
            scale_factor = max(0.1, scale_factor - 0.1)
            transformed_data = resizeImage(image_data, scale_factor)
            update_image(window, transformed_data)

    window.close()

def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def update_image(window, image_data):
    if image_data is not None:
        imgbytes = cv2.imencode(".png", image_data)[1].tobytes()
        window["-IMAGE-"].update(data=imgbytes)

def save_image(save_path, image_data):
    image = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, image)

if __name__ == "__main__":
    main()
