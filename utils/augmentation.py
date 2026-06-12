import numpy as np
from scipy.ndimage import rotate, shift
from PIL import Image

def augment_image(img, dataset='mnist'):
    img_aug = img.copy()
    if dataset == 'mnist':
        max_rotation = 15
        max_shift = 2
        max_zoom = 0.15
    else:  # digits
        max_rotation = 10
        max_shift = 1
        max_zoom = 0.1

    # Rotación
    angle = np.random.uniform(-max_rotation, max_rotation)
    img_aug = rotate(img_aug, angle, reshape=False, order=1, cval=0, mode='constant')
    
    # Desplazamiento
    shift_y = np.random.randint(-max_shift, max_shift+1)
    shift_x = np.random.randint(-max_shift, max_shift+1)
    img_aug = shift(img_aug, (shift_y, shift_x), cval=0, order=1)
    
    # Zoom (redimensionar con PIL para evitar errores)
    zoom_factor = np.random.uniform(1-max_zoom, 1+max_zoom)
    h, w = img_aug.shape
    new_h = int(round(h * zoom_factor))
    new_w = int(round(w * zoom_factor))
    # Convertir a PIL para redimensionar
    pil_img = Image.fromarray((img_aug * 255).astype(np.uint8))
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    img_resized = np.array(pil_img, dtype=np.float32) / 255.0
    
    # Ajustar al tamaño original (h, w) mediante recorte o padding
    if new_h >= h and new_w >= w:
        # Recortar centro
        y_start = (new_h - h) // 2
        x_start = (new_w - w) // 2
        img_resized = img_resized[y_start:y_start+h, x_start:x_start+w]
    else:
        # Padding
        pad_y = (h - new_h) // 2
        pad_x = (w - new_w) // 2
        padded = np.zeros((h, w), dtype=np.float32)
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized
        img_resized = padded
    
    # Asegurar valores en [0,1]
    img_aug = np.clip(img_resized, 0, 1)
    return img_aug

def augment_batch(images, dataset='mnist', prob=0.7):
    augmented = images.copy()
    for i in range(images.shape[0]):
        if np.random.rand() < prob:
            augmented[i] = augment_image(images[i], dataset=dataset)
    return augmented