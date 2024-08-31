def vector_str(vector):
    str_list = []
    for i, el in enumerate(vector):
        str_list.append(f"{el:.4f}")
    return " ".join(str_list)

def get_idx_of_image(image, translations):
    img_idx_map = {tuple(img): idx for idx, img in enumerate(translations)}
    return img_idx_map[tuple(image)]
    