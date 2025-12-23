import torch

def sliding_window_inference(model, image, window_size, step, device, tag=True):
    """
    image: Tensor (1, C, H, W)
    return: Tensor (1, 1, H, W)
    """
    _, _, h, w = image.shape
    pred_sum = torch.zeros((1, 1, h, w), device=device)
    count_map = torch.zeros((1, 1, h, w), device=device)

    window_x_list = list(range(0, w - window_size + 1, step))
    window_y_list = list(range(0, h - window_size + 1, step))
    # 增补边界窗口
    if window_x_list[-1] != w - window_size:
        window_x_list.append(w - window_size)
    if window_y_list[-1] != h - window_size:
        window_y_list.append(h - window_size)
    
    for y in window_y_list:
        for x in window_x_list:
            cropped_image = image[:, :, y:y + window_size, x:x + window_size]
            with torch.no_grad():
                _, pred = model(cropped_image.to(device), tag)
            pred_sum[:, :, y:y + window_size, x:x + window_size] += pred
            count_map[:, :, y:y + window_size, x:x + window_size] += 1
    
    return pred_sum / (count_map + 1e-6)
