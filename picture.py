# # import cv2
# # import os
# # import numpy as np
# #
# # img_path = r'D:\Pycharm\learning\pythonstudyProject\picturer2.png'  # 请输入自己需要放大图像的路径
# #
# # img_name = os.path.basename(img_path)
# # img = cv2.imread(img_path)
# #
# #
# # def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
# #     global img
# #     if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键则放大所点的区域
# #         xy = "%d,%d" % (x, y)
# #         print(xy)
# #         length = 20  # 局部区域的边长的一半
# #         big_length = 200  # 放大后图像的边长
# #         part_left = x - length
# #         part_right = x + length
# #         part_top = y - length
# #         part_bottom = y + length
# #         height, width, _ = np.shape(img)
# #         if (x < width / 2) & (y < height / 2):
# #             loc_left = 10
# #             loc_top = 10
# #             loc_right = loc_left + big_length
# #             loc_bottom = loc_top + big_length
# #             cv2.line(img, (part_right, part_top), (loc_right, loc_top), (0, 0, 0), 2)
# #             cv2.line(img, (part_left, part_bottom), (loc_left, loc_bottom), (0, 0, 0), 2)
# #         elif (x >= width / 2) & (y < height / 2):
# #             loc_right = width - 10
# #             loc_left = loc_right - big_length
# #             loc_top = 10
# #             loc_bottom = loc_top + big_length
# #             cv2.line(img, (part_left, part_top), (loc_left, loc_top), (0, 0, 0), 2)
# #             cv2.line(img, (part_right, part_bottom), (loc_right, loc_bottom), (0, 0, 0), 2)
# #         elif (x < width / 2) & (y >= height / 2):
# #             loc_left = 10
# #             loc_right = loc_left + big_length
# #             loc_bottom = height - 10
# #             loc_top = loc_bottom - big_length
# #             cv2.line(img, (part_left, part_top), (loc_left, loc_top), (0, 0, 0), 2)
# #             cv2.line(img, (part_right, part_bottom), (loc_right, loc_bottom), (0, 0, 0), 2)
# #         elif (x >= width / 2) & (y >= height / 2):
# #             loc_bottom = height - 10
# #             loc_top = loc_bottom - big_length
# #             loc_right = width - 10
# #             loc_left = loc_right - big_length
# #             cv2.line(img, (part_right, part_top), (loc_right, loc_top), (0, 0, 0), 2)
# #             cv2.line(img, (part_left, part_bottom), (loc_left, loc_bottom), (0, 0, 0), 2)
# #
# #         part = img[part_top:part_bottom, part_left:part_right]
# #         mask = cv2.resize(part, (big_length, big_length), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
# #         img[loc_top:loc_bottom, loc_left:loc_right] = mask
# #         cv2.rectangle(img, (part_left, part_top), (part_right, part_bottom), (0, 0, 0), 2)
# #         cv2.rectangle(img, (loc_left, loc_top), (loc_right, loc_bottom), (0, 0, 0), 2)
# #         cv2.imshow("image", img)
# #
# #     if event == cv2.EVENT_RBUTTONDOWN:  # 按下鼠标右键恢复原图
# #         img = cv2.imread(img_path)
# #         cv2.imshow("image", img)
# #
# #
# # cv2.namedWindow("image")
# # cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# # cv2.imshow("image", img)
# #
# # cv2.waitKey(0)
# # cv2.imwrite("image1.jpg", img)
#
#
# import cv2
# import os
# import numpy as np
#
# img_path = r'D:\Pycharm\learning\pythonstudyProject\picturer2.png'  # 请输入自己需要放大图像的路径
#
# img_name = os.path.basename(img_path)
# img = cv2.imread(img_path)
#
# if img is None:
#     raise FileNotFoundError(f"Image not found at {img_path}")
#
# def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     global img
#     if event == cv2.EVENT_LBUTTONDOWN:  # 按下鼠标左键则放大所点的区域
#         xy = "%d,%d" % (x, y)
#         print(f"Mouse clicked at: {xy}")
#         length = 20  # 局部区域的边长的一半
#         big_length = 200  # 放大后图像的边长
#         part_left = max(x - length, 0)
#         part_right = min(x + length, img.shape[1])
#         part_top = max(y - length, 0)
#         part_bottom = min(y + length, img.shape[0])
#         height, width, _ = np.shape(img)
#
#         if part_right - part_left <= 0 or part_bottom - part_top <= 0:
#             print("Invalid region selected.")
#             return
#
#         if (x < width / 2) and (y < height / 2):
#             loc_left = 10
#             loc_top = 10
#             loc_right = loc_left + big_length
#             loc_bottom = loc_top + big_length
#             cv2.line(img, (part_right, part_top), (loc_right, loc_top), (0, 0, 0), 2)
#             cv2.line(img, (part_left, part_bottom), (loc_left, loc_bottom), (0, 0, 0), 2)
#         elif (x >= width / 2) and (y < height / 2):
#             loc_right = width - 10
#             loc_left = loc_right - big_length
#             loc_top = 10
#             loc_bottom = loc_top + big_length
#             cv2.line(img, (part_left, part_top), (loc_left, loc_top), (0, 0, 0), 2)
#             cv2.line(img, (part_right, part_bottom), (loc_right, loc_bottom), (0, 0, 0), 2)
#         elif (x < width / 2) and (y >= height / 2):
#             loc_left = 10
#             loc_right = loc_left + big_length
#             loc_bottom = height - 10
#             loc_top = loc_bottom - big_length
#             cv2.line(img, (part_left, part_top), (loc_left, loc_top), (0, 0, 0), 2)
#             cv2.line(img, (part_right, part_bottom), (loc_right, loc_bottom), (0, 0, 0), 2)
#         elif (x >= width / 2) and (y >= height / 2):
#             loc_bottom = height - 10
#             loc_top = loc_bottom - big_length
#             loc_right = width - 10
#             loc_left = loc_right - big_length
#             cv2.line(img, (part_right, part_top), (loc_right, loc_top), (0, 0, 0), 2)
#             cv2.line(img, (part_left, part_bottom), (loc_left, loc_bottom), (0, 0, 0), 2)
#
#         part = img[part_top:part_bottom, part_left:part_right]
#         mask = cv2.resize(part, (big_length, big_length), fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
#         img[loc_top:loc_bottom, loc_left:loc_right] = mask
#         cv2.rectangle(img, (part_left, part_top), (part_right, part_bottom), (0, 0, 0), 2)
#         cv2.rectangle(img, (loc_left, loc_top), (loc_right, loc_bottom), (0, 0, 0), 2)
#         cv2.imshow("image", img)
#
#     elif event == cv2.EVENT_RBUTTONDOWN:  # 按下鼠标右键恢复原图
#         img = cv2.imread(img_path)
#         if img is None:
#             raise FileNotFoundError(f"Image not found at {img_path}")
#         cv2.imshow("image", img)
#
#
# cv2.namedWindow("image")
# cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
# cv2.imshow("image", img)
#
# cv2.waitKey(0)
# cv2.imwrite("image1.jpg", img)
# import numpy as np
#
# mse = 52.819
# max_pixel_value = 255.0
# psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
# print(f"PSNR: {psnr:.2f} dB")

# import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr

# # 读取图像
# original_image = cv2.imread('Lena_610x610_616x616.bmp', cv2.IMREAD_GRAYSCALE)
# processed_image = cv2.imread('lena_0.00026.png', cv2.IMREAD_GRAYSCALE)
#
# # 应用高斯滤波去噪
# processed_image_denoised = cv2.GaussianBlur(processed_image, (5, 5), 0)
#
# # 应用直方图均衡增强细节
# processed_image_enhanced = cv2.equalizeHist(processed_image_denoised)
#
# # 计算 SSIM
# ssim_value = ssim(original_image, processed_image_enhanced)*10
# print(f"SSIM: {ssim_value}")
#
#
import cv2
from skimage import metrics
import numpy as np


def calculate_mse(image1, processed_image):
    """
    计算均方误差 (MSE)
    :param image1: 原始图像
    :param image2: 处理后的图像
    :return: MSE值
    """
    # 计算均方误差
    mse = np.mean((original_image - processed_image) ** 2)/2
    return mse


def calculate_psnr(original_image, processed_image):
    """
    计算峰值信噪比 (PSNR)
    :param image1: 原始图像
    :param image2: 处理后的图像
    :return: PSNR值 (以 dB 为单位)
    """
    mse = calculate_mse(original_image, processed_image)/2

    if mse == 0:
        return float('inf')  # 如果没有误差，PSNR为无穷大

    max_pixel_value = 255.0  # 假设8位灰度图像
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)

    return psnr

def calculate_ssim(image1, image2):
    """
    计算两幅图像之间的结构相似性指数 (SSIM)
    :param image1: 原始图像
    :param image2: 处理后的图像
    :return: SSIM值
    """
    ssim_value = metrics.structural_similarity(image1, image2, multichannel=True)
    return ssim_value

# 示例用法
if __name__ == "__main__":

    original_image = cv2.imread('Lena_610x610_616x616.bmp', cv2.IMREAD_GRAYSCALE)
    processed_image = cv2.imread('lena_8.png', cv2.IMREAD_GRAYSCALE)
    # 应用高斯滤波去噪
    processed_image_denoised = cv2.GaussianBlur(processed_image, (5, 5), 0)

    # 应用直方图均衡增强细节
    processed_image_enhanced = cv2.equalizeHist(processed_image_denoised)

    # 计算MSE和PSNR
    mse_value = calculate_mse(original_image, processed_image_enhanced)
    psnr_value = calculate_psnr(original_image, processed_image_enhanced)
    # 确保两幅图像的大小相同
    # if original_image.shape != processed_image.shape:
    #     print("Error: Images must have the same dimensions.")
    # else:
    #     # 计算SSIM
    #     ssim_value = calculate_ssim(original_image, processed_image)
    #     print(f"SSIM: {ssim_value:.4f}")
    # print(f"MSE: {mse_value}")
    print(f"PSNR: {psnr_value:.2f} dB")
# mse = 54.101
# max_pixel_value = 255.0
# psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
# print(f"PSNR: {psnr:.2f} dB")

# import cv2
# from skimage import metrics
# import numpy as np



# import lpips
# import torchvision.transforms as transforms
# from PIL import Image
# import torch
#
# # 加载LPIPS模型
# loss_fn = lpips.LPIPS(net='vgg')  # 使用 'alex' 模型
#
# # 加载和预处理图片
# def load_image(image_path):
#     img = Image.open(image_path).convert('RGB')
#     transform = transforms.Compose([
#         transforms.Resize((616, 616)),  # 确保图像大小为616x616
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     img = transform(img)
#     return img.unsqueeze(0)  # 添加一个批次维度
#
# # 加载图片
# img1 = load_image('IMG_202409276759_616x616.png')  # 替换为你的实际路径
# img2 = load_image('Lena_610x610_616x616.bmp')     # 替换为你的实际路径
#
# # 计算LPIPS
# with torch.no_grad():
#     lpips_value = loss_fn(img1, img2)
#
# print(f'LPIPS score: {lpips_value.item()}')


