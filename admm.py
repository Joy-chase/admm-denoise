import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import cg, LinearOperator
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import matplotlib

matplotlib.use('TkAgg')

# 读取图像
def read_image(filepath):
    img = Image.open(filepath).convert("L")  # 转换为灰度图像
    return np.array(img)

# 添加泊松噪声
def add_poisson_noise(image):
    noisy = np.random.poisson(image)
    return np.clip(noisy, 0, 255).astype(np.uint8)

# 软阈值函数，用于 z 子问题的解决
def soft_thresholding(x, lambd):
    return np.sign(x) * np.maximum(np.abs(x) - lambd, 0)

# 定义线性运算符 Ax
def create_linear_operator(size, rho):
    def Ax(x):
        return x + rho * x

    return LinearOperator((size, size), matvec=Ax)

# ADMM 迭代求解
def admm_tv_denoising(y, lambda_tv=1.0, rho=1.0, num_iterations=1000):  # 默认迭代次数为1000
    # 初始化变量
    x = np.copy(y)
    z = np.copy(y)
    u = np.zeros_like(y)

    size = y.size
    A = create_linear_operator(size, rho)

    for _ in range(num_iterations):
        # x 子问题更新，使用共轭梯度法
        b = y + rho * (z - u)
        x, _ = cg(A, b.flatten(), x0=x.flatten())
        x = x.reshape(y.shape)

        # z 子问题更新，使用软阈值处理 TV 正则化
        z = soft_thresholding(x + u, lambda_tv / rho)

        # u 更新（拉格朗日乘子更新）
        u = u + x - z

    return x

# 评价去噪效果
def evaluate_denoising(original_img, denoised_img):
    # 确保图像具有相同的数据类型
    original_img = original_img.astype(np.float64)
    denoised_img = denoised_img.astype(np.float64)

    # 指定 data_range
    psnr_value = psnr(original_img, denoised_img, data_range=original_img.max() - original_img.min())
    ssim_value = ssim(original_img, denoised_img, data_range=original_img.max() - original_img.min())
    return psnr_value, ssim_value

# 图像路径（本地路径）
image_path = r"D:\桌面\ct去噪\ceshi\10.jpg"  # 替换为你的图像路径

# 读取图像
clean_img = read_image(image_path)

# 添加泊松噪声
noisy_img = add_poisson_noise(clean_img)

# 参数范围
lambda_tv_values = np.arange(0.1, 0.4, 0.1)  # 0.1 到 2.0，步长为 0.1
rho_values = np.arange(0.01, 2.01, 0.1)  # 0.01 到 2.0，步长为 0.1
num_iterations = 1000  # 设置迭代次数为1000

# 存储最佳结果
best_psnr = 0
best_ssim = 0
best_params = {}
best_denoised_img = None

# 遍历所有参数组合
for lambda_tv in lambda_tv_values:
    for rho in rho_values:
        denoised_img = admm_tv_denoising(noisy_img, lambda_tv, rho, num_iterations)
        psnr_value, ssim_value = evaluate_denoising(clean_img, denoised_img)

        print(f"Params: λ={lambda_tv:.1f}, ρ={rho:.2f} -> PSNR: {psnr_value:.2f}, SSIM: {ssim_value:.4f}")

        if psnr_value > best_psnr:
            best_psnr = psnr_value
            best_ssim = ssim_value
            best_params = {'lambda_tv': lambda_tv, 'rho': rho}
            best_denoised_img = denoised_img

# 显示结果
print(f"Best Params: λ={best_params['lambda_tv']}, ρ={best_params['rho']} -> PSNR: {best_psnr:.2f}, SSIM: {best_ssim:.4f}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(clean_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Noisy Image')
plt.imshow(noisy_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Best Denoised Image')
plt.imshow(best_denoised_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()