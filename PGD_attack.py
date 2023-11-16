import torch
from set_up import image_path, image_save, model, transform, inv_transform
from PIL import Image


def cw_loss(output, original_img, adv_img, targeted_t, k=torch.tensor(7)):
    A = 0.5 * (torch.tanh(adv_img) + 1)
    L_r = torch.sqrt(torch.sum((A - original_img)**2))
    out = output.clone().squeeze()
    Z_t = out[targeted_t].clone()
    out[targeted_t] = float('-inf')
    max_i = torch.max(out)
    L_t = torch.max(-k, max_i - Z_t)
    return L_r + L_t


def pgd_attack(model, original_image,  target_label, epsilon, alpha, num_iter):
    # 输入:
    # LCNN_model: 目标神经网络模型
    # original_image: 原始输入图像
    # true_label: 真实标签
    # epsilon: 对抗样本的扰动幅度
    # alpha: 梯度步长
    # num_iter: 迭代次数

    # 初始化对抗样本
    adv_image = original_image.clone()
    adv_image.requires_grad = True
    # 开始迭代
    for i in range(num_iter):
        # 计算模型输出和损失
        output = model(adv_image)
        # loss = torch.nn.functional.cross_entropy(output, true_label)
        loss = cw_loss(output, original_image, adv_image, target_label)
        print(f'iterator {i}, loss: {loss.item()}')
        # 计算损失相对于输入的梯度
        gradient = torch.autograd.grad(loss, adv_image, allow_unused=True)[0]
        # 生成对抗样本
        # torch.nn.utils.clip_grad_norm_(gradient, max_norm=1.0)
        adv_image = adv_image - alpha * torch.sign(gradient)
        print(torch.max(gradient))

        # 对抗样本裁剪到合理范围内
        adv_image = torch.max(torch.min(adv_image, original_image + epsilon), original_image - epsilon)
        # 将对抗样本投影到合理的输入空间内
        adv_image = torch.clamp(adv_image, -1.5, 2)

    return adv_image


if __name__ == '__main__':
    image = Image.open(image_path).convert('RGB')
    image_size = image.size
    print(image_size)
    # 使用 PGD 攻击
    # 281: 虎斑猫, 283：波斯猫; 404: 大型客机, 405: 飞艇; 259: 博美犬（波美拉尼亚小狗）, 260: 松狮犬
    # 前者是测试样例的原始标签， 后者是选定的目标攻击标签
    adversarial_example = pgd_attack(model, transform(image).unsqueeze(0), torch.tensor(260), 0.01, 0.001, 100)
    image = inv_transform(adversarial_example.squeeze())
    image.save(image_save)
    print(f'image is saved in {image_save}')
    # image.show()
