from set_up import model, image_save,image_path, transform, image_name
import torch
from PIL import Image


def test_acc():
    if image_name == 'dog':
        ori = 259
        adv = 260
    elif image_name == 'airplane':
        ori = 404
        adv = 405
    elif image_name == 'cat':
        ori = 281
        adv = 283
    else:
        return
    model.eval()
    original_count = 0
    adv_count = 0
    range_ = 100
    for i in range(range_):
        out = model(transform(Image.open(image_path).convert('RGB')).unsqueeze(0))
        _, predicted = torch.max(out.data, 1)
        if predicted[0] == torch.tensor(ori):
            original_count += 1

        out = model(transform(Image.open(image_save).convert('RGB')).unsqueeze(0))
        _, predicted = torch.max(out.data, 1)
        if predicted[0] == torch.tensor(ori):
            print('the adversarial image failed')
            break
        elif predicted[0] == torch.tensor(adv):
            adv_count += 1
        else:
            print(predicted[0])

    print(f'original_acc: {original_count/range_ * 100}%  adv_acc: {adv_count/range_ * 100}%')


if __name__ == '__main__':
    test_acc()
