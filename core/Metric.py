from PIL import Image
import numpy as np
import math
import os


IMAGE_FORMAT = '.png'
INFER_DIR = './prediction_dir/'
LABEL_DIR = './label_dir/'


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


if __name__ == '__main__':
    import argparse
    import glob
    import os

    parser = argparse.ArgumentParser(description='Evaluate change detection predictions.')
    parser.add_argument('--pred_dir', type=str, required=True, help='Directory with predicted mask images (e.g., _pred_change.png)')
    parser.add_argument('--label_dir', type=str, required=True, help='Directory with ground truth mask images')
    parser.add_argument('--suffix', type=str, default='_pred_change.png', help='Suffix of prediction files to match (default: _pred_change.png)')
    parser.add_argument('--format', type=str, default='.png', help='Image file format (default: .png)')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes (default: 2)')
    args = parser.parse_args()

    pred_files = sorted(glob.glob(os.path.join(args.pred_dir, f'*{args.suffix}')))
    name_list = [os.path.basename(f).replace(args.suffix, '') for f in pred_files]

    hist = np.zeros((args.num_class, args.num_class))
    all_label_values = set()
    for idx, name in enumerate(name_list):
        infer_file = os.path.join(args.pred_dir, f'{name}{args.suffix}')
        label_file = os.path.join(args.label_dir, f'{name}{args.format}')
        if not os.path.exists(infer_file):
            print(f"Prediction not found: {infer_file}")
            continue
        if not os.path.exists(label_file):
            print(f"Label not found: {label_file}")
            continue
        infer = Image.open(infer_file)
        label = Image.open(label_file)
        if label.mode != 'L':
            label = label.convert('L')
        infer_array = np.array(infer)
        label_array = np.array(label)
        # Remap 255 -> 1 for binary masks
        if label_array.max() == 255 and args.num_class == 2:
            label_array = (label_array == 255).astype(np.uint8)
        # Remap color to class index for multiclass masks (update mapping as needed)
        if args.num_class > 2:
            color_to_class = {38: 0, 75: 1, 128: 2, 150: 3, 200: 4, 255: 5}  # <-- Update this mapping for your dataset!
            label_indices = np.zeros_like(label_array)
            for color, idx_c in color_to_class.items():
                label_indices[label_array == color] = idx_c
            label_array = label_indices
        # Debug: print unique values
        print(f"[DEBUG] {infer_file} unique pred: {np.unique(infer_array)}")
        print(f"[DEBUG] {label_file} unique label: {np.unique(label_array)}")
        all_label_values.update(np.unique(label_array))
        # Debug: save first 5 prediction/label pairs
        if idx < 5:
            from PIL import Image as PILImage
            PILImage.fromarray(infer_array.astype(np.uint8)).save(f'debug_pred_{idx}.png')
            PILImage.fromarray(label_array.astype(np.uint8)).save(f'debug_label_{idx}.png')
        if infer_array.shape != label_array.shape:
            print(f"[WARNING] Shape mismatch: {infer_file} {infer_array.shape} vs {label_file} {label_array.shape} -- Skipping.")
            continue
        # Check for out-of-range or negative values
        if (infer_array.max() >= args.num_class or label_array.max() >= args.num_class or
            infer_array.min() < 0 or label_array.min() < 0):
            print(f"[WARNING] Out-of-range or negative class in {infer_file} or {label_file}. Unique values: pred={np.unique(infer_array)}, label={np.unique(label_array)} -- Skipping.")
            continue
        hist += get_hist(infer_array, label_array, args.num_class)
    print("All unique label values in dataset:", all_label_values)

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

    print('Mean IoU = %.5f' % IoU_mean)
    print('Sek = %.5f' % Sek)

