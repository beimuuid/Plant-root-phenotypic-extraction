import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import shutil
import scipy.ndimage
from skimage.measure import label
import scipy.ndimage.morphology

def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    # loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    # # if test on whole image (Disitinctions-646), please uncomment this line
    loss = loss = np.sum(error_map ** 2) / (pred.shape[0] * pred.shape[1])

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    # loss = np.sum(error_map * (trimap == 128))

    # # if test on whole image (Disitinctions-646), please uncomment this line
    loss = np.sum(error_map)

    return loss / 1000, np.sum(trimap == 128) / 1000

def gauss(x, sigma):
    y = np.exp(-x ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))
    return y


def dgauss(x, sigma):
    y = -x * gauss(x, sigma) / (sigma ** 2)
    return y


def gaussgradient(im, sigma):
    epsilon = 1e-2
    halfsize = np.ceil(sigma * np.sqrt(-2 * np.log(np.sqrt(2 * np.pi) * sigma * epsilon))).astype(int)
    size = 2 * halfsize + 1
    hx = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            u = [i - halfsize, j - halfsize]
            hx[i, j] = gauss(u[0], sigma) * dgauss(u[1], sigma)

    hx = hx / np.sqrt(np.sum(np.abs(hx) * np.abs(hx)))
    hy = hx.transpose()

    gx = scipy.ndimage.convolve(im, hx, mode='nearest')
    gy = scipy.ndimage.convolve(im, hy, mode='nearest')

    return gx, gy

def compute_gradient_loss(pred, target, trimap):

    pred = pred / 255.0
    target = target / 255.0

    pred_x, pred_y = gaussgradient(pred, 1.4)
    target_x, target_y = gaussgradient(target, 1.4)

    pred_amp = np.sqrt(pred_x ** 2 + pred_y ** 2)
    target_amp = np.sqrt(target_x ** 2 + target_y ** 2)

    error_map = (pred_amp - target_amp) ** 2
    # loss = np.sum(error_map[trimap == 128])
    loss = np.sum(error_map)

    return loss / 1000.


def compute_connectivity_error(pred, target, trimap, step):
    pred = pred / 255.0
    target = target / 255.0
    h, w = pred.shape

    thresh_steps = list(np.arange(0, 1 + step, step))
    l_map = np.ones_like(pred, dtype=float) * -1
    for i in range(1, len(thresh_steps)):
        pred_alpha_thresh = (pred >= thresh_steps[i]).astype(int)
        target_alpha_thresh = (target >= thresh_steps[i]).astype(int)

        omega = getLargestCC(pred_alpha_thresh * target_alpha_thresh).astype(int)
        flag = ((l_map == -1) & (omega == 0)).astype(int)
        l_map[flag == 1] = thresh_steps[i - 1]

    l_map[l_map == -1] = 1

    pred_d = pred - l_map
    target_d = target - l_map
    pred_phi = 1 - pred_d * (pred_d >= 0.15).astype(int)
    target_phi = 1 - target_d * (target_d >= 0.15).astype(int)
    # loss = np.sum(np.abs(pred_phi - target_phi)[trimap == 128])
    loss = np.sum(np.abs(pred_phi - target_phi))

    return loss / 1000.

def getLargestCC(segmentation):
    labels = label(segmentation, connectivity=1)
    largestCC = labels == np.argmax(np.bincount(labels.flat))
    return largestCC


def evaluate(args):
    img_names = []
    mse_loss_unknown = []
    sad_loss_unknown = []
    grad_loss_unknown = []
    conn_loss_unknown = []


    bad_case = []

    for i, img in tqdm(enumerate(os.listdir(args.label_dir))):

        if not((os.path.isfile(os.path.join(args.pred_dir, img)) and
                os.path.isfile(os.path.join(args.label_dir, img)) and
                os.path.isfile(os.path.join(args.trimap_dir, img)))):
            print('[{}/{}] "{}" skipping'.format(i, len(os.listdir(args.label_dir)), img))
            continue

        pred = cv2.imread(os.path.join(args.pred_dir, img), 0).astype(np.float32)
        label = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
        trimap = cv2.imread(os.path.join(args.trimap_dir, img), 0).astype(np.float32)

        # calculate loss
        mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
        sad_loss_unknown_ = compute_sad_loss(pred, label, trimap)[0]
        conn_loss_unknown_ = compute_connectivity_error(pred, label, trimap, 0.1)
        grad_loss_unknown_ = compute_gradient_loss(pred, label, trimap)


        # save for average
        img_names.append(img)

        mse_loss_unknown.append(mse_loss_unknown_)  # mean l2 loss per unknown pixel
        sad_loss_unknown.append(sad_loss_unknown_)  # l1 loss on unknown area
        conn_loss_unknown.append(conn_loss_unknown_)  # l1 loss on unknown area
        grad_loss_unknown.append(grad_loss_unknown_)  # l1 loss on unknown area

    print('* Whole Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean(),
          'Conn:', np.array(conn_loss_unknown).mean(), 'Grad:', np.array(grad_loss_unknown).mean())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, required=True, help="output dir")
    parser.add_argument('--label-dir', type=str, default='', help="GT alpha dir")
    parser.add_argument('--trimap-dir', type=str, default='', help="trimap dir")

    args = parser.parse_args()

    evaluate(args)