import scipy.io as sio
import numpy as np
import torch
import os
import time
from swinir.network_swinir import SwinIR as net


def soft(x, T):
    '''
    function
    y = soft(x, T)

    T = T + eps;
    y = max(abs(x) - T, 0);
    y = y. / (y + T). * x;
    '''
    eps = 2.2204e-16
    T = T + eps

    y = torch.nn.functional.relu(x.abs() - T)
    y = y / (y + T) * x
    return y


def element2pixel(z, padp, ind):
    '''
    padding element 2 pixel coordinates
    z [N,1] =[3801,1]
    out:padp [L,L]=[128,128]
    '''
    L = padp.shape[0]
    padp = padp.reshape(-1, 1)
    padp[ind, 0] = z
    padp = padp.reshape([L, L]).t()
    return padp


def pixel2element(padp, ind):
    z = padp.t().reshape([-1, 1])[ind, 0]
    return z


def test( path, file):
    data = sio.loadmat(os.path.join(path, file))
    ind = data['ind']
    if path == 'data':
        dv = data['dv']
        gt = data['reference']
        width = 39
        lam = 0.15
        beta = 0.1
        scale_v=1
    elif path == 'data2017':
        dv = data['vi'] - data['vh']
        dv = dv * np.sqrt(2.6037e-06 ** 2 / np.mean(dv ** 2))
        img=data['img']
        img=img.reshape([-1,1])
        gt = img[ind]
        width = 39
        lam = 0.15
        beta = 0.1
        scale_v=1
    elif path == 'datalung':
        dv = data['dv']
        gt = data['reference']
        max_ = np.max(np.abs(gt))
        print(max_)
        gt=gt/max_
        scale_v = np.sqrt(2.6037e-06 ** 2 / np.mean(dv ** 2))
        dv = dv * scale_v
        width=41
        ind=ind.T
        lam = 0.1
        beta = 10

    L = data['L']
    try:
        Jpixel = data['Jpixel']
    except:
        Jpixel = data['Y']


    dv = torch.from_numpy(dv).float().to(device)
    Jpixel = torch.from_numpy(Jpixel).to(device).float()
    _, N = Jpixel.shape[0], Jpixel.shape[1]
    I = torch.eye(N).float().to(device)
    L = torch.from_numpy(L).float().to(device)

    norm_y = dv.square().mean().sqrt()
    Jpixel = Jpixel / norm_y
    dv = dv / norm_y
    Ab = Jpixel.T @ dv

    [UF, SF, _] = torch.linalg.svd(Jpixel.T @ Jpixel)

    rho = 10 * lam + 0.01  # e-12
    IF = UF @ torch.diag(1. / (SF + rho)) @ UF.T

    x = IF @ Jpixel.T @ dv
    z = IF @ Jpixel.T @ dv

    ind = np.double(ind)
    pred = torch.zeros(width,width).float().to(device)

    print('start iter')

    d = torch.zeros_like(x)

    i = 0
    res_p = torch.tensor(torch.inf)
    res_d = torch.tensor(torch.inf)
    tol = 1e-4
    tol1 = np.sqrt(N) * tol
    tol2 = np.sqrt(N) * tol
    num_iter = 100
    mu_changed = 0
    JTJ = Jpixel.t() @ Jpixel
    IF = torch.linalg.inv(JTJ + rho * I + beta * L)

    with torch.no_grad():
        while i < num_iter and (torch.abs(res_p) > tol1 or torch.abs(res_d) > tol2):
            if i % 10 == 0:
                z0 = z

            x = IF @ (Ab + rho * (z + d))
            z = soft(x - d, lam / rho)

            ## plugin start
            ## PnP
            pred = element2pixel(z, pred, ind)

            pred_min = pred.min()
            scale = pred.max() - pred.min()

            pred = (pred - pred_min) / scale
            pred = pred.unsqueeze(0).unsqueeze(0).repeat(1, 1, 1, 1)

            window_size = 8
            _, _, h_old, w_old = pred.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            pred = torch.cat([pred, torch.flip(pred, [2])], 2)[:, :, :h_old + h_pad, :]
            pred = torch.cat([pred, torch.flip(pred, [3])], 3)[:, :, :, :w_old + w_pad]

            pred = model(pred)
            pred = pred[..., :h_old, :w_old]
            pred = (pred[0, 0] * scale + pred_min)

            z = pixel2element(pred, ind)
            ## PnP

            d = d - (x - z)

            if i % 10 == 0:
                res_p = torch.norm(x - z)
                res_d = rho * torch.norm(z - z0)
                print(' i = {}, res_p = {:.6f}, res_d = {:.6f}'.format(i, res_p, res_d))

                # update  mu
                if res_p > 10 * res_d:
                    rho = rho * 2
                    d = d / 2
                    mu_changed = 1
                elif res_d > 10 * res_p:
                    rho = rho / 2
                    d = d * 2
                    mu_changed = 1

                if mu_changed:
                    IF = torch.linalg.inv(JTJ + rho * I + beta * L)
                    mu_changed = 0
            i = i + 1


    z = z.cpu().numpy()/scale_v
    if path == 'datalung':z =z /np.max(np.abs(z))
    err = np.sqrt(np.mean((z - gt) ** 2))


    # os.makedirs('./out1',exist_ok=True)
    # sio.savemat('./out1/'+file, {'pred': z})
    return err


import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-data", "--data",  # 可选参数，可以用 -o 或 --output
    default="datalung",  # 默认值
    help="dataset"
)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = './swinir/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth'  # 15db 0.1472

model = net(upscale=1, in_chans=1, img_size=128, window_size=8,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='', resi_connection='1conv')

param_key_g = 'params'
pretrained_model = torch.load(model_path)
model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
                      strict=True)

model.eval()
model = model.to(device)


rmse = []
tic = time.time()
path = args.data
file = os.listdir(path)
for i in range(len(file)):
    print('run: ',file[i])
    a = test( path, file[i])
    rmse.append(a)
toc = time.time()
print(f'RMSE: {np.mean(rmse):.4f} | Time: {toc - tic:.2f}s')
