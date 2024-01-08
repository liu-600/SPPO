import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from glob import glob
import nibabel as nib

# load data
brats = sorted(glob('BraTs/*.gz'))
brats_gt = sorted(glob('BraTsseg/*.gz'))

brats_tra = []
brats_label = []
for i, j in zip(brats, brats_gt):
    brats_np = nib.load(i).get_fdata()
    brats_gt_np = nib.load(j).get_fdata()

    assert brats_gt_np.shape == brats_np.shape

    z = brats_np.shape[-1]
    for zi in range(z):
        brats_tra.append(brats_np[..., zi].flatten())  # 横断层拉直，维度=240x240=57600
        if brats_gt_np[..., zi].any():
            brats_label.append(1)
        else:
            brats_label.append(0)

brats_array = np.array(brats_tra, dtype='uint8')  # [6200, 57600]
brats_label_array = np.array(brats_label, dtype='uint8')  # [6200]

tsne = manifold.TSNE(n_components=2, init='pca', random_state=42).fit_transform(brats_array)
# tsne 归一化， 这一步可做可不做
x_min, x_max = tsne.min(0), tsne.max(0)
tsne_norm = (tsne - x_min) / (x_max - x_min)
normal_idxs = (brats_label_array == 0)
abnorm_idxs = (brats_label_array == 1)
tsne_normal = tsne_norm[normal_idxs]
tsne_abnormal = tsne_norm[abnorm_idxs]
plt.figure(figsize=(8, 8))
plt.scatter(tsne_normal[:, 0], tsne_normal[:, 1], 1, color='red', label='Healthy slices')
# tsne_normal[i, 0]为横坐标，X_norm[i, 1]为纵坐标，1为散点图的面积， color给每个类别设定颜色
plt.scatter(tsne_abnormal[:, 0], tsne_abnormal[:, 1], 1, color='green', label='Anomalous slices')
plt.legend(loc='upper left')
plt.show()
