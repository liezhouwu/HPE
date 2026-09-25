"""
mmfi_wifi.metrics — 3D 姿态评估指标 (官方口径 + 骨盆对齐口径)
================================================================
MM-Fi 论文 (NeurIPS 2023) 的文字定义与官方代码实现不一致, 两种口径都提供:

【官方代码口径 — Table 3 数字的真实来源, 复现对标用】
  官方 mmfi_lib/evaluate.py 的 calulate_error:
    MPJPE    = mean(||pred - gt||)  绝对相机系坐标, **无任何对齐**
    PA-MPJPE = Procrustes 相似变换 (平移+旋转+缩放) 对齐后的 MPJPE
  证据: Table 3 WiFi S3 MPJPE 爆炸(367.8) 而 PA 平坦(~121), 是"全局平移误差"
  的特征, 在骨盆对齐口径下不可能出现 (实测 E04 骨盆位置相对训练场景平移
  239-326mm, 用训练均值绝对姿态作常数预测即得 S1=242.6/S2=258.9/S3=349.3)。

【论文文字口径 — 骨盆对齐, 诊断用】
  "MPJPE measures ... after aligning the pelvis of the estimated and true
   3D pose."  即先把预测和 GT 都平移到骨盆原点再算欧氏距离。

骨盆定义: 17 关键点 COCO 布局中 L_hip=11, R_hip=12 的中点。
"""

import numpy as np

# COCO-17 关键点名称 (索引即 MM-Fi ground_truth.npy 的顺序)
KP_NAMES = ['nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
            'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
            'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
            'L_knee', 'R_knee', 'L_ankle', 'R_ankle']

PELVIS_JOINTS = (11, 12)  # L_hip, R_hip


def pelvis_of(kps):
    """kps: (..., 17, 3) -> pelvis: (..., 3)"""
    return 0.5 * (kps[..., PELVIS_JOINTS[0], :] + kps[..., PELVIS_JOINTS[1], :])


def to_root_relative(kps):
    """绝对 3D 坐标 -> 骨盆相对坐标。kps: (..., 17, 3)"""
    return kps - pelvis_of(kps)[..., None, :]


def compute_similarity_transform(X, Y, compute_optimal_scale=True):
    """
    Procrustes 相似变换 (ported from MATLAB procrustes)。
    与 mmfi_lib.evaluate.compute_similarity_transform 一致,
    增加了零范数保护。X: 目标 (N,3), Y: 待对齐 (N,3)。
    返回 (d, Z, T, b, c): Z 为对齐后的 Y。
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    if normX < 1e-12 or normY < 1e-12:
        # 退化帧: 不做旋转/缩放, 仅平移对齐
        return 0.0, Y0 + muX, np.eye(X.shape[1]), 1.0, muX - muY

    X0 = X0 / normX
    Y0 = Y0 / normY

    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)
    return d, Z, T, b, c


def mpjpe_official_mm(pred, gt):
    """
    官方 calulate_error 的 MPJPE: 绝对坐标直接欧氏距离, **无任何对齐**, 返回 mm。
    pred/gt: (N, 17, 3) numpy, 相机系绝对坐标 (米)。
    这是 Table 3 数字的口径 — 全局位置误差 (含场景平移) 全部计入。
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    err = np.sqrt(np.sum(np.square(pred - gt), axis=2))  # (N, 17)
    return float(err.mean() * 1000.0)


def pa_mpjpe_official_mm(pred, gt):
    """
    官方 calulate_error 的 PA-MPJPE (Procrustes 含缩放), 返回 mm。
    与 pa_mpjpe_mm 的数学完全一致 (Procrustes 平移不变, 根相对与否无关),
    单独命名仅为了调用处口径自明。
    """
    return pa_mpjpe_mm(pred, gt, already_root_relative=True)


def evaluate_pose_official(pred, gt):
    """官方口径: 返回 (mpjpe_official_mm, pa_mpjpe_official_mm)。绝对坐标输入。"""
    return mpjpe_official_mm(pred, gt), pa_mpjpe_official_mm(pred, gt)


def mpjpe_mm(pred, gt, already_root_relative=False):
    """
    论文 MPJPE (骨盆对齐 3D 欧氏距离均值), 返回 mm。
    pred/gt: (N, 17, 3) numpy。
    already_root_relative: 输入是否已是骨盆相对坐标。
      默认 False —— 即使训练目标是根相对, 模型输出的 pelvis 不保证在原点
      (实测偏差可达 ~200mm), 必须重新对齐预测的骨盆才能得到论文定义的 MPJPE。
      GT 已是根相对时, to_root_relative 是 no-op, 无副作用。
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if not already_root_relative:
        pred = to_root_relative(pred)
        gt = to_root_relative(gt)
    err = np.sqrt(np.sum(np.square(pred - gt), axis=2))  # (N, 17)
    return float(err.mean() * 1000.0)


def pa_mpjpe_mm(pred, gt, already_root_relative=False):
    """
    论文 PA-MPJPE (Procrustes 含缩放对齐后的 MPJPE), 返回 mm。
    pred/gt: (N, 17, 3) numpy。Procrustes 平移不变, 与是否根相对无关。
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if not already_root_relative:
        pred = to_root_relative(pred)
        gt = to_root_relative(gt)
    N = pred.shape[0]
    errs = np.zeros((N, pred.shape[1]))
    for n in range(N):
        _, Z, _, _, _ = compute_similarity_transform(gt[n], pred[n],
                                                     compute_optimal_scale=True)
        errs[n] = np.sqrt(np.sum(np.square(Z - gt[n]), axis=1))
    return float(errs.mean() * 1000.0)


def evaluate_pose(pred, gt, already_root_relative=False):
    """返回 (mpjpe_mm, pa_mpjpe_mm)。"""
    return (mpjpe_mm(pred, gt, already_root_relative),
            pa_mpjpe_mm(pred, gt, already_root_relative))


def per_joint_mpjpe_mm(pred, gt, already_root_relative=False):
    """逐关节 MPJPE, 返回 {name: mm}。pred/gt: (N, 17, 3)"""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if not already_root_relative:
        pred = to_root_relative(pred)
        gt = to_root_relative(gt)
    err = np.sqrt(np.sum(np.square(pred - gt), axis=2))  # (N, 17)
    return {KP_NAMES[j]: float(err[:, j].mean() * 1000.0) for j in range(17)}
