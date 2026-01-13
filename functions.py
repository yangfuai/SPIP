# -*- coding: utf-8 -*-
import numpy as np
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.optimize import least_squares
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#翼型读取函数
def read_airfoil_data(file_path):
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"翼型文件 {file_path} 未找到")

    data_lines = [line.strip() for line in lines[1:] if line.strip()]
    x_coords = []
    y_coords = []
    for line in data_lines:
        try:
            x, y = map(float, line.split())
            x_coords.append(x)
            y_coords.append(y)
        except ValueError:
            continue
    if len(x_coords) < 20:
        raise ValueError(f"翼型数据点数过少 ({len(x_coords)})，建议至少50个点")
    return {'x': x_coords, 'y': y_coords}

#翼型预处理函数
def preprocess_airfoil_data(coordinates, normalize=False, tolerance=1e-6):
    """
    加载并处理翼型坐标，能鲁棒处理两种顺序的坐标
    1. 尾缘 → 上翼面 → 前缘 → 下翼面 → 尾缘（常见于 UIUC）
    2. 尾缘 → 下翼面 → 前缘 → 上翼面 → 尾缘（常见于 XFOIL、Profili）

    返回
    ----
    UpX_interp, UpY_interp : list
        上翼面插值后坐标（从前缘 x=0 到尾缘 x=1）
    LowX_interp, LowY_interp : list
        下翼面插值后坐标（从前缘 x=0 到尾缘 x=1）
    """

    # ====================== 辅助函数1：清理重复点 ======================
    def clean_coordinates(x, y):
        """
        把坐标按 x 从小到大排序，并去除 x 完全相同的重复点（防止插值出错）
        """
        coords = np.array(sorted(zip(x, y), key=lambda p: p[0]))  # 按 x 排序
        x_clean, y_clean = coords[:, 0], coords[:, 1]
        # 相邻 x 差值 > 1e-10 才保留（去除完全相同的点）
        mask = np.diff(x_clean, prepend=x_clean[0] - 1) > 1e-10
        return x_clean[mask], y_clean[mask]

    # ====================== 辅助函数2：余弦分布插值 ======================
    def cosine_interpolation(x_orig, y_orig):
        """
        使用余弦分布（前缘密集、尾缘稀疏）
        """
        x_orig, y_orig = clean_coordinates(x_orig, y_orig)
        if len(x_orig) < 2:
            raise ValueError("插值点数不足")
        x_min, x_max = np.min(x_orig), np.max(x_orig)
        if x_max <= x_min:
            raise ValueError("x 范围无效，无法插值")
        # 归一化到 [0,1]
        x_norm = (x_orig - x_min) / (x_max - x_min)
        # 余弦分布：t 从 0→π，对应 x 从 0→1，前缘更密集
        t = np.linspace(0, np.pi, 100)
        x_new = (1 - np.cos(t)) / 2  # 余弦分布
        interp = interp1d(x_norm, y_orig)
        y_new = interp(x_new)
        return x_new.tolist(), y_new.tolist()

    # ============================== 主处理流程 ==============================

    # 1. 读取原始坐标，转成 numpy 数组
    X = np.array(coordinates['x'], dtype=float)
    Y = np.array(coordinates['y'], dtype=float)
    if len(X) != len(Y):
        raise ValueError("X 和 Y 坐标的长度不匹配")
    if len(X) < 10:
        raise ValueError(f"翼型数据点数过少 ({len(X)})，建议至少10个点")
    # 2. 自动识别前缘（x最小）和尾缘（x最大）
    x_min, x_max = np.min(X), np.max(X)
    min_x_idx = np.where(np.abs(X - x_min) <= tolerance)[0]  # 可能有多个点在同一x
    max_x_idx = np.where(np.abs(X - x_max) <= tolerance)[0]

    # 取平均值，得到更精确的前/尾缘坐标
    leading_edge_x = np.mean(X[min_x_idx])
    leading_edge_y = np.mean(Y[min_x_idx])
    trailing_edge_x = np.mean(X[max_x_idx])
    trailing_edge_y = np.mean(Y[max_x_idx])

    chord = trailing_edge_x - leading_edge_x
    if chord <= 0:
        raise ValueError("弦长为零或负值，无法归一化")

    # 3. 平移：把前缘移到原点 (0,0)
    X = X - leading_edge_x
    Y = Y - leading_edge_y

    # 4. 归一化 + 旋转（让弦线在 x 轴上）
    if normalize:
        # 计算弦线与 x 轴的夹角
        theta = np.arctan2(trailing_edge_y, trailing_edge_x)
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        # 旋转坐标系
        X_rot = X * cos_t - Y * sin_t
        Y_rot = X * sin_t + Y * cos_t
        # 再除以弦长，得到弦长=1
        X, Y = X_rot / chord, Y_rot / chord
    else:
        # 只归一化弦长，不旋转
        X, Y = X / chord, Y / chord

    # 5. 找到归一化后前缘点（x≈0）的索引
    le_idx = np.argmin(np.abs(X))  # x 最接近 0 的点
    if np.sum(np.abs(X - X[le_idx]) < 1e-8) > 1:  # 如果有多个点 x 完全一样
        candidates = np.where(np.abs(X - X[le_idx]) < 1e-8)[0]
        le_idx = candidates[np.argmin(np.abs(Y[candidates]))]  # 取 y 最接近 0 的

    # 6. 自动判断上/下翼面（支持两种顺序）
    # 取前缘左右各一小段，看哪边 y 值更大 → 上翼面
    left_part = Y[max(0, le_idx - 5):le_idx + 1]  # 前缘左边
    right_part = Y[le_idx:min(len(Y), le_idx + 6)]  # 前缘右边
    left_mean_y = np.mean(left_part) if len(left_part) > 0 else 0
    right_mean_y = np.mean(right_part) if len(right_part) > 0 else 0

    if left_mean_y > right_mean_y:  # 情况1：左段是上翼面
        upper_segment = Y[:le_idx + 1][::-1]  # 尾缘→前缘（上）
        lower_segment = Y[le_idx:]  # 前缘→尾缘（下）
        upper_x = X[:le_idx + 1][::-1]
        lower_x = X[le_idx:]
    else:  # 情况2：右段是上翼面
        upper_segment = Y[le_idx:][::-1]  # 尾缘→前缘（上）
        lower_segment = Y[:le_idx + 1]  # 前缘→尾缘（下）
        upper_x = X[le_idx:][::-1]
        lower_x = X[:le_idx + 1]

    # 7. 清理上/下翼面各自的重复点
    upper_x, upper_y = clean_coordinates(upper_x, upper_segment)
    lower_x, lower_y = clean_coordinates(lower_x, lower_segment)

    # 8. 余弦插值
    # 插值前把上翼面反转成“前缘→尾缘”
    UpX_interp, UpY_interp = cosine_interpolation(upper_x[::-1], upper_y[::-1])
    LowX_interp, LowY_interp = cosine_interpolation(lower_x, lower_y)
    # 由于余弦分布，x=0 是第一个点，x=1 是最后一个点
    print("x=0 (前缘) 坐标:")
    print(f"  上翼面: (x={UpX_interp[0]:.6f}, y={UpY_interp[0]:.6f})")
    print(f"  下翼面: (x={LowX_interp[0]:.6f}, y={LowY_interp[0]:.6f})")
    print("x=1 (尾缘) 坐标:")
    print(f"  上翼面: (x={UpX_interp[-1]:.6f}, y={UpY_interp[-1]:.6f})")
    print(f"  下翼面: (x={LowX_interp[-1]:.6f}, y={LowY_interp[-1]:.6f})")

    # 返回最终结果：上翼面 + 下翼面（都是从前缘 x=0 开始）
    return UpX_interp, UpY_interp, LowX_interp, LowY_interp

#翼型几何特征计算函数
def compute_airfoil_geometrical_feature(UpX, UpY, LowX, LowY):
    """
    从归一化后的上/下翼面坐标提取 14 个经典几何特征

    输入：
        UpX, UpY  → 上翼面坐标（从前缘 x=0 到尾缘 x=1）
        LowX, LowY → 下翼面坐标（从前缘 x=0 到尾缘 x=1）
    输出：14 个几何特征
    """
    # 转成 numpy 数组
    UpX, UpY = np.array(UpX), np.array(UpY)
    LowX, LowY = np.array(LowX), np.array(LowY)
    x = UpX  # 上、下翼面 x 完全一致,所以使用UpX作为横坐标

    # ==================== 1. 对称性判断 ====================
    # 鲁棒性处理对称翼型
    symmetry_error = np.mean(np.abs(UpY + LowY))
    is_symmetric = symmetry_error < 1e-3

    # ==================== 2. 中弧线 & 厚度分布 ====================
    camber = (UpY + LowY) / 2  # 中弧线
    thick = UpY - LowY  # 厚度分布

    # ==================== 3. 基本几何量 ====================
    C = float(np.max(camber))  # 最大相对弯度
    XC = float(x[np.argmax(camber)])  # 最大弯度位置
    T = float(np.max(thick))  # 最大相对厚度
    XT = float(x[np.argmax(thick)])  # 最大厚度位置

    z_TE = float(camber[-1])  # 尾缘中弧线高度
    delta_Z_TE = float(thick[-1])  # 尾缘开口厚度

    XC = np.clip(XC, 0.01, 0.99)
    XT = np.clip(XT, 0.01, 0.99)

    # 对称翼型特殊处理
    if np.mean(np.abs(UpY + LowY)) < 1e-3:
        C, XC, z_TE = 0.0, 0.5, 0.0

    # ==================== 4. 角度参数 ====================
    n_p = max(10, int(len(x) * 0.01))  # 这个值同时用于前缘和尾缘
    # ===== 尾缘角度 alpha_TE（线性拟合）=====
    m_TE = np.polyfit(x[-n_p:], camber[-n_p:], 1)[0]  # 中弧线尾缘斜率
    alpha_TE = np.arctan(m_TE)
    m_LE = np.polyfit(x[:n_p], camber[:n_p], 1)[0]  # 中弧线前缘斜率
    alpha_LE = np.arctan(m_LE)

    m_up_te = np.polyfit(x[-n_p:], UpY[-n_p:], 1)[0]
    m_lo_te = np.polyfit(x[-n_p:], LowY[-n_p:], 1)[0]
    beta_TE = abs(np.arctan(m_up_te) - np.arctan(m_lo_te))

    # ==================== 5. 曲率与曲率变化率 ====================
    """
    ###翼型最大厚度、弯度处一阶导数为0,所以曲率在该处等于该处的二阶导数，曲率变化率等于该点的三阶导数####
    对于任意平面曲线 y = f(x)，曲率为
    κ = y'' / (1 + (y')²)^(3/2)
    最大弯度、厚度点处是极大值点,即y'(x) = 0
    当 y' = 0 时，曲率公式退化为：
    κ = y'' / (1 + 0)^(3/2) = y''

    在微分几何中，平面曲线的“曲率变化率”定义为：
    dκ/ds = dκ/dx × dx/ds
    任意平面曲线 y = f(x)，从 x 到 x+dx 的弧长微元 ds 为：
    ds = √(dx² + dy²)
       = √(dx² + (f'(x)·dx)²)
       = √(dx² · (1 + [f'(x)]²))
       = dx · √(1 + [f'(x)]²)
    因此：
        ds/dx = √(1 + (f')²)
        dx/ds = 1 / √(1 + (f')²)
    带入曲率对弧长 s 的变化率公式,得
    dκ/ds = (dκ/dx) / √(1 + (f'(x))²)
    在最大弯度和厚度处f'(x)=0
    所以 dκ/ds = (dκ/dx) 
    dκ/dx=(y'''((1 + (y')²)-3y'(y'')²)/((1 + (y')²)^3
    当 y' = 0 时
    曲率变化率dκ/dx=y'''
    """
    # 三次样条拟合
    cs_camber = CubicSpline(x, camber, bc_type='natural')
    cs_thick = CubicSpline(x, thick, bc_type='natural')

    d2z_C_XC = float(cs_camber(XC, 2))  # 最大弯度处中弧线曲率
    d3z_C_XC = float(cs_camber(XC, 3))  # 最大弯度处曲率变化率

    d2t_XT = float(cs_thick(XT, 2))  # 最大厚度处厚度线曲率
    d3t_XT = float(cs_thick(XT, 3))  # 最大厚度处曲率变化率

    # ==================== 6. 前缘半径（最小二乘圆拟合）================
    le_mask = x <= 0.005
    x_le = np.concatenate([UpX[le_mask], LowX[le_mask]])
    y_le = np.concatenate([UpY[le_mask], LowY[le_mask]])

    def circle_residuals(p, xx, yy):
        return np.sqrt((xx - p[0]) ** 2 + (yy - p[1]) ** 2) - p[2]

    # 初始猜测：圆心在 (0,0)，半径 0.01
    res = least_squares(circle_residuals, [0.0, 0.0, 0.01], args=(x_le, y_le),
                        bounds=([-0.05, -0.1, 0.0], [0.05, 0.1, 0.2]))
    R_LE = max(res.x[2], 1e-6)  # 防止负值

    # ==================== 6. 返回 14 个几何特征（顺序固定）================
    params = [
        float(C),  # 0. 最大相对弯度
        float(XC),  # 1. 最大弯度位置 (x/c)
        float(z_TE),  # 2. 尾缘中弧线高度
        float(alpha_TE),  # 3. 尾缘中弧线角度 (rad)
        float(alpha_LE),  # 4. 前缘中弧线角度 (rad)
        float(d2z_C_XC),  # 5. 最大弯度处中弧线曲率（二阶导数）
        float(d3z_C_XC),  # 6. 最大弯度处曲率变化率（三阶导数）
        float(XT),  # 7. 最大厚度位置 (x/c)
        float(T),  # 8. 最大相对厚度
        float(R_LE),  # 9. 前缘半径（归一化）
        float(beta_TE),  # 10. 尾缘夹角 (rad)
        float(delta_Z_TE),  # 11. 尾缘开口厚度
        float(d2t_XT),  # 12. 最大厚度处厚度分布曲率
        float(d3t_XT)  # 13. 最大厚度处厚度分布曲率变化率
    ]
    return params



from scipy.optimize import fsolve
#SPIP翼型直观参数化函数
def SPIP_fit(params, n_points=100):
    """
    中弧线前段多项式（x, x^1.5, x^2.5, x^3.5, x^4.5），后段多项式（x, x^1.5, x^2.5, x^3.5, x^4.5, x^5.5）；
    厚度线前段 x^0.5, x^1.5, x^2.5, x^3.5, x^4.5，后段 x^0.5, x^1.5, x^2.5, x^3.5, x^4.5, x^5.5。
    """
    try:
        C, XC, z_TE, alpha_TE, alpha_LE, d2z_C_XC, d3z_C_XC, XT, T, R_LE, beta_TE, delta_Z_TE, d2t_XT, d3t_XT = params
    except ValueError:
        raise ValueError("参数数量必须为 14")
    if not (0.01 < XC < 0.99 and 0.01 < XT < 0.99):
        raise ValueError("XC 和 XT 必须在 (0.01, 0.99) 范围内以确保数值稳定性")

    # 生成余弦分布的 x 坐标
    t = np.linspace(0, np.pi, n_points)
    x = (1 - np.cos(t)) / 2

    # ------------------------------------------------------------
    # 1. 中弧线分段多项式拟合模型
    # ------------------------------------------------------------
    def compute_camber_line(x, C, XC, z_TE, alpha_TE, alpha_LE, d2z_C_XC, d3z_C_XC):
        def camber_constraints(coeffs, XC, C, z_TE, alpha_TE, alpha_LE, d2z_C_XC, d3z_C_XC):
            a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6 = coeffs

            # ---------- 前段（5 约束） ----------
            # (1) 前缘斜率 = alpha_LE
            eq1 = a1 - np.tan(alpha_LE)
            # (2) 最大弯度处值 = C
            eq2 = (a1 * XC + a2 * XC ** 1.5 + a3 * XC ** 2.5 + a4 * XC ** 3.5 + a5 * XC ** 4.5) - C
            # (3) 最大弯度处一阶导数 = 0
            eq3 = (a1 + 1.5 * a2 * XC ** 0.5 + 2.5 * a3 * XC ** 1.5 + 3.5 * a4 * XC ** 2.5 + 4.5 * a5 * XC ** 3.5)
            # (4) 最大弯度处二阶导数 = d2z_C_XC
            eq4 = (0.75 * a2 * XC ** (
                -0.5) + 3.75 * a3 * XC ** 0.5 + 8.75 * a4 * XC ** 1.5 + 15.75 * a5 * XC ** 2.5) - d2z_C_XC
            # (5) 最大弯度处三阶导数 = d3z_C_XC
            eq5 = (-0.375 * a2 * XC ** (-1.5) + 1.875 * a3 * XC ** (
                -0.5) + 13.125 * a4 * XC ** 0.5 + 39.375 * a5 * XC ** 1.5) - d3z_C_XC

            # ---------- 后段（5 约束） ----------
            # (6) 最大弯度处值 = C
            eq6 = (b1 * XC + b2 * XC ** 1.5 + b3 * XC ** 2.5 + b4 * XC ** 3.5 + b5 * XC ** 4.5 + b6 * XC ** 5.5) - C
            # (7) 最大弯度处一阶导数 = 0
            eq7 = (
                        b1 + 1.5 * b2 * XC ** 0.5 + 2.5 * b3 * XC ** 1.5 + 3.5 * b4 * XC ** 2.5 + 4.5 * b5 * XC ** 3.5 + 5.5 * b6 * XC ** 4.5)
            # (8) 最大弯度处二阶导数 = d2z_C_XC
            eq8 = (0.75 * b2 * XC ** (
                -0.5) + 3.75 * b3 * XC ** 0.5 + 8.75 * b4 * XC ** 1.5 + 15.75 * b5 * XC ** 2.5 + 27.75 * b6 * XC ** 3.5) - d2z_C_XC
            # (9) 最大弯度处三阶导数 = d3z_C_XC
            eq9 = (-0.375 * b2 * XC ** (-1.5) + 1.875 * b3 * XC ** (
                -0.5) + 13.125 * b4 * XC ** 0.5 + 39.375 * b5 * XC ** 1.5 + 110.25 * b6 * XC ** 2.5) - d3z_C_XC
            # (10) 尾缘高度 = z_TE
            eq10 = (b1 + b2 + b3 + b4 + b5 + b6) - z_TE
            # (11)尾缘斜率 = alpha_TE
            eq11 = (b1 + 1.5 * b2 + 2.5 * b3 + 3.5 * b4 + 4.5 * b5 + 5.5 * b6) - np.tan(alpha_TE)
            return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11]

        # fsolve求解初始猜测
        initial_guess = [np.tan(alpha_LE), 0, 0, 0, 0, np.tan(alpha_TE), 0, 0, 0, 0, 0]
        coeffs = fsolve(camber_constraints, initial_guess, args=(XC, C, z_TE, alpha_TE, alpha_LE, d2z_C_XC, d3z_C_XC))
        a1, a2, a3, a4, a5, b1, b2, b3, b4, b5, b6 = coeffs

        # 中弧线几何坐标重构
        z_C = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi <= XC:
                z_C[i] = (a1 * xi + a2 * xi ** 1.5 + a3 * xi ** 2.5 + a4 * xi ** 3.5 + a5 * xi ** 4.5)
            else:
                z_C[i] = (b1 * xi + b2 * xi ** 1.5 + b3 * xi ** 2.5 + b4 * xi ** 3.5 + b5 * xi ** 4.5 + b6 * xi ** 5.5)

        return z_C  # 返回中弧线几何坐标

    # ------------------------------------------------------------
    # 2. 厚度线分段多项式拟合模型
    # ------------------------------------------------------------
    def compute_thickness(x, XT, T, R_LE, beta_TE, delta_Z_TE, d2t_XT, d3t_XT):
        def thickness_constraints(coeffs, XT, T, R_LE, beta_TE, delta_Z_TE, d2t_XT, d3t_XT):
            t1, t2, t3, t4, t5, r1, r2, r3, r4, r5, r6 = coeffs

            # ---------- 前段（5 约束） ----------
            # (1) 前缘半径约束 t1= 2√2*R_LE
            eq1 = t1 - 2 * np.sqrt(2 * R_LE)
            # (2) 最大厚度处值 = T
            eq2 = (t1 * XT ** 0.5 + t2 * XT ** 1.5 + t3 * XT ** 2.5 + t4 * XT ** 3.5 + t5 * XT ** 4.5) - T
            # (3) 最大厚度处一阶导数 = 0
            eq3 = (0.5 * t1 * XT ** (
                -0.5) + 1.5 * t2 * XT ** 0.5 + 2.5 * t3 * XT ** 1.5 + 3.5 * t4 * XT ** 2.5 + 4.5 * t5 * XT ** 3.5)
            # (4) 最大厚度处二阶导数 = d2t_XT
            eq4 = (-0.25 * t1 * XT ** (-1.5) + 0.75 * t2 * XT ** (
                -0.5) + 3.75 * t3 * XT ** 0.5 + 8.75 * t4 * XT ** 1.5 + 15.75 * t5 * XT ** 2.5) - d2t_XT
            # (5) 最大厚度处三阶导数 = d3t_XT
            eq5 = (0.375 * t1 * XT ** (-2.5) - 0.375 * t2 * XT ** (-1.5) + 1.875 * t3 * XT ** (
                -0.5) + 13.125 * t4 * XT ** 0.5 + 39.375 * t5 * XT ** 1.5) - d3t_XT

            # ---------- 后段（5 约束） ----------
            # (6) 最大厚度处值 = T
            eq6 = (
                              r1 * XT ** 0.5 + r2 * XT ** 1.5 + r3 * XT ** 2.5 + r4 * XT ** 3.5 + r5 * XT ** 4.5 + r6 * XT ** 5.5) - T
            # (7) 最大厚度处一阶导数 = 0
            eq7 = (0.5 * r1 * XT ** (
                -0.5) + 1.5 * r2 * XT ** 0.5 + 2.5 * r3 * XT ** 1.5 + 3.5 * r4 * XT ** 2.5 + 4.5 * r5 * XT ** 3.5 + 5.5 * r6 * XT ** 4.5)
            # (8) 最大厚度处二阶导数 = d2t_XT
            eq8 = (-0.25 * r1 * XT ** (-1.5) + 0.75 * r2 * XT ** (
                -0.5) + 3.75 * r3 * XT ** 0.5 + 8.75 * r4 * XT ** 1.5 + 15.75 * r5 * XT ** 2.5 + 27.75 * r6 * XT ** 3.5) - d2t_XT
            # (9) 最大厚度处三阶导数 = d3t_XT
            eq9 = (0.375 * r1 * XT ** (-2.5) - 0.375 * r2 * XT ** (-1.5) + 1.875 * r3 * XT ** (
                -0.5) + 13.125 * r4 * XT ** 0.5 + 39.375 * r5 * XT ** 1.5 + 110.25 * r6 * XT ** 2.5) - d3t_XT
            # (10) 尾缘厚度delta_Z_TE
            eq10 = (r1 + r2 + r3 + r4 + r5 + r6) - delta_Z_TE
            # (11)尾缘斜率2*np.tan(beta_TE/2)
            eq11 = (0.5 * r1 + 1.5 * r2 + 2.5 * r3 + 3.5 * r4 + 4.5 * r5 + 5.5 * r6) - (-2 * np.tan(beta_TE / 2))
            return [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9, eq10, eq11]

        # fsolve求解初始猜测
        initial_guess = [2 * np.sqrt(2 * R_LE), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        coeffs = fsolve(thickness_constraints, initial_guess, args=(XT, T, R_LE, beta_TE, delta_Z_TE, d2t_XT, d3t_XT))
        t1, t2, t3, t4, t5, r1, r2, r3, r4, r5, r6 = coeffs

        # 厚度线几何坐标重构
        t = np.zeros_like(x)
        for i, xi in enumerate(x):
            if xi <= XT:
                t[i] = (t1 * xi ** 0.5 + t2 * xi ** 1.5 + t3 * xi ** 2.5 + t4 * xi ** 3.5 + t5 * xi ** 4.5)
            else:
                t[i] = (
                            r1 * xi ** 0.5 + r2 * xi ** 1.5 + r3 * xi ** 2.5 + r4 * xi ** 3.5 + r5 * xi ** 4.5 + r6 * xi ** 5.5)

        return t  # 返回厚度线几何坐标

    # ------------------------------------------------------------
    # 翼型几何坐标重构
    # ------------------------------------------------------------
    z_C = compute_camber_line(x, C, XC, z_TE, alpha_TE, alpha_LE, d2z_C_XC, d3z_C_XC)
    t = compute_thickness(x, XT, T, R_LE, beta_TE, delta_Z_TE, d2t_XT, d3t_XT)

    z_up = z_C + 0.5 * t
    z_low = z_C - 0.5 * t
    return x, z_up, z_low, z_C, t


from scipy.optimize import least_squares
from scipy.interpolate import interp1d
#最小二乘法求解翼型的SPIP拟合参数
def compute_fitting_params_Least_Squares(initial_params, up_x, up_y, low_x, low_y):
    """
    使用最小二乘法优化 IPARSEC 参数，使拟合翼型逼近原始上/下表面。
    """
    initial_params = np.array(initial_params)

    def residuals(params):
        try:
            x, z_up, z_low, _, _ = SPIP_fit(params, len(up_x))
            Pchip_up_ori = PchipInterpolator(up_x, up_y)
            Pchip_low_ori = PchipInterpolator(low_x, low_y)
            res_up = PchipInterpolator(x, z_up)(x) - Pchip_up_ori(x)
            res_low = PchipInterpolator(x, z_low)(x) - Pchip_low_ori(x)
            w = np.where(x < 0.2, 1, 1.0)
            return np.concatenate([w * res_up, w * res_low])
        except:
            return np.ones(2 * len(up_x)) * 1e6

    # ========== 关键修复：放宽边界，特别是导数项 ==========
    bounds = (
        [-0.3, 0.01, -0.1, -np.pi / 3, -np.pi / 3, -200, -5e5, 0.01, 0.05, 1e-6, 0, 0, -200, -5e5],  # 下界
        [0.3, 0.99, 0.1, np.pi / 3, np.pi / 3, 100, 5e5, 0.99, 0.60, 0.40, np.pi / 2, 0.15, 100, 5e5]  # 上界
    )

    result = least_squares(residuals, initial_params, bounds=bounds, method='trf', ftol=1e-12, xtol=1e-12, gtol=1e-12,
                           max_nfev=30000, verbose=0)
    opt_params = result.x
    final_mae = np.mean(abs(residuals(opt_params)))
    print(f"优化完成，WMAE ≈ {final_mae:.6f}")

    return opt_params, final_mae


#畸变翼型检测函数
def detect_deformed_airfoil(UpX, UpY, LowX, LowY):
    """
    1.自交检测；
    2.光滑性检测
    （1）主段光滑性检测（x/c > 0.03）
    主段（x/c > 0.03）光滑性异常（曲率 + 斜率变化 + 中弧线联合判定）
    曲率太大 → 局部弯得太狠
    斜率变化率太大 → 表面有拐点
    弯度斜率变化太大 → 中弧线不顺
    只有三个同时超标才判异常，避免误判正常高升力翼型
    （2）前缘区域精细检测（x/c ≤ 0.03）
    a.正常检测：三个指标同时超标 → 前缘不够光滑
    b.极端异常：任一指标严重爆炸 → 前缘畸变成尖角/锯齿/S形
    """

    # 计算斜率变化率
    def compute_slope_variation(x, y):
        """
        计算局部斜率变化率，覆盖整个 x/c 范围
        参数:
        - x, y: 曲线坐标数组
        返回: 一阶导数，斜率变化率，均值，标准差，最大值
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if len(x) < 2 or len(y) < 2:
            raise ValueError("Input arrays must have at least 2 points.")
        if len(x) != len(y):
            raise ValueError("x and y arrays must have the same length.")

        dx = np.gradient(x)
        dy = np.gradient(y)
        first_diff = dy / (dx + 1e-10)  # 一阶导数 dy/dx
        slope_change = np.abs(np.diff(first_diff))  # 斜率变化率
        mean_slope_change = np.mean(slope_change) if slope_change.size > 0 else 0
        std_slope_change = np.std(slope_change, ddof=1) if slope_change.size > 0 else 0
        max_slope_change = np.max(slope_change) if slope_change.size > 0 else 0
        return first_diff, slope_change, mean_slope_change, std_slope_change, max_slope_change

    # 计算曲率
    def compute_curvature(x, y, num_interp_points=200):
        """
        计算曲线曲率 κ = |y''| / (1 + (y')²)^{3/2}，覆盖整个 x/c 范围
        使用 Pchip 插值进行平滑计算二阶导数
        参数:
        - x, y: 曲线坐标数组
        - num_interp_points: 插值点数（默认 200）
        返回: 曲率数组，均值，标准差，最大值
        """
        x = np.array(x, dtype=float)
        y = np.array(y, dtype=float)
        if len(x) < 2 or len(y) < 2:
            raise ValueError("Input arrays must have at least 2 points.")
        if len(x) != len(y):
            raise ValueError("x and y arrays must have the same length.")

        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]

        interp = PchipInterpolator(x, y)
        x_interp = np.linspace(x.min(), x.max(), num_interp_points)
        y_interp = interp(x_interp)

        dy_dx = np.gradient(y_interp, x_interp)
        d2y_dx2 = np.gradient(dy_dx, x_interp)
        curvature = np.abs(d2y_dx2) / (1 + dy_dx ** 2) ** 1.5

        mean_curvature = np.mean(curvature) if curvature.size > 0 else 0
        std_curvature = np.std(curvature, ddof=1) if curvature.size > 0 else 0
        max_curvature = np.max(curvature) if curvature.size > 0 else 0

        return curvature, mean_curvature, std_curvature, max_curvature

    # 计算厚度和中弧线
    def compute_thickness_camber(UpX, UpY, LowX, LowY, num_points=100):
        """
        计算厚度分布和弯度分布，使用余弦分布的 x_common
        参数:
        - UpX, UpY: 上翼面坐标
        - LowX, LowY: 下翼面坐标
        - num_points: 插值点数（默认 100）
        返回: 厚度分布，弯度分布，x_common
        """
        UpX = np.array(UpX, dtype=float)
        UpY = np.array(UpY, dtype=float)
        LowX = np.array(LowX, dtype=float)
        LowY = np.array(LowY, dtype=float)
        if len(UpX) < 2 or len(LowX) < 2:
            raise ValueError("Upper or lower surface must have at least 2 points.")
        if len(UpX) != len(UpY) or len(LowX) != len(LowY):
            raise ValueError("x and y arrays must have the same length for each surface.")
        theta = np.linspace(0, np.pi, num_points)
        x_common = 0.5 * (1 - np.cos(theta))  # 余弦分布：x/c 从 0 到 1
        interp_up = PchipInterpolator(UpX, UpY)
        interp_low = PchipInterpolator(LowX, LowY)
        UpY_interp = interp_up(x_common)
        LowY_interp = interp_low(x_common)

        thickness = abs(UpY_interp - LowY_interp)
        camber = (UpY_interp + LowY_interp) / 2

        return thickness, camber, x_common

    # 自交检测
    def is_airfoil_self_intersecting(UpX, UpY, LowX, LowY):
        """
        检测翼型是否自交。
        参数:
            UpX, UpY: 上表面 x, y 坐标列表
            LowX, LowY: 下表面 x, y 坐标列表
        返回:
            bool: True 表示自交，False 表示不自交
        """

        def line_segment_intersect(p1, p2, q1, q2):
            """
            判断两线段是否相交。
            p1, p2: 第一条线段的两个端点 (x, y)
            q1, q2: 第二条线段的两个端点 (x, y)
            """

            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

            def intersect(A, B, C, D):
                return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

            return intersect(p1, p2, q1, q2)

        # 将上下表面坐标组合为点列表
        upper_points = list(zip(UpX, UpY))
        lower_points = list(zip(LowX, LowY))

        # 检查每条上表面线段与下表面线段是否相交
        for i in range(len(upper_points) - 1):
            for j in range(len(lower_points) - 1):
                p1, p2 = upper_points[i], upper_points[i + 1]
                q1, q2 = lower_points[j], lower_points[j + 1]
                # 排除首尾点（前缘和尾缘厚度为0，允许重合）
                if (i == 0 and j == 0) or (i == len(upper_points) - 2 and j == len(lower_points) - 2):
                    continue
                # 检查线段是否相交
                if line_segment_intersect(p1, p2, q1, q2):
                    return True
        return False
    reasons = []  # 畸形原因
    UpX = np.array(UpX, dtype=float)
    UpY = np.array(UpY, dtype=float)
    LowX = np.array(LowX, dtype=float)
    LowY = np.array(LowY, dtype=float)

    # -------------------------- 1. 自交检测 --------------------------
    intersection = is_airfoil_self_intersecting(UpX, UpY, LowX, LowY)
    if intersection:
        reasons.append("上下翼面发生自交")

    # -------------------------- 2. 光滑性检测 --------------------------
    # 计算评估值
    _, camber, x_common = compute_thickness_camber(UpX, UpY, LowX, LowY)  # 计算中弧线曲线
    # 翼面斜率变化率（np.diff 后的长度比原始少1）
    _, upper_slope_change, _, _, upper_max_slope_change = compute_slope_variation(UpX, UpY)  # 上翼面斜率变化率及最大值
    _, lower_slope_change, _, _, lower_max_slope_change = compute_slope_variation(LowX, LowY)  # 下翼面斜率变化率及最大值
    _, camber_slope_change, _, _, camber_max_slope_change = compute_slope_variation(x_common, camber)  # 中弧线斜率变化率及最大值
    # 翼面曲率
    upper_curvature, _, _, upper_max_curvature = compute_curvature(UpX, UpY)  # 上翼面斜率变化率及最大值
    lower_curvature, _, _, lower_max_curvature = compute_curvature(LowX, LowY)  # 下翼面斜率变化率及最大值
    # 为曲率准备对应的 x 坐标（插值后等间距）
    x_upper_interp = np.linspace(UpX.min(), UpX.max(), len(upper_curvature))
    x_lower_interp = np.linspace(LowX.min(), LowX.max(), len(lower_curvature))
    ################################## 后段光滑性检测 ###################################################
    # 分离主段
    upper_mask = UpX[:-1] > 0.03
    lower_mask = LowX[:-1] > 0.03
    camber_mask = x_common[:-1] > 0.03

    # 获取主段翼型数据（x/c > 0.03）
    upper_curvature_mask = x_upper_interp > 0.03
    lower_curvature_mask = x_lower_interp > 0.03

    # 主段上翼面检测（联合中弧线斜率变化率）x/c > 0.03
    upper_condition = False
    if upper_curvature_mask.any() and upper_mask.any() and camber_mask.any():
        upper_curvature_x = upper_curvature[upper_curvature_mask]
        upper_slope_change_x = upper_slope_change[upper_mask]
        camber_slope_change_x = camber_slope_change[camber_mask]

        upper_max_curvature_x = np.max(upper_curvature_x) if upper_curvature_x.size > 0 else 0
        upper_max_slope_change_x = np.max(upper_slope_change_x) if upper_slope_change_x.size > 0 else 0
        camber_max_slope_change_x = np.max(camber_slope_change_x) if camber_slope_change_x.size > 0 else 0

        upper_condition = (upper_max_curvature_x > 10 and
                           upper_max_slope_change_x > 0.1 and
                           camber_max_slope_change_x > 0.1)

        if upper_condition:
            reasons.append(f"上翼面异常（x/c > 0.03）：曲率最大值 {upper_max_curvature_x:.6f} >10, "
                           f"斜率变化率最大值 {upper_max_slope_change_x:.6f} > 0.1, "
                           f"弯度斜率变化率最大值 {camber_max_slope_change_x:.6f} > 0.1")
    else:
        reasons.append("上翼面或弯度 x/c > 0.03 区域无数据")

    # 主段下翼面检测 （联合中弧线斜率变化率）x/c > 0.03
    lower_condition = False
    if lower_curvature_mask.any() and lower_mask.any() and camber_mask.any():
        lower_curvature_x = lower_curvature[lower_curvature_mask]
        lower_slope_change_x = lower_slope_change[lower_mask]
        camber_slope_change_x = camber_slope_change[camber_mask]

        lower_max_curvature_x = np.max(lower_curvature_x) if lower_curvature_x.size > 0 else 0
        lower_max_slope_change_x = np.max(lower_slope_change_x) if lower_slope_change_x.size > 0 else 0
        camber_max_slope_change_x = np.max(camber_slope_change_x) if camber_slope_change_x.size > 0 else 0

        lower_condition = (lower_max_curvature_x > 10 and
                           lower_max_slope_change_x > 0.1 and
                           camber_max_slope_change_x > 0.1)

        if lower_condition:
            reasons.append(f"下翼面异常（x/c > 0.03）：曲率最大值 {lower_max_curvature_x:.6f} > 10, "
                           f"斜率变化率最大值 {lower_max_slope_change_x:.6f} > 0.1, "
                           f"弯度斜率变化率最大值 {camber_max_slope_change_x:.6f} > 0.1")
    else:
        reasons.append("下翼面或弯度 x/c > 0.03 区域无数据")

    back_condition = upper_condition or lower_condition
    ################################## 前缘光滑性检测 ###################################################
    # 前缘光滑性 检测x/c <= 0.03
    # 提取前缘区域x/c <= 0.03的数据
    camber_x_mask = x_common[:-1] <= 0.03
    upper_x_mask = UpX[:-1] <= 0.03
    lower_x_mask = LowX[:-1] <= 0.03
    upper_curvature_x_mask = x_upper_interp <= 0.03
    lower_curvature_x_mask = x_lower_interp <= 0.03
    leading_edge_valid = True

    if (camber_x_mask.any() and upper_x_mask.any() and upper_curvature_x_mask.any() and
            lower_x_mask.any() and lower_curvature_x_mask.any()):

        # 上下翼面、中弧线联合检测

        # 前缘区域上翼面、中弧线联合检测
        # 上表面前缘指标
        upper_curvature_x = upper_curvature[upper_curvature_x_mask]
        upper_slope_change_x = upper_slope_change[upper_x_mask]
        camber_slope_change_x = camber_slope_change[camber_x_mask]

        upper_max_curvature_x = np.max(upper_curvature_x) if upper_curvature_x.size > 0 else 0
        upper_max_slope_change_x = np.max(np.abs(upper_slope_change_x)) if upper_slope_change_x.size > 0 else 0
        camber_max_slope_change_x = np.max(np.abs(camber_slope_change_x)) if camber_slope_change_x.size > 0 else 0

        upper_leading_condition = (upper_max_curvature_x > 60 and
                                   upper_max_slope_change_x > 8 and
                                   camber_max_slope_change_x > 8)
        if upper_leading_condition:
            leading_edge_valid = False
            reasons.append(f"x/c ≤ 0.03 区域上翼面异常：曲率最大值 {upper_max_curvature_x:.6f} > 60, "
                           f"斜率变化率绝对值最大值 {upper_max_slope_change_x:.6f} > 8, "
                           f"弯度斜率变化率绝对值最大值 {camber_max_slope_change_x:.6f} > 8")

        # 前缘区域下翼面、中弧线联合检测
        # 下翼面前缘指标
        lower_curvature_x = lower_curvature[lower_curvature_x_mask]
        lower_slope_change_x = lower_slope_change[lower_x_mask]

        lower_max_curvature_x = np.max(lower_curvature_x) if lower_curvature_x.size > 0 else 0
        lower_max_slope_change_x = np.max(np.abs(lower_slope_change_x)) if lower_slope_change_x.size > 0 else 0

        # 条件检测
        lower_leading_condition = (lower_max_curvature_x > 60 and
                                   lower_max_slope_change_x > 8 and
                                   camber_max_slope_change_x > 8)

        if lower_leading_condition:
            leading_edge_valid = False
            reasons.append(f"x/c ≤ 0.03 区域下翼面异常：曲率最大值 {lower_max_curvature_x:.6f} > 60, "
                           f"斜率变化率绝对值最大值 {lower_max_slope_change_x:.6f} > 8, "
                           f"弯度斜率变化率绝对值最大值 {camber_max_slope_change_x:.6f} > 8")

        # 前缘区域单个指标极端检测
        # 前缘区域上翼面极端检测
        upper_leading_extreme = (upper_max_curvature_x > 120 or
                                 upper_max_slope_change_x > 40 or
                                 camber_max_slope_change_x > 40)

        if upper_leading_extreme:
            leading_edge_valid = False
            reasons.append(f"x/c ≤ 0.03 区域上翼面极端异常："
                           f"曲率最大值 {upper_max_curvature_x:.6f} > 120 或 "
                           f"斜率变化率绝对值最大值 {upper_max_slope_change_x:.6f} > 40 或 "
                           f"弯度斜率变化率绝对值最大值 {camber_max_slope_change_x:.6f} > 40")

        # 前缘区域下翼面极端检测
        lower_leading_extreme = (lower_max_curvature_x > 120 or
                                 lower_max_slope_change_x > 40 or
                                 camber_max_slope_change_x > 40)
        if lower_leading_extreme:
            leading_edge_valid = False
            reasons.append(f"x/c ≤ 0.03 区域下翼面极端异常："
                           f"曲率最大值 {lower_max_curvature_x:.6f} > 120 或 "
                           f"斜率变化率绝对值最大值 {lower_max_slope_change_x:.6f} > 40 或 "
                           f"弯度斜率变化率绝对值最大值 {camber_max_slope_change_x:.6f} > 40")
    else:
        reasons.append("x/c ≤ 0.03 区域无足够数据（上翼面、下翼面或弯度）")

    leading_edge_invalid = upper_leading_condition or lower_leading_condition or upper_leading_extreme or lower_leading_extreme

    # Determine if airfoil is valid
    is_valid = not (intersection or back_condition or leading_edge_invalid)


    return is_valid, reasons


#基于SPIP的翼型变体生成函数
def generate_and_plot_airfoils(params, num_samples=1000, variation_pct=0.25, seed=42, distortion_detect=False):
    """
    生成随机扰动的翼型参数并绘图，支持部分参数固定 + 可选畸变检测
    返回:
        generated_df: 包含 ID + 所有参数 + distortion 列的 DataFrame
                      （若未开启检测，distortion 全为 False）
    """
    # 参数名（请确保与你的 SPIP_fit 函数顺序完全一致！）
    params_name = ['C', 'XC', 'z_TE', 'alpha_TE', 'alpha_LE', 'd2z_C_XC', 'd3z_C_XC', 'XT', 'T', 'R_LE', 'beta_TE',
                   'delta_Z_TE', 'd2t_XT', 'd3t_XT']

    fixed_params_name = []

    params = np.asarray(params, dtype=float)
    np.random.seed(seed)

    # 处理固定参数名
    if fixed_params_name is None:
        fixed_names = []
    elif isinstance(fixed_params_name, str):
        fixed_names = [fixed_params_name]
    else:
        fixed_names = list(fixed_params_name)

    fixed_indices = [i for i, name in enumerate(params_name) if name in fixed_names]
    vary_indices = [i for i in range(len(params)) if i not in fixed_indices]

    # 生成参数矩阵
    generated_params = np.tile(params, (num_samples, 1))  # 先复制基准值

    for i in vary_indices:
        low = params[i] * (1 - variation_pct)
        high = params[i] * (1 + variation_pct)
        # 物理约束保护
        if params_name[i] in ['XC', 'XT']:
            low = max(low, 0.05)
            high = min(high, 0.95)
        if params_name[i] == 'R_LE':
            low = max(low, 1e-6)
        if params_name[i] == 'beta_TE':
            high = min(high, np.pi / 2)
        if params_name[i] == 'T':
            high = min(high, 0.6)  # 相对厚度一般不超过60%

        generated_params[:, i] = np.random.uniform(low, high, size=num_samples)

    # 构建 DataFrame
    generated_df = pd.DataFrame(generated_params, columns=params_name)
    generated_df.insert(0, 'ID', range(1, num_samples + 1))
    generated_df['distortion'] = False

    # ==================== 可选畸变检测 ====================
    if distortion_detect:
        print("正在进行畸变检测（此步骤较慢）...", end="")
        distortion_count = 0
        failed_indices = []

        for i in range(num_samples):
            try:
                x, z_up, z_lo, _, _ = SPIP_fit(generated_params[i])
                is_valid, _ = detect_deformed_airfoil(x, z_up, x, z_lo)
                if not is_valid:
                    generated_df.at[i, 'distortion'] = True
                    distortion_count += 1
                    failed_indices.append(i)
            except Exception as e:
                # SPIP_fit 本身报错也视为畸变（防止崩溃）
                generated_df.at[i, 'distortion'] = True
                distortion_count += 1
                failed_indices.append(i)

        distortion_rate = distortion_count / num_samples * 100
        print(f"完成！畸变数量: {distortion_count}，畸变率: {distortion_rate:.2f}%")
    else:
        print("畸变检测已关闭")

    # ==================== 绘图 ====================
    plt.figure(figsize=(12, 8))

    # 绘制所有生成翼型
    for i in range(num_samples):
        color = 'red' if distortion_detect and generated_df.at[i, 'distortion'] else 'blue'
        alpha = 0.4
        linewidth = 1.2

        x, z_up, z_lo, _, _ = SPIP_fit(generated_params[i])
        x_plot = np.concatenate([x, x[::-1]])
        z_plot = np.concatenate([z_up, z_lo[::-1]])
        plt.plot(x_plot, z_plot, color=color, alpha=alpha, linewidth=linewidth)

    # 绘制基准翼型（粗红线）
    x0, z_up0, z_lo0, _, _ = SPIP_fit(params)
    x0_plot = np.concatenate([x0, x0[::-1]])
    z0_plot = np.concatenate([z_up0, z_lo0[::-1]])
    plt.plot(x0_plot, z0_plot, 'darkred', linewidth=3, label='Reference Airfoil')

    plt.axis('equal')
    plt.xlabel('x/c')
    plt.ylabel('z/c')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return generated_df

import aerosandbox as asb
#基于neuralfoil的翼型性能计算函数
import numpy as np
import aerosandbox as asb
def compute_air_aerosandbox(airfoil_coords, Re, Ma, alpha):
    coords = np.asarray(airfoil_coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("airfoil_coords 必须是 (N, 2) 形状")

    alphas = np.atleast_1d(alpha)
    airfoil = asb.Airfoil(coordinates=coords)
    output_list = []

    for alpha_val in alphas:
        try:
            aero = airfoil.get_aero_from_neuralfoil(
                alpha=alpha_val,        # 直接传度数
                Re=Re,
                mach=Ma,
                model_size="xxxlarge"
            )
            Cl = float(np.asarray(aero["CL"]).item())
            Cd = float(np.asarray(aero["CD"]).item())
            Cm = float(np.asarray(aero["CM"]).item())

            results = {
                'alpha': float(alpha_val),
                'Re'   : float(Re),
                'Ma'   : float(Ma),
                'CL'   : Cl,
                'CD'   : Cd,
                'CM'   : Cm
            }
        except Exception as e:
            print(f"Warning: α={alpha_val:.1f}° 计算失败: {e}")
            results = {
                'alpha': float(alpha_val), 'Re': float(Re), 'Ma': float(Ma),
                'CL': np.nan, 'CD': np.nan, 'CM': np.nan
            }
        output_list.append(results)

    return output_list[0] if len(output_list) == 1 else output_list

# # 主程序示例
# if __name__ == "__main__":
#     # 读取基准翼型数据
#     file_path = r'C:\Users\Administrator\YFA\PycharmProjects\tfgpu_file\cumpute_airfoil\coordinate\DU93-W-210.prof'
#     coordinate = read_airfoil_data(file_path)
#     UpX, UpY, LowX, LowY = preprocess_airfoil_data(coordinate, normalize=True)
#
#     # 计算初始参数
#     initial_params = compute_airfoil_geometrical_feature(UpX, UpY, LowX, LowY)
#     base_params, final_mae = compute_fitting_params_Least_Squares(initial_params, UpX, UpY, LowX, LowY)
