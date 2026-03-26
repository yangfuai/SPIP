# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import zipfile
from pathlib import Path
try:
    from pyxfoil import XFoil
except ImportError:
    XFoil = None

# ==================== 中文字体 + 数学符号支持 ====================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

# ==================== 美化CSS ====================
st.set_page_config(page_title="SPIP 翼型直观参数化设计平台", page_icon="airplane", layout="wide")

st.markdown("""
<style>
    .main {background: linear-gradient(135deg, #0f172a, #1e293b, #334155);}
    .card {background: rgba(30,41,59,0.96); backdrop-filter: blur(15px); border-radius: 20px; padding: 30px; 
           border: 1px solid rgba(100,200,255,0.3); box-shadow: 0 15px 40px rgba(0,0,0,0.6);}
    .title-glow {font-size: 68px; font-weight: 900; background: linear-gradient(90deg, #00ffff, #7c3aed, #ff00ff);
                 -webkit-background-clip: text; color: transparent; text-align: center; text-shadow: 0 0 40px rgba(0,255,255,0.6);}
    .stButton>button {background: linear-gradient(90deg, #00d4ff, #7c3aed); border: none; border-radius: 16px; height: 60px; font-size: 20px; font-weight: bold; color: white;}
    .stDownloadButton>button {background: linear-gradient(90deg, #ff0080, #ff6b6b);}
    .success-box {background: linear-gradient(90deg, #11998e, #38ef7d); -webkit-background-clip: text; color: transparent; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# ==================== 初始化 session_state ====================
if "current_params" not in st.session_state:
    st.session_state.current_params = None
if "params_fixed" not in st.session_state:
    st.session_state.params_fixed = False
if "selected_params" not in st.session_state:
    st.session_state.selected_params = []
if "xfoil_df" not in st.session_state:
    st.session_state.xfoil_df = pd.DataFrame()
if "neuralfoil_df" not in st.session_state:
    st.session_state.neuralfoil_df = pd.DataFrame()

# ==================== 导入函数 ====================
try:
    from functions import (
        read_airfoil_data, preprocess_airfoil_data, compute_airfoil_geometrical_feature,
        SPIP_fit, compute_fitting_params_Least_Squares, detect_deformed_airfoil,
        generate_and_plot_airfoils, compute_air_aerosandbox
    )
except Exception as e:
    st.error("导入 functions 模块失败，请检查 functions.py 是否在同一目录")
    st.code(str(e))
    st.stop()

# ==================== 全局标题 ====================
st.markdown("<h1 class='title-glow'>SPIP</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:26px; color:#94a3b8; margin-bottom:40px;'>"
            "14个几何直观参数 · 翼型直观参数化与设计 · 翼型性能分析", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["翼型直观参数化与设计", "翼型性能分析"])

# ==================== 全局常量 ====================
COL_NAMES = ['C', 'XC', 'z_TE', 'alpha_TE', 'alpha_LE', 'd2z_C_XC', 'd3z_C_XC', 'XT', 'T', 'R_LE', 'beta_TE','delta_Z_TE', 'd2t_XT', 'd3t_XT']
idx = {n: i for i, n in enumerate(COL_NAMES)}

camber = ['alpha_LE', 'XC', 'C', 'd2z_C_XC', 'd3z_C_XC', 'z_TE', 'alpha_TE']
thick = ['R_LE', 'XT', 'T', 'd2t_XT', 'd3t_XT', 'delta_Z_TE', 'beta_TE']

sym = {'alpha_LE': '$\\alpha_{LE}$', 'XC': '$X_{C}$', 'C': '$C$', 'd2z_C_XC': '$Cxx_c$',
       'd3z_C_XC': '$Cxxx_c$', 'z_TE': '$Z_{TE}$', 'alpha_TE': '$\\alpha_{TE}$',
       'R_LE': '$R_{LE}$', 'XT': '$X_{T}$', 'T': '$T$', 'd2t_XT': '$Txx_t$',
       'd3t_XT': '$Txxx_t$', 'delta_Z_TE': '$\\Delta Z_{TE}$', 'beta_TE': '$\\beta_{TE}$'}

name_cn = {'alpha_LE': "前缘方向角", 'XC': "最大弯度位置", 'C': "最大相对弯度",
           'd2z_C_XC': "最大弯度处二阶导数", 'd3z_C_XC': "最大弯度处三阶导数",
           'z_TE': "尾缘中弧线高度", 'alpha_TE': "尾缘方向角",
           'R_LE': "前缘半径", 'XT': "最大厚度位置", 'T': "最大相对厚度",
           'd2t_XT': "最大厚度处二阶导数", 'd3t_XT': "最大厚度处三阶导数",
           'delta_Z_TE': "尾缘开口厚度", 'beta_TE': "尾缘夹角"}

rng = {'alpha_LE':"-45° ~ 45°", 'XC':"0.1 ~ 0.9", 'C':"-0.15 ~ 0.15",
               'd2z_C_XC':"-5 ~ 1", 'd3z_C_XC':"-25 ~ 50", 'z_TE':"-0.05 ~ 0.05",
               'alpha_TE':"-45° ~ 45°", 'R_LE':"0.003 ~ 0.25", 'XT':"0.2 ~ 0.65",
               'T':"0.05 ~ 0.6", 'd2t_XT':"-5 ~ 1", 'd3t_XT':"-25 ~ 50",
               'delta_Z_TE':"0 ~ 0.1", 'beta_TE':"0° ~ 60°"}

# ==================== 改进的翼型坐标保存函数（去重前缘点） ====================
def save_airfoil_dat(x, zu, zl, name="Airfoil"):
    """生成 Selig 格式翼型文件，确保前缘只有一个点 (x=0)"""
    # 上表面从后缘到前缘，下表面从前缘到后缘 → 合并后前缘会有两个 x≈0 的点
    x_upper = x[::-1]          # 从前缘到后缘（上表面）
    y_upper = zu[::-1]
    x_lower = x[1:]            # 从前缘后第二个点开始（避免重复 x=0）
    y_lower = zl[1:]

    # 完整坐标：上表面（后缘→前缘） + 下表面（前缘→后缘）
    x_full = np.concatenate([x_upper, x_lower])
    y_full = np.concatenate([y_upper, y_lower])

    # 构造文件内容
    content = f"{name}\n"
    for xx, yy in zip(x_full, y_full):
        content += f"{xx:.10f}  {yy:.10f}\n"
    return content


# ==================================================================
#                        Tab 1：翼型参数化与直观设计
# ==================================================================
with tab1:
    uploaded_file = st.file_uploader("上传翼型坐标文件（Selig 格式）", type=["dat", "prof", "txt"], key="main_uploader")

    if uploaded_file is not None:
        Path("temp_airfoil.dat").write_bytes(uploaded_file.getvalue())
        coords = read_airfoil_data("temp_airfoil.dat")
        UpX, UpY, LowX, LowY = preprocess_airfoil_data(coords, normalize=True)
        st.success(f"成功加载翼型：**{uploaded_file.name}**")

        init_params = compute_airfoil_geometrical_feature(UpX, UpY, LowX, LowY)
        opt_params, final_mae = compute_fitting_params_Least_Squares(init_params, UpX, UpY, LowX, LowY)
        st.info(f"SPIP参数化拟合完成，MAE ≈ {final_mae:.6f}")
        st.session_state.current_params = opt_params.copy()

        # 原始 vs 拟合对比图
        st.markdown("### Original airfoil vs SPIP fitting airfoil")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(UpX, UpY, 'b-', lw=3, label='Original upper')
        ax1.plot(LowX, LowY, 'b-', lw=3, label='Original lower')
        x_fit, zu_fit, zl_fit, _, _ = SPIP_fit(opt_params)
        ax1.plot(x_fit, zu_fit, 'r--', lw=2.5, label='Fitted upper ')
        ax1.plot(x_fit, zl_fit, 'r--', lw=2.5, label='Fitted lower')
        ax1.set_xlabel('x/c', fontsize=14); ax1.set_ylabel('z/c', fontsize=14)
        ax1.grid(True, alpha=0.3); ax1.set_aspect('equal', adjustable='box')
        ax1.set_title("Original vs Fitted", fontsize=15, pad=15)
        ax1.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot(fig1)

        # 重置 + 固定按钮
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("重置参数", type="primary", key="reset_btn"):
                st.session_state.current_params = opt_params.copy()
                st.session_state.params_fixed = False
                for i in range(7):
                    p = camber[i]; val_rad = opt_params[idx[p]]
                    if p in ['alpha_LE', 'alpha_TE']:
                        st.session_state[f"L{i}"] = round(np.degrees(val_rad), 5)
                    else:
                        st.session_state[f"L{i}"] = float(val_rad)
                    p = thick[i]; val_rad = opt_params[idx[p]]
                    if p == 'beta_TE':
                        st.session_state[f"R{i}"] = round(np.degrees(val_rad), 5)
                    else:
                        st.session_state[f"R{i}"] = float(val_rad)
                st.rerun()
        with col_btn2:
            if st.button("固定参数" if not st.session_state.params_fixed else "解锁参数", type="primary"):
                st.session_state.params_fixed = not st.session_state.params_fixed
                st.rerun()

        # 14个参数手动调节
        st.markdown("### 参数手动调节（14个SPIP参数）")
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("<div style='text-align:center;font-size:19px;font-weight:bold;color:#00d4ff'>弯度相关参数</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='text-align:center;font-size:19px;font-weight:bold;color:#ff6b6b'>厚度相关参数</div>", unsafe_allow_html=True)

        for i in range(7):
            left, right = st.columns([1, 1])
            with left:
                p = camber[i]
                v = st.session_state.current_params[idx[p]]
                label = f"{sym[p]} {name_cn[p]}（参考范围：{rng[p]}）"
                if p in ['alpha_LE', 'alpha_TE']:
                    deg = np.degrees(v)
                    new_deg = st.number_input(label, value=round(deg, 5), step=0.01, format="%.5f", key=f"L{i}")
                    st.session_state.current_params[idx[p]] = np.radians(new_deg)
                    new_rad = np.radians(new_deg)
                    st.markdown(
                        f"<small style='color:#00d4ff'>→ {new_deg:+.3f}°</small> "
                        f"<small style='color:#ff6ec7'>(∆θ = {new_rad:+.5f} rad)</small>",
                        unsafe_allow_html=True
                    )
                else:
                    step = 1e-6 if abs(v) < 1e-3 else abs(v) * 0.02
                    new_v = st.number_input(label, value=float(v), step=step, format="%.7f", key=f"L{i}")
                    st.session_state.current_params[idx[p]] = new_v

            with right:
                p = thick[i]
                v = st.session_state.current_params[idx[p]]
                label = f"{sym[p]} {name_cn[p]}（参考范围：{rng[p]}）"
                if p == 'beta_TE':
                    deg = np.degrees(v)
                    new_deg = st.number_input(label, value=round(deg, 5), min_value=0.0, max_value=60.0, step=0.01,
                                              format="%.5f", key=f"R{i}")
                    st.session_state.current_params[idx[p]] = np.radians(new_deg)
                    new_rad = np.radians(new_deg)
                    st.markdown(
                        f"<small style='color:#00d4ff'>→ {new_deg:+.3f}°</small> "
                        f"<small style='color:#ff6ec7'>(∆θ = {new_rad:+.5f} rad)</small>",
                        unsafe_allow_html=True
                    )
                else:
                    step = 1e-6 if p in ['R_LE', 'delta_Z_TE'] else 1e-4 if p in ['XT', 'T'] else 0.1
                    new_v = st.number_input(label, value=float(v), step=step, format="%.7f", key=f"R{i}")
                    st.session_state.current_params[idx[p]] = new_v

        # 实时重构显示
        st.markdown("### 当前参数实时生成的翼型")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        try:
            x_new, zu_new, zl_new, _, _ = SPIP_fit(st.session_state.current_params)
            xp = np.concatenate([x_new[::-1], x_new])
            zp = np.concatenate([zl_new[::-1], zu_new])
            ax2.plot(xp, zp, 'g-', lw=2, label='airfoil of current params')
            ax2.plot(UpX, UpY, 'b-', alpha=0.25)
            ax2.plot(LowX, LowY, 'b-', alpha=0.25, label='original airfoil')
            ax2.set_xlabel('x/c'); ax2.set_ylabel('z/c')
            ax2.grid(True, alpha=0.3); ax2.set_aspect('equal', adjustable='box')
            ax2.set_title("reconstructed airfoil", fontsize=15, pad=15)
            ax2.legend(fontsize=11, loc='center left', bbox_to_anchor=(1, 0.5))

            is_valid, reasons = detect_deformed_airfoil(x_new, zu_new, x_new, zl_new)
            st.write("**畸形检测：**", "翼型光滑有效" if is_valid else "检测到畸形！")
            if not is_valid:
                for r in reasons: st.warning("• " + r)
        except Exception as e:
            ax2.text(0.5, 0.5, "参数超出范围\n无法生成", ha='center', va='center',
                     transform=ax2.transAxes, fontsize=20, color='red')
            st.error(f"生成失败：{e}")
        st.pyplot(fig2)

        # ==================== 下载当前翼型（改进版：无重复前缘点） ====================
        st.markdown("### 下载当前翼型")
        if st.button("下载当前翼型坐标 (.dat 格式)", type="primary"):
            try:
                x_new, zu_new, zl_new, _, _ = SPIP_fit(st.session_state.current_params)
                content = save_airfoil_dat(x_new, zu_new, zl_new,
                                         name=f"{Path(uploaded_file.name).stem}_SPIP")
                st.download_button("点击下载 .dat 文件", content,
                                   file_name=f"{Path(uploaded_file.name).stem}_SPIP.dat",
                                   mime="text/plain")
            except Exception as e:
                st.error(f"当前参数无效: {e}")

        # 批量生成
        st.markdown("---")
        st.markdown("### 批量生成变体翼型族")
        c1, c2, c3, c4 = st.columns(4)
        with c1: n_samples = st.number_input("生成数量", 1, 1000, 50, 10)
        with c2: variation = st.slider("扰动幅度 (%)", 1, 30, 20) / 100
        with c3: detect_on = st.checkbox("开启畸形筛除", True)
        with c4: seed = st.number_input("随机种子", value=42)

        st.markdown("### 选择扰动参数")
        st.markdown("**点击复选框选择扰动参数：**")
        if 'selected_params' not in st.session_state:
            st.session_state.selected_params = COL_NAMES

        col_left, col_right = st.columns(2)
        with col_left:
            st.markdown("#### 弯度相关参数")
            for param in camber:
                st.checkbox(f"{sym[param]} {name_cn[param]}", value=param in st.session_state.selected_params, key=f"check_{param}")
        with col_right:
            st.markdown("#### 厚度相关参数")
            for param in thick:
                st.checkbox(f"{sym[param]} {name_cn[param]}", value=param in st.session_state.selected_params, key=f"check_{param}")
        st.info(f"当前已选择扰动参数数量：{len([k for k in st.session_state.keys() if k.startswith('check_') and st.session_state[k]])} / {len(COL_NAMES)}")

        if st.button("开始批量生成并可视化", type="primary"):
            with st.spinner("正在生成中..."):
                df = generate_and_plot_airfoils(
                    st.session_state.current_params,
                    num_samples=n_samples,
                    variation_pct=variation,
                    seed=int(seed),
                    distortion_detect=detect_on
                )
            valid_df = df[~df['distortion']] if detect_on else df
            st.success(f"生成完成！有效翼型：{len(valid_df)} 个")

            fig3, ax3 = plt.subplots(figsize=(9, 7))
            for _, row in valid_df.iterrows():
                p = row[COL_NAMES].to_numpy()
                try:
                    x, zu, zl, _, _ = SPIP_fit(p)
                    ax3.plot(np.concatenate([x[::-1], x]), np.concatenate([zl[::-1], zu]), color='lightblue', alpha=0.7, lw=1)
                except: continue
            xb, zub, zlb, _, _ = SPIP_fit(st.session_state.current_params)
            ax3.plot(np.concatenate([xb[::-1], xb]), np.concatenate([zlb[::-1], zub]), 'darkred', lw=2, label='base airfoil')
            ax3.set_aspect('equal'); ax3.grid(alpha=0.3)
            ax3.set_xlabel('x/c'); ax3.set_ylabel('z/c')
            ax3.set_title(f"变体翼型族（共 {len(valid_df)} 个有效）", fontsize=15)
            ax3.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5))
            st.pyplot(fig3)

            if len(valid_df) > 0:
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for _, row in valid_df.iterrows():
                        p = row[COL_NAMES].to_numpy()
                        try:
                            x, zu, zl, _, _ = SPIP_fit(p)
                            content = save_airfoil_dat(x, zu, zl, name=f"Airfoil_{int(row['ID']):05d}")
                            zf.writestr(f"airfoil_{int(row['ID']):05d}.dat", content)
                        except: continue
                buffer.seek(0)
                st.download_button(f"下载全部 {len(valid_df)} 个有效翼型 (.zip)", buffer, "SPIP_airfoils.zip", "application/zip")

    else:
        # 原始欢迎页面
        st.markdown("""
            <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                        padding: 80px; border-radius: 25px; text-align: center; color: white; 
                        margin: 50px 0; box-shadow: 0 20px 50px rgba(0,0,0,0.6);">
                <h1 style="margin:0; color: #00eeff; font-size: 58px; font-weight: 900;">
                    SPIP · 翼型直观参数化设计平台
                </h1>
                <p style="font-size: 26px; margin: 30px 0; color: #a0f7ff;">
                    请上传您的翼型文件，开始翼型设计之旅
                </p>
            </div>
        """, unsafe_allow_html=True)

# ==================================================================
#                        Tab 2: 翼型性能分析
# ==================================================================
def plot_airfoil_preview(coords, title="airfoil of current params"):
    fig, ax = plt.subplots(figsize=(11, 5))
    if isinstance(coords, dict):
        x, y = coords['x'], coords['y']
    else:
        x, y = coords[:, 0], coords[:, 1]
    ax.plot(x, y, color='k', linewidth=2, label=title)

    ax.set_aspect('equal')
    ax.grid(alpha=0.3, color='white')
    ax.set_title(title, fontsize=20, fontweight='bold', color='k', pad=20)
    ax.set_xlabel('x/c', fontsize=14)
    ax.set_ylabel('y/c', fontsize=14)
    plt.tight_layout()
    return fig

with tab2:
    if st.session_state.current_params is None:
        st.warning("请先在「翼型参数化与直观设计」标签页上传并完成拟合")
        st.stop()
    st.subheader("NeuralFoil 快速预测(支持-180°-180°)")
    src_n = st.radio("翼型来源", ["使用当前参数化翼型", "上传独立翼型"], horizontal=True, key="nsrc")

    if src_n == "上传独立翼型":
        nf_file = st.file_uploader("上传翼型坐标文件", type=["dat","txt"], key="nf_up")
        if not nf_file: st.info("等待上传..."); st.stop()
        Path("temp_nf.dat").write_bytes(nf_file.getvalue())
        coords_raw = read_airfoil_data("temp_nf.dat")
        UpX, UpY, LowX, LowY = preprocess_airfoil_data(coords_raw, normalize=True)
        airfoil_coords_n = np.column_stack((np.concatenate([UpX[::-1], UpX]), np.concatenate([LowY[::-1], UpY])))
        airfoil_name_n = nf_file.name.rsplit('.', 1)[0]
    else:
        x, zu, zl, _, _ = SPIP_fit(st.session_state.current_params)
        airfoil_coords_n = np.column_stack((np.concatenate([x[::-1], x]), np.concatenate([zl[::-1], zu])))
        airfoil_name_n = "当前参数化翼型"

    st.pyplot(plot_airfoil_preview(airfoil_coords_n, airfoil_name_n))

    col1n, col2n = st.columns(2)
    with col1n:
        Re_n = st.number_input("雷诺数 Re", 100000.0, 20000000.0, 1000000.0, 100000.0, format="%.0f", key="re_n")
        Ma_n = st.number_input("马赫数 Ma", 0.0, 0.95, 0.30, 0.01, key="ma_n")
        a_min_n = st.number_input("攻角起始 (°)", -180.0, 180.0, -8.0, 0.5, key="amin_n")
    with col2n:
        a_max_n = st.number_input("攻角终止 (°)", -180.0, 180.0, 16.0, 0.5, key="amax_n")
        a_step_n = st.number_input("攻角步长 (°)", 0.25, 5.0, 1.0, 0.25, key="astep_n")

    if st.button("开始 NeuralFoil 计算", type="primary", use_container_width=True):
        with st.spinner("NeuralFoil 极速预测中"):
            alphas_deg = np.arange(a_min_n, a_max_n + a_step_n/2, a_step_n)
            try:
                results = compute_air_aerosandbox(
                    airfoil_coords=airfoil_coords_n,
                    Re=Re_n,
                    Ma=Ma_n,
                    alpha=alphas_deg
                )
                df_n = pd.DataFrame(results)
                st.session_state.neuralfoil_df = df_n
                st.success(f"完成！共 {len(df_n)} 个攻角")
            except Exception as e:
                st.error(f"NeuralFoil 计算出错：{e}")
                st.session_state.neuralfoil_df = pd.DataFrame()

    if not st.session_state.neuralfoil_df.empty:
        dfn = st.session_state.neuralfoil_df
        st.subheader("性能数据表")
        st.dataframe(dfn[['alpha','CL','CD','CM']].round(6), use_container_width=True)
        st.download_button("下载气动性能数据", dfn.to_csv(index=False).encode(),
                           f"{airfoil_name_n}_NeuralFoil_Re{int(Re_n)}_Ma{Ma_n:.2f}.csv", "text/csv")

        st.subheader("气动性能曲线")
        c1, c2 = st.columns(2)
        with c1:
            f1, a1 = plt.subplots(figsize=(6.5,4.8)); a1.plot(dfn['alpha'], dfn['CL'], 'o-', color='#00ffea', lw=3, ms=6)
            a1.set_title('CL', fontweight='bold'); a1.grid(alpha=0.3); st.pyplot(f1)
            f3, a3 = plt.subplots(figsize=(6.5,4.8)); a3.plot(dfn['alpha'], dfn['CM'], 'o-', color='#9c27b0', lw=3, ms=6)
            a3.set_title('CM', fontweight='bold'); a3.grid(alpha=0.3); st.pyplot(f3)
        with c2:
            f2, a2 = plt.subplots(figsize=(6.5,4.8)); a2.plot(dfn['alpha'], dfn['CD'], 'o-', color='#ff4081', lw=3, ms=6)
            a2.set_title('CD', fontweight='bold'); a2.grid(alpha=0.3); st.pyplot(f2)
            f4, a4 = plt.subplots(figsize=(6.5,4.8))
            ld = dfn['CL'] / dfn['CD'].replace(0, np.nan)
            a4.plot(dfn['alpha'], ld, 'o-', color='#ffd700', lw=3.5, ms=6)
            a4.set_title('CL/CD', fontweight='bold', color='#ffd700'); a4.grid(alpha=0.3); st.pyplot(f4)
    else:
        st.info("点击上方按钮开始 NeuralFoil 计算")

st.markdown("<p style='text-align:center; color:#64748b; margin-top:100px; font-size:20px;'>SPIP 翼型直观参数化设计平台 ©2025 </p>", unsafe_allow_html=True)
