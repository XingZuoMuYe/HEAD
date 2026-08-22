# 最新版 (V12):
# 1. 修复了右图周车的绘制逻辑（与师兄修复的左图逻辑保持一致）
# 2. 增加了物体类型过滤，只绘制车辆

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime
import matplotlib.patches as patches

def visualize_prediction(batch, prediction, draw_index=0,
                         save_path="prediction_vs_gt.jpg", rotate=180):
    """
    一个健壮的、专业的轨迹预测可视化函数。
    [V12] 修复右图逻辑，并增加物体类型过滤 (只画车辆)。
    """
    print(f"Generating visualization with object filtering (V12) and saving to base path: {save_path}...")

    # ——— 内部辅助函数 (unchanged) ———
    IS_RELATIVE, ZERO_THR, DUP_EPS, JUMP_THR = False, 1e-6, 1e-4, 8.0

    def rotate_xy(xy, deg):
        if deg % 360 == 0: return xy
        rad = np.deg2rad(deg);
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float32);
        return xy @ R.T

    def _valid_rows(xy):
        mask = ~(np.isclose(xy[:, 0], 0.0, atol=ZERO_THR) & np.isclose(xy[:, 1], 0.0, atol=ZERO_THR));
        return xy[mask]

    def _longest_clean_segment(xy):
        if xy.ndim != 2 or xy.shape[0] < 2: return None
        xy = _valid_rows(xy);
        if xy.shape[0] < 2: return None
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        keep = np.r_[True, d > DUP_EPS];
        xy = xy[keep]
        if xy.shape[0] < 2: return None
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        breaks = np.where(d > JUMP_THR)[0]
        if breaks.size == 0: return xy
        starts = np.r_[0, breaks + 1];
        ends = np.r_[breaks + 1, xy.shape[0]]
        seg_lens = ends - starts;
        idx = int(np.argmax(seg_lens))
        seg = xy[starts[idx]:ends[idx]]
        return seg if seg.shape[0] >= 2 else None

    def _extract_modes_for_center(pred_traj_raw, bd, draw_index, fut_T):
        arr = np.array(pred_traj_raw);
        N_agents = int(bd['obj_trajs'][draw_index].shape[0]);
        ego_local = 0
        try:
            if 'track_index_to_predict' in bd: ego_local = int(np.array(bd['track_index_to_predict'][draw_index].cpu()))
        except Exception:
            pass
        if arr.ndim == 3 and arr.shape[-2] == fut_T: return arr
        if arr.ndim == 4:
            if arr.shape[0] == N_agents and arr.shape[2] == fut_T: return arr[ego_local]
            if arr.shape[1] == N_agents and arr.shape[2] == fut_T: return arr[:, ego_local]
            for ax, s in list(enumerate(arr.shape))[::-1]:
                if s == 1: arr = np.squeeze(arr, axis=ax)
            if arr.ndim == 3 and arr.shape[-2] == fut_T: return arr
        M, T = arr.shape[0], arr.shape[-2];
        return arr.reshape(M, T, -1)[..., :2]

    def draw_vehicle_bounding_box(ax, center_x, center_y, heading_rad, length, width, color, zorder, alpha=0.7):
        half_length, half_width = length / 2, width / 2
        corners_local = np.array([[half_length, half_width], [-half_length, half_width], [-half_length, -half_width],
                                  [half_length, -half_width]])
        corners_world = rotate_xy(corners_local, np.rad2deg(heading_rad)) + np.array([center_x, center_y])
        corners_rotated_scene = rotate_xy(corners_world, rotate)
        vehicle_polygon = patches.Polygon(corners_rotated_scene, closed=True, facecolor=color, edgecolor='black',
                                          alpha=alpha, zorder=zorder)
        ax.add_patch(vehicle_polygon)

    # --- Data prep and canvas creation (unchanged) ---
    bd = batch['input_dict'];
    map_lanes = bd['map_polylines'][draw_index].cpu().numpy()
    past_traj = bd['obj_trajs'][draw_index].cpu().numpy();
    future_traj = bd['obj_trajs_future_state'][draw_index].cpu().numpy()
    past_mask = bd['obj_trajs_mask'][draw_index].cpu().numpy()
    pred_prob = prediction['predicted_probability'][draw_index].detach().cpu().numpy();
    pred_raw = prediction['predicted_trajectory'][draw_index].detach().cpu().numpy()
    map_xy = map_lanes[..., :2];
    fut_T = int(future_traj.shape[1])
    try:
        ego_idx = int(np.array(bd['track_index_to_predict'][draw_index].cpu()))
    except Exception:
        ego_idx = 0

    # <<< 修复 2: 提取物体类型，只绘制车辆 >>>
    is_vehicle = np.zeros(past_traj.shape[0], dtype=bool)
    for i in range(past_traj.shape[0]):
        agent_past_mask = past_mask[i]
        if agent_past_mask.any():
            last_valid_idx = np.where(agent_past_mask)[0][-1]
            state = past_traj[i, last_valid_idx]
            # 假设 state[6:9] 是 [vehicle, pedestrian, cyclist] 的 one-hot 编码
            if state.shape[0] > 8:
                obj_type = np.argmax(state[6:9])
                if obj_type == 0:  # 0 是车辆
                    is_vehicle[i] = True
            else:  # 如果没有类型信息，为兼容旧数据，默认视为车辆
                is_vehicle[i] = True
    # <<< 修复 2 结束 >>>

    clouds = [];
    for lane in map_xy: clouds.append(_valid_rows(lane))
    for arr in past_traj: clouds.append(_valid_rows(arr[..., :2]))
    for arr in future_traj: clouds.append(_valid_rows(arr[..., :2]))
    clouds = [c for c in clouds if c is not None and c.size > 0]
    if len(clouds):
        all_xy = rotate_xy(np.concatenate(clouds, axis=0), rotate);
        mn, mx = all_xy.min(axis=0), all_xy.max(axis=0);
        pad = 5.0
        xlim, ylim = (float(mn[0] - pad), float(mx[0] + pad)), (float(mn[1] - pad), float(mx[1] + pad))
    else:
        xlim, ylim = (-35, 35), (-35, 35)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300, constrained_layout=True)
    axes[0].set_title("Ground Truth");
    axes[1].set_title("Predicted Trajectories")
    for ax in axes:
        ax.set_aspect('equal');
        ax.axis('off');
        ax.set_xlim(*xlim);
        ax.set_ylim(*ylim)
        map_type = np.argmax(map_lanes[..., 0, -20:], axis=-1)
        for i, lane in enumerate(map_xy):
            lane_type = map_type[i]
            if lane_type == 0: continue
            lane_points = _valid_rows(lane)
            if lane_points.shape[0] < 2: continue
            rotated_points = rotate_xy(lane_points, rotate)
            if lane_type in [1, 2, 3]:
                style = {'color': 'grey', 'linestyle': 'dotted', 'linewidth': 1.0}
            else:
                style = {'color': 'grey', 'linestyle': '-', 'linewidth': 0.2}
            ax.plot(rotated_points[:, 0], rotated_points[:, 1], zorder=1, **style)

    def draw_poly(ax, arr, color, lw=1.6, alpha=0.95):
        xy = _valid_rows(arr[..., :2])
        if xy.shape[0] >= 2:
            xy = rotate_xy(xy, rotate)
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha, zorder=3)

    # --- Ego car t=0 state (师兄修改的逻辑) ---
    ego_past_mask = past_mask[ego_idx]
    if ego_past_mask.any():
        last_valid_idx = np.where(ego_past_mask)[0][-1]
        length, width = past_traj[ego_idx, last_valid_idx, 3:5]
        start_state = past_traj[ego_idx, last_valid_idx]
        start_pos = start_state[:2]
        sin_h, cos_h = start_state[33:35] if start_state.shape[0] > 34 else (0, 1)
        start_heading = np.arctan2(sin_h, cos_h)
        if length < 0.1 or width < 0.1: length, width = 4.5, 2.0
    else:
        length, width, start_pos, start_heading = 0, 0, None, None
        print("幽灵车出现")

    # --- Left Plot: Ground Truth (使用师兄的逻辑 + 类型过滤) ---
    for i, arr in enumerate(past_traj):
        if not is_vehicle[i]: continue  # <<< 过滤
        draw_poly(axes[0], arr, color='black' if i == ego_idx else '#AAAAAA', lw=2.0 if i == ego_idx else 1.0)
    if start_pos is not None:
        draw_vehicle_bounding_box(axes[0], start_pos[0], start_pos[1], start_heading, length, width, color='#d62728',
                                  zorder=5, alpha=0.6)
    for i, arr in enumerate(future_traj):
        if not is_vehicle[i]: continue  # <<< 过滤
        draw_poly(axes[0], arr, color='#d62728' if i == ego_idx else '#FFDAB9', lw=2.5 if i == ego_idx else 1.5)

        # --- 周车 t=0 框 (师兄修复的逻辑) ---
        if i != ego_idx:
            valid_future_points = _valid_rows(arr)
            agent_past_mask = past_mask[i]
            if agent_past_mask.any():
                last_valid_idx_other = np.where(agent_past_mask)[0][-1]
                state_other = past_traj[i, last_valid_idx_other]
                final_pos = state_other[:2]
                length_other, width_other = past_traj[i, last_valid_idx_other, 3:5]
                if state_other.shape[0] > 34:
                    sin_h, cos_h = state_other[33:35]
                    final_heading = np.arctan2(sin_h, cos_h)
                else:
                    if valid_future_points.shape[0] >= 2:  # Fallback
                        delta_pos = valid_future_points[-1, :2] - valid_future_points[-2, :2]
                        final_heading = np.arctan2(delta_pos[1], delta_pos[0])
                    else:
                        final_heading = 0.0  # 无法确定朝向

                if length_other < 0.1 or width_other < 0.1: length_other, width_other = 4.5, 2.0
                draw_vehicle_bounding_box(axes[0], final_pos[0], final_pos[1], final_heading, length_other, width_other,
                                          color='#FF8C00', zorder=5)

    # --- Right Plot: Prediction (同步师兄的逻辑 + 类型过滤) ---
    draw_poly(axes[1], past_traj[ego_idx], color='black', lw=2.0, alpha=0.8)  # 自车历史
    if start_pos is not None:  # 自车 t=0 框
        draw_vehicle_bounding_box(axes[1], start_pos[0], start_pos[1], start_heading, length, width, color='#d62728',
                                  zorder=5, alpha=0.6)

    # 周车历史
    for i, arr in enumerate(past_traj):
        if not is_vehicle[i]: continue  # <<< 过滤
        if i == ego_idx: continue
        draw_poly(axes[1], arr, color='#AAAAAA', lw=1.0, alpha=0.5)

    # 周车未来 + t=0 框
    for i, arr in enumerate(future_traj):
        if not is_vehicle[i]: continue  # <<< 过滤
        if i == ego_idx: continue
        draw_poly(axes[1], arr, color='#FFDAB9', lw=1.5, alpha=0.5)

        # <<< 修复 1: 同步师兄的左图逻辑，修复bug >>>
        valid_future_points = _valid_rows(arr)
        agent_past_mask = past_mask[i]
        if agent_past_mask.any():  # 必须有历史轨迹才能画 t=0 的框
            last_valid_idx_other = np.where(agent_past_mask)[0][-1]
            state_other = past_traj[i, last_valid_idx_other]
            final_pos = state_other[:2]  # t=0 位置
            length_other, width_other = past_traj[i, last_valid_idx_other, 3:5]

            if state_other.shape[0] > 34:
                sin_h, cos_h = state_other[33:35]
                final_heading = np.arctan2(sin_h, cos_h)
            else:
                if valid_future_points.shape[0] >= 2:  # Fallback
                    delta_pos = valid_future_points[-1, :2] - valid_future_points[-2, :2]
                    final_heading = np.arctan2(delta_pos[1], delta_pos[0])
                else:
                    final_heading = 0.0  # 无法确定朝向

            if length_other < 0.1 or width_other < 0.1: length_other, width_other = 4.5, 2.0
            draw_vehicle_bounding_box(axes[1], final_pos[0], final_pos[1], final_heading, length_other, width_other,
                                      color='#FF8C00', zorder=5, alpha=0.5)  # 注意alpha=0.5
        # <<< 修复 1 结束 >>>

    # --- Prediction modes and Legend (unchanged from senior's code) ---
    order = np.argsort(-pred_prob);
    pred_prob, pred_raw = pred_prob[order], pred_raw[order]
    pred_modes = _extract_modes_for_center(pred_raw, bd, draw_index, fut_T)
    n_modes = pred_modes.shape[0]
    cmap = plt.get_cmap('tab10' if n_modes <= 10 else ('tab20' if n_modes <= 20 else 'hsv'), n_modes)
    pred_mode_handles = []
    p_max = np.max(pred_prob) if len(pred_prob) > 0 else 1.0
    for k in range(n_modes):
        p = float(pred_prob[k]);
        color = cmap(k)
        p_normalized = p / p_max;
        p_nonlinear = p_normalized ** 2
        alpha = 0.25 + (1 - 0.25) * p_nonlinear
        linewidth = 2.0 + (3.5 - 2.0) * p_normalized
        seg = _longest_clean_segment(pred_modes[k, :, :2])
        if seg is None: continue
        seg_rotated = rotate_xy(seg, rotate)
        axes[1].plot(seg_rotated[:, 0], seg_rotated[:, 1], color=color, lw=linewidth, alpha=alpha, zorder=4)
        pred_mode_handles.append(Line2D([0], [0], color=color, lw=3.5, label=f"Mode {k} (p={p:.2f})"))

    # 图例逻辑 (t=0框)
    gt_legend_handles = [
        patches.Rectangle((0, 0), 1, 1, facecolor='#d62728', edgecolor='black', alpha=0.6, label='Ego @ t=0'),
        Line2D([0], [0], color='black', lw=2, label='Ego Past (GT)'),
        Line2D([0], [0], color='#d62728', lw=2.5, label='Ego Future (GT)'),
        Line2D([0], [0], color='#AAAAAA', lw=1, label='Other Past (GT)'),
        Line2D([0], [0], color='#FFDAB9', lw=1.5, label='Other Future (GT)'),
        patches.Rectangle((0, 0), 1, 1, facecolor='#FF8C00', edgecolor='black', alpha=0.7, label='Other @ t=0')
    ]
    axes[0].legend(handles=gt_legend_handles, loc='upper right', fontsize='x-small',
                   frameon=True, framealpha=0.7, fancybox=True)

    pred_legend_handles = [
                              patches.Rectangle((0, 0), 1, 1, facecolor='#d62728', edgecolor='black', alpha=0.6,
                                                label='Ego @ t=0'),
                          ] + pred_mode_handles
    axes[1].legend(handles=pred_legend_handles, loc='upper right', ncol=2, fontsize=6,
                   frameon=True, framealpha=0.7, fancybox=True,
                   title='Ego Predictions', title_fontsize=7)

    # --- Save figure (unchanged) ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    root, ext = os.path.splitext(save_path)
    if not ext: ext = ".jpg"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f");
    final_path = f"{root}_{ts}{ext}"
    fig.savefig(final_path, dpi=300, bbox_inches="tight", format=ext.lstrip("."))
    print(f"✅ Image saved to {final_path}")
    plt.close(fig)
    return None
