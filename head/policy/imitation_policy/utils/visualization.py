import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from datetime import datetime
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe


def check_loaded_data(ax, data, index=0):
    agents_all = np.concatenate([data['obj_trajs'][..., :2],
                                 data['obj_trajs_future_state'][..., :2]], axis=-2)
    map_polys = data['map_polylines']

    if agents_all.ndim == 4:
        agents = agents_all[index]
        map_item = map_polys[index]
        ego_index = int(data['track_index_to_predict'][index])
    else:
        agents = agents_all
        map_item = map_polys
        ego_index = int(data['track_index_to_predict'])

    def draw_seg(p1, p2, color, lw=2, alpha=0.8):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw, color=color, alpha=alpha)

    # 画地图（当前逻辑是用当前点 -> 预点(6:8) 作为段）
    for lane in map_item:
        for i in range(len(lane) - 1):
            p1 = lane[i, :2]
            p2 = lane[i, 6:8]
            if not (np.allclose(p1, 0) or np.allclose(p2, 0)):
                draw_seg(p1, p2, color='grey', lw=1, alpha=0.5)

    def draw_traj(traj, lw=2, ego=False):
        T = len(traj)
        for t in range(T - 1):
            p1, p2 = traj[t], traj[t+1]
            if np.allclose(p1, 0) or np.allclose(p2, 0):  # ✅ 有效性判断
                continue
            color = (1 - t/T, 0, t/T) if ego else (0, 1 - t/T, t/T)
            draw_seg(p1, p2, color=color, lw=lw, alpha=0.9 if ego else 0.7)

    for i in range(agents.shape[0]):
        draw_traj(agents[i], lw=2, ego=(i == ego_index))

    ax.set_aspect('equal')
    ax.axis('off')
    return ax




def visualize_batch_data(ax, data):
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- 解码对象（和你第二版一致）----
    def decode_obj_trajs(obj_trajs):
        obj_trajs_xy = obj_trajs[..., :2]
        obj_lw = obj_trajs[..., -1, 3:5]
        obj_type_onehot = obj_trajs[..., -1, 6:9]
        obj_type = np.argmax(obj_type_onehot, axis=-1)
        obj_heading_encoding = obj_trajs[..., -1, 33:35]
        return obj_trajs_xy, obj_lw, obj_type, obj_heading_encoding

    obj_trajs = data['obj_trajs']
    map_polys = data['map_polylines']

    obj_trajs_xy, obj_lw, obj_type, obj_heading = decode_obj_trajs(obj_trajs)
    obj_trajs_future_state = data['obj_trajs_future_state'][..., :2]
    all_traj = np.concatenate([obj_trajs_xy, obj_trajs_future_state], axis=-2)

    # ---- 画轨迹（保持你第二版的配色/分段）----
    def draw_trajectory(trajectory, line_width, ego=False, past_T=4):
        def interpolate_color(start_color, end_color, t, total_t):
            return [(1 - t / total_t) * s + (t / total_t) * e for s, e in zip(start_color, end_color)]
        def draw_line_with_mask(p1, p2, color, lw=4):
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linewidth=lw, color=color, alpha=0.5)

        T = len(trajectory)
        for t in range(T - 1):
            if ego:
                start_color = (0, 0, 0.5); end_color = (0.53, 0.81, 0.98)  # 天蓝渐变
            else:
                start_color = (0, 0.5, 0);   end_color = (0.56, 0.93, 0.56) # 草绿渐变
            color = interpolate_color(start_color, end_color, t, T)
            if trajectory[t, 0] and trajectory[t + 1, 0]:
                draw_line_with_mask(trajectory[t], trajectory[t + 1], color=color, lw=line_width)

    ego_idx = int(data['track_index_to_predict'])
    for i in range(obj_trajs.shape[0]):
        draw_trajectory(all_traj[i], line_width=3, ego=(i == ego_idx))

    # ---- 车辆包络（与你第二版一致）----
    def plot_objects(obj_xy, obj_lw, obj_heading, obj_mask):
        for i in range(len(obj_lw)):
            if obj_mask[i]:
                length, width = obj_lw[i]
                sin_angle, cos_angle = obj_heading[i]
                angle = np.arctan2(sin_angle, cos_angle)
                x, y = obj_xy[i]
                rect = plt.Rectangle((-length/2, -width/2), length, width, angle=0,
                                     facecolor='none', edgecolor='grey', linewidth=1)
                t = ax.transData
                rot = plt.matplotlib.transforms.Affine2D().rotate_around(0, 0, angle).translate(x, y) + t
                rect.set_transform(rot)
                ax.add_patch(rect)

    obj_mask = data['obj_trajs_mask']
    plot_objects(obj_trajs_xy[:, -1], obj_lw, obj_heading, obj_mask[:, -1])

    # ---- 画地图【完全按第一版】----
    # 1) 取坐标
    map_xy = map_polys[..., :2]
    # 2) 类型 one-hot：固定使用 [9:29]（和第一版一致）
    map_type_oh = map_polys[..., 0, 9:29]
    map_type = np.argmax(map_type_oh, axis=-1)  # (L,)

    for idx, t in enumerate(map_type):
        lane = map_xy[idx]
        if t == 0:
            continue
        if t in [1, 2, 3]:
            color = 'grey'; linestyle = 'dotted'; linewidth = 1.0  # 中心线：灰色虚线
        else:
            color = 'grey'; linestyle = '-';       linewidth = 0.2  # 其他：灰色细实线

        for i in range(len(lane) - 1):
            # 第一版判定：只检查 x 是否为 0
            if lane[i, 0] and lane[i + 1, 0]:
                ax.plot([lane[i, 0], lane[i + 1, 0]],
                        [lane[i, 1], lane[i + 1, 1]],
                        linewidth=linewidth, color=color, linestyle=linestyle)

    # ---- 画布设置（同第一版）----
    vis_range = 35
    ax.set_aspect('equal')
    ax.axis('off')
    ax.grid(True)
    ax.set_xlim(-vis_range, vis_range)
    ax.set_ylim(-vis_range, vis_range)
    return ax







def concatenate_images(images, rows, cols):
    # Determine individual image size
    width, height = images[0].size

    # Create a new image with the total size
    total_width = width * cols
    total_height = height * rows
    new_im = Image.new('RGB', (total_width, total_height))

    # Paste each image into the new image
    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        new_im.paste(image, (col * width, row * height))

    return new_im


def concatenate_varying(image_list, column_counts):
    if not image_list or not column_counts:
        return None

    # Assume all images have the same size, so we use the first one to calculate ratios
    original_width, original_height = image_list[0].size
    total_height = original_height * column_counts[0]  # Total height is based on the first column

    columns = []  # To store each column of images

    start_idx = 0  # Starting index for slicing image_list

    for count in column_counts:
        # Calculate new height for the current column, maintaining aspect ratio
        new_height = total_height // count
        scale_factor = new_height / original_height
        new_width = int(original_width * scale_factor)

        column_images = []
        for i in range(start_idx, start_idx + count):
            # Resize image proportionally
            resized_image = image_list[i].resize((new_width, new_height), Image.Resampling.LANCZOS)
            column_images.append(resized_image)

        # Update start index for the next batch of images
        start_idx += count

        # Create a column image by vertically stacking the resized images
        column = Image.new('RGB', (new_width, total_height))
        y_offset = 0
        for img in column_images:
            column.paste(img, (0, y_offset))
            y_offset += img.height

        columns.append(column)

    # Calculate the total width for the new image
    total_width = sum(column.width for column in columns)

    # Create the final image to concatenate all column images
    final_image = Image.new('RGB', (total_width, total_height))
    x_offset = 0
    for column in columns:
        final_image.paste(column, (x_offset, 0))
        x_offset += column.width

    return final_image


from matplotlib.lines import Line2D

def visualize_prediction(batch, prediction, draw_index=0,
                         save_path="/home/dw/HEAD/tests/figure/prediction_vs_gt.jpg",
                         rotate=180):
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # 如果 predicted_trajectory 是相对位移(Δx,Δy)，把它设 True（常见于一些策略头）
    IS_RELATIVE = False

    # —— 可调参数：去噪 & 断裂判定 —— #
    ZERO_THR   = 1e-6   # 判定“无效点”的阈值：近似(0,0)
    DUP_EPS    = 1e-4   # 判定“相邻重复点”的阈值
    JUMP_THR   = 8.0    # 判定“异常大跳变”的距离阈值（米），超过则断开

    def rotate_xy(xy, deg):
        if deg % 360 == 0:
            return xy
        rad = np.deg2rad(deg)
        c, s = np.cos(rad), np.sin(rad)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        return xy @ R.T

    def _valid_rows(xy):
        """过滤近似(0,0)的行"""
        mask = ~(np.isclose(xy[:, 0], 0.0, atol=ZERO_THR) & np.isclose(xy[:, 1], 0.0, atol=ZERO_THR))
        return xy[mask]

    def _longest_clean_segment(xy):
        """
        1) 去掉相邻重复点
        2) 按“异常大跳变”打断
        3) 取最长连续段
        返回 (K,2) 或 None
        """
        if xy.ndim != 2 or xy.shape[0] < 2:
            return None

        # 只保留有效点
        xy = _valid_rows(xy)
        if xy.shape[0] < 2:
            return None

        # 去掉相邻重复点
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        keep = np.r_[True, d > DUP_EPS]
        xy = xy[keep]
        if xy.shape[0] < 2:
            return None

        # 在大跳变处断开
        d = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        breaks = np.where(d > JUMP_THR)[0]
        if breaks.size == 0:
            return xy

        starts = np.r_[0, breaks + 1]
        ends   = np.r_[breaks + 1, xy.shape[0]]
        seg_lens = ends - starts
        idx = int(np.argmax(seg_lens))
        seg = xy[starts[idx]:ends[idx]]
        return seg if seg.shape[0] >= 2 else None

    def _extract_modes_for_center(pred_traj_raw, bd, draw_index, fut_T):
        """
        统一返回中心目标的 (M, T, 2)
        支持形状：
          (M, T, 2)
          (N_agents, M, T, 2)  -> 取中心/ego
          (M, N_agents, T, 2)  -> 取中心/ego
          以及上述变体里含有 size=1 的维度（自动 squeeze）
        """
        arr = np.array(pred_traj_raw)
        N_agents = int(bd['obj_trajs'][draw_index].shape[0])

        ego_local = 0
        try:
            if 'track_index_to_predict' in bd:
                ego_local = int(np.array(bd['track_index_to_predict'][draw_index].detach().cpu()))
        except Exception:
            pass

        if arr.ndim == 3 and arr.shape[-2] == fut_T:
            return arr  # (M,T,2)
        if arr.ndim == 4:
            # (N_agents, M, T, 2)
            if arr.shape[0] == N_agents and arr.shape[2] == fut_T:
                return arr[ego_local]
            # (M, N_agents, T, 2)
            if arr.shape[1] == N_agents and arr.shape[2] == fut_T:
                return arr[:, ego_local]
            # squeeze 多余 1 维
            for ax, s in list(enumerate(arr.shape))[::-1]:
                if s == 1:
                    arr = np.squeeze(arr, axis=ax)
            if arr.ndim == 3 and arr.shape[-2] == fut_T:
                return arr
        # 最后兜底：尝试 reshape 成 (M,T,2)
        M = arr.shape[0]
        T = arr.shape[-2]
        return arr.reshape(M, T, -1)[..., :2]

    # --- 取数据 ---
    bd = batch['input_dict']
    map_lanes  = bd['map_polylines'][draw_index].detach().cpu().numpy()              # (L, P, C)
    past_traj  = bd['obj_trajs'][draw_index].detach().cpu().numpy()                  # (N, T_past, C)
    future_traj= bd['obj_trajs_future_state'][draw_index].detach().cpu().numpy()     # (N, T_fut, C)
    pred_prob  = prediction['predicted_probability'][draw_index].detach().cpu().numpy()  # (M,)
    pred_raw   = prediction['predicted_trajectory'][draw_index].detach().cpu().numpy()

    map_xy = map_lanes[..., :2]
    fut_T  = int(future_traj.shape[1])

    # --- 视野范围 ---
    clouds = []
    for lane in map_xy:        clouds.append(_valid_rows(lane))
    for arr in past_traj:      clouds.append(_valid_rows(arr[..., :2]))
    for arr in future_traj:    clouds.append(_valid_rows(arr[..., :2]))
    clouds = [c for c in clouds if c.size > 0]

    if len(clouds):
        all_xy = rotate_xy(np.concatenate(clouds, axis=0), rotate)
        mn = all_xy.min(axis=0); mx = all_xy.max(axis=0)
        pad = 5.0
        xlim = (float(mn[0] - pad), float(mx[0] + pad))
        ylim = (float(mn[1] - pad), float(mx[1] + pad))
    else:
        xlim = (-35, 35); ylim = (-35, 35)

    # --- 画布 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300, constrained_layout=True)
    axes[0].set_title("Ground Truth")
    axes[1].set_title("Predicted Trajectories")
    for ax in axes:
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)

        # --- 地图 ---
        map_type = np.argmax(map_lanes[..., 0, -20:], axis=-1)

        for i, lane in enumerate(map_xy):
            lane_type = map_type[i]

            # --- 以下为第一版的完整画线逻辑 ---

            # 1. 忽略类型为 0 的线
            if lane_type == 0:
                continue

            lane_points = _valid_rows(lane)
            if lane_points.shape[0] < 2:
                continue

            rotated_points = rotate_xy(lane_points, rotate)

            # 2. 根据类型分配样式
            if lane_type in [1, 2, 3]:
                # 中心线样式
                style = {'color': 'grey', 'linestyle': 'dotted', 'linewidth': 1.0}
            else:
                # 其他线样式
                style = {'color': 'grey', 'linestyle': '-', 'linewidth': 0.2}

            # 在两个子图上都画出这条线
            for ax in axes:
                ax.plot(rotated_points[:, 0], rotated_points[:, 1], zorder=1, **style)
    # --- 左：GT（过去蓝，未来橙） ---
    def draw_poly(ax, arr, color, lw=1.6, alpha=0.95):
        xy = _valid_rows(arr[..., :2])
        if xy.shape[0] >= 2:
            xy = rotate_xy(xy, rotate)
            ax.plot(xy[:, 0], xy[:, 1], color=color, lw=lw, alpha=alpha, zorder=3)

    for arr in past_traj:   draw_poly(axes[0], arr, (0.25, 0.45, 1.0))
    for arr in future_traj: draw_poly(axes[0], arr, (1.0, 0.5, 0.2))

    # --- 右：预测多-模态（仅中心目标的各个 mode） ---

    # 1. 在右图上，画出被预测对象的历史轨迹作为参考
    try:
        ego_idx = int(np.array(bd['track_index_to_predict'][draw_index].detach().cpu()))
        ego_past_traj = past_traj[ego_idx]
        draw_poly(axes[1], ego_past_traj, color=(0.25, 0.45, 1.0), lw=1.6, alpha=0.6)
    except Exception as e:
        print(f"提醒：未能绘制历史轨迹到预测图上。错误: {e}")

    # 2. 对预测结果按概率从高到低排序（只执行一次）
    order = np.argsort(-pred_prob)
    pred_prob = pred_prob[order]
    pred_modes = _extract_modes_for_center(pred_raw, bd, draw_index, fut_T)[order]

    if IS_RELATIVE:
        pred_modes = np.cumsum(pred_modes, axis=1)

    # 3. 初始化颜色、图例等设置
    n_modes = pred_modes.shape[0]
    cmap = plt.get_cmap('tab10' if n_modes <= 10 else ('tab20' if n_modes <= 20 else 'hsv'), n_modes)
    legend_handles = []

    # --- 新逻辑：基于当前场景的最高概率进行归一化 ---
    # a. 找到当前所有模式中的最高概率
    p_max = np.max(pred_prob) if len(pred_prob) > 0 else 1.0

    # 4. 循环绘制每一种预测模式
    for k in range(n_modes):
        p = float(pred_prob[k])
        color = cmap(k)

        # b. 将当前概率p归一化，让p_max对应的值为1.0
        p_normalized = p / p_max

        # c. 对归一化后的概率进行平方，拉开视觉差距
        p_nonlinear = p_normalized ** 2

        # d. 计算透明度，现在最高概率的轨迹alpha绝对是1.0
        MIN_ALPHA = 0.25
        alpha = MIN_ALPHA + (1 - MIN_ALPHA) * p_nonlinear

        # e. 让线宽也基于归一化后的概率变化，突出最强者
        MIN_LW = 2.0
        MAX_LW = 3.5
        linewidth = MIN_LW + (MAX_LW - MIN_LW) * p_normalized

        seg = _longest_clean_segment(pred_modes[k, :, :2])
        if seg is None:
            continue
        seg = rotate_xy(seg, rotate)

        # 使用新参数绘图
        axes[1].plot(seg[:, 0], seg[:, 1], color=color, lw=linewidth, alpha=alpha, zorder=4)

        # 图例
        legend_handles.append(Line2D([0], [0], color=color, lw=MAX_LW, label=f"Mode {k} (p={p:.2f})"))

    axes[1].legend(handles=legend_handles, loc='upper right', frameon=False, fontsize=7, ncol=1)
    # ... 函数的其余部分保持不变 ...
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 给文件名加时间戳
    root, ext = os.path.splitext(save_path)
    if not ext:
        ext = ".jpg"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    final_path = f"{root}_{ts}{ext}"

    fig.savefig(final_path, dpi=300, bbox_inches="tight", format=ext.lstrip("."))
    print(f"✅ 图像已保存到 {final_path}")
    plt.close(fig)
    return None


