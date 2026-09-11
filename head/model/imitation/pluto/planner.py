import time
import torch
import numpy as np
from scipy.special import softmax

from head.model.imitation.pluto.features.builder import PlutoTestDataset
from head.model.imitation.pluto.model.pluto_model import PlanningModel


class PlutoInference:
    def __init__(self, cfg):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.dataset = PlutoTestDataset(cfg, is_validation=True)

    def _extract_predicted_agent_tokens(self, pluto_feature, pred, controlled_agent_id=None):
        agent_tokens = pluto_feature.get("agent_tokens", None)
        if agent_tokens is None:
            return []
        tokens = list(agent_tokens)
        if pred is None:
            return tokens
        num_pred = int(len(pred))
        if len(tokens) == num_pred + 1:
            if controlled_agent_id is not None and str(tokens[0]) == str(controlled_agent_id):
                return tokens[1:]
            return tokens[1:]
        return tokens[:num_pred]

    def _extract_cbv_ego_future(self, current_state, pluto_feature, pred, controlled_agent_id=None):
        if pred is None or len(pred) == 0:
            return None
        metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
        sdc_id = metadata.get("sdc_id", None)
        pred_tokens = self._extract_predicted_agent_tokens(pluto_feature, pred,
                                                           controlled_agent_id=controlled_agent_id)
        if sdc_id is not None:
            for idx, token in enumerate(pred_tokens):
                if str(token) == str(sdc_id) and idx < len(pred):
                    return np.asarray(pred[idx], dtype=np.float32)
        return np.asarray(pred[0], dtype=np.float32)

    def _build_cbv_rollout_sample(
            self,
            current_state,
            pluto_feature,
            out,
            pred_local,
            timestep,
            controlled_agent_id,
    ):
        if controlled_agent_id is None:
            return None
        raw_logits = out.get("probability", None)
        raw_traj = out.get("trajectory", None)
        if raw_logits is None:
            return None
        if raw_traj is None:
            raw_traj = out.get("candidate_trajectories", None)
        if raw_traj is None:
            return None

        raw_logits = raw_logits[0].detach().cpu().numpy().astype(np.float32)
        raw_traj = raw_traj[0].detach().cpu().numpy().astype(np.float32)
        if raw_logits.ndim != 2 or raw_traj.ndim != 4:
            return None

        n_ref, n_mode = raw_logits.shape
        candidate_ref_idx = np.repeat(np.arange(n_ref, dtype=np.int64)[:, None], n_mode, axis=1)
        ego_future = self._extract_cbv_ego_future(
            current_state,
            pluto_feature,
            pred_local,
            controlled_agent_id=controlled_agent_id,
        )
        if ego_future is None:
            return None

        metadata = current_state.get("metadata", {}) if isinstance(current_state, dict) else {}
        return {
            "pluto_feature": {"data": pluto_feature},
            "old_logits": raw_logits,
            "candidate_trajectories": raw_traj,
            "ego_future": np.asarray(ego_future, dtype=np.float32),
            "candidate_ref_idx": candidate_ref_idx,
            "predictions": np.asarray(pred_local, dtype=np.float32),
            "info": {
                "frame_idx": int(timestep),
                "cbv_id": str(controlled_agent_id),
                "controlled_agent_id": str(controlled_agent_id),
                "scenario_token": metadata.get("scenario_id", metadata.get("token", None)),
            },
        }

    def initialize_model(self):
        """Initialize the model for Pluto."""
        self.model = PlanningModel(config=self.cfg)
        model_ckpt = self.cfg.get('ckpt_path', None)
        if model_ckpt is not None and model_ckpt != "null":
            ckpt = torch.load(model_ckpt, map_location=self.device, weights_only=False)
            if 'state_dict' in ckpt:
                state_dict = {k.replace('model.', ''): v for k, v in ckpt['state_dict'].items()}
                self.model.load_state_dict(state_dict, strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def run_inference(self, current_state, timestep, controlled_agent_id=None):
        t1 = time.time()
        with torch.no_grad():
            [pluto_feature] = self.dataset.process_scenario(
                current_state,
                timestep,
                # timestep - 1,
                controlled_agent_id=controlled_agent_id,
            )
            # print('更新状态',current_state['tracks']['ego']['state']['position'][:,:2])
            # print('特征',pluto_feature['agent']['position'][0])
            t2 = time.time()
            ref_lines = pluto_feature['reference_line']

            # batch is the standard UniTraj batch_dict: {'batch_size', 'input_dict', ...}
            from head.model.imitation.pluto.features.pluto_utils import collate_pluto_dicts, to_feature_tensor_dict
            input_dict = collate_pluto_dicts([to_feature_tensor_dict(pluto_feature)])
            t3 = time.time()

            def to_device(data, device):
                if isinstance(data, torch.Tensor):
                    return data.to(device)
                elif isinstance(data, dict):
                    return {k: to_device(v, device) for k, v in data.items()}
                elif isinstance(data, list):
                    return [to_device(v, device) for v in data]
                return data

            input_dict = to_device(input_dict, self.device)
            t4 = time.time()

            out = self.model.forward(input_dict)
            t5 = time.time()

            planner_candidates = out["candidate_trajectories"][0].detach().cpu().numpy()
            control_candidates = (out["trajectory"][0].detach().cpu().numpy() if out["trajectory"] is not None else None)
            pred = out['output_prediction'][0].detach().cpu().numpy()

            probability = out["probability"][0].cpu().numpy()

            n_ref, n_mode, T, C = planner_candidates.shape
            planner_candidates = planner_candidates.reshape(-1, T, C)
            probability = probability.reshape(-1)
            if control_candidates is not None and len(control_candidates.shape) == 4:
                control_candidates = control_candidates.reshape(
                    n_ref * n_mode, control_candidates.shape[-2], control_candidates.shape[-1]
                )

            topk = self.cfg.get('candidate_max_num', 20)
            sorted_idx = np.argsort(-probability)
            sorted_candidate_trajectories = planner_candidates[sorted_idx][:topk]
            sorted_control_candidates = None
            if control_candidates is not None:
                sorted_control_candidates = control_candidates[sorted_idx][:topk]
            sorted_probability = softmax(probability[sorted_idx][:topk])
            sorted_ref_idx = None
            if n_ref is not None and n_mode not in (None,0):
                sorted_ref_idx = (sorted_idx[:topk]//n_mode).astype(np.int64)

            input_data = input_dict.data if hasattr(input_dict, 'data') else input_dict

            sorted_candidate_trajectories_local = np.array(sorted_candidate_trajectories, copy=True)
            pred_local = np.array(pred, copy=True)

            planner_candidates_for_eval_local = sorted_candidate_trajectories_local
            if sorted_candidate_trajectories_local.shape[1] > 0:
                planner_candidates_for_eval_local = np.concatenate(
                    [
                        sorted_candidate_trajectories_local[..., 0:1, :],
                        sorted_candidate_trajectories_local,
                    ],
                    axis=-2,
                )

            # Evaluate in the model/local frame so candidates, predictions, reference lines,
            # map features, and agent features stay in the same coordinate system.
            neural_only = self.cfg.get('trajectory_selection_mode', 'hybrid') == 'neural_only'
            evaluator = None
            rule_based_scores = np.zeros_like(sorted_probability)
            if not neural_only:
                from head.model.imitation.pluto.trajectory_evaluator import TrajectoryEvaluator
                evaluator_config = dict(self.cfg)
                builder = getattr(self.dataset, "_builder", None)
                if builder is not None and hasattr(builder, "_get_rear_axle_to_center"):
                    evaluator_config["rear_axle_to_center"] = builder._get_rear_axle_to_center()
                ego_shape = input_data.get("agent", {}).get("shape")
                if ego_shape is not None and ego_shape.shape[1] > 0:
                    ego_width, ego_length = (
                        ego_shape[0, 0, -1].detach().cpu().numpy().tolist()
                    )
                    evaluator_config["ego_width"] = float(ego_width)
                    evaluator_config["ego_length"] = float(ego_length)
                evaluator = TrajectoryEvaluator(evaluator_config)
                _debug_route_ids = None
                if ref_lines is not None and isinstance(ref_lines, dict):
                    _debug_route_ids = ref_lines.get("route_id", None)
                    if _debug_route_ids is not None:
                        _debug_route_ids = np.asarray(_debug_route_ids).tolist()
                rule_based_scores = evaluator.evaluate(
                    planner_candidates_for_eval_local,
                    input_data,
                    pred_local,
                    candidate_ref_idx=sorted_ref_idx,
                    debug_context={
                        "timestep": int(timestep),
                        "sorted_idx": sorted_idx[:topk].tolist(),
                        "route_ids": _debug_route_ids,
                        "max_debug_candidates": min(12, int(topk)),
                    },
                )

            origin = input_data['origin'][0].detach().cpu().numpy()[:2]
            angle = float(input_data['angle'][0].detach().cpu().numpy())
            rot_mat = np.array(
                [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
            )
            sorted_candidate_trajectories[..., :2] = (
                    np.matmul(sorted_candidate_trajectories[..., :2], rot_mat) + origin
            )
            sorted_candidate_trajectories[..., 2] += angle

            if sorted_control_candidates is not None:
                sorted_control_candidates = np.array(sorted_control_candidates, copy=True)
                sorted_control_candidates[..., :2] = (
                        np.matmul(sorted_control_candidates[..., :2], rot_mat) + origin
                )
                if sorted_control_candidates.shape[-1] >= 4:
                    sorted_control_candidates[..., 2:4] = np.matmul(
                        sorted_control_candidates[..., 2:4], rot_mat
                    )
                elif sorted_control_candidates.shape[-1] >= 3:
                    heading = sorted_control_candidates[..., 2] + angle
                    sorted_control_candidates[..., 2] = heading
                if sorted_control_candidates.shape[-1] >= 6:
                    sorted_control_candidates[..., 4:6] = np.matmul(
                        sorted_control_candidates[..., 4:6], rot_mat
                    )

            pred[..., :2] = np.matmul(pred[..., :2], rot_mat) + origin
            pred[..., 2] += angle
            pred[..., 3:5] = np.matmul(pred[..., 3:5], rot_mat)

            if ref_lines is not None:
                import copy
                ref_lines = copy.deepcopy(ref_lines)
                ref_lines['position'] = np.matmul(ref_lines['position'], rot_mat) + origin
                if 'orientation' in ref_lines:
                    ref_lines['orientation'] += angle

            learning_weight = float(self.cfg.get('learning_based_score_weight', 0.25))
            rule_weight = float(self.cfg.get('rule_based_score_weight', 1.0))
            feasible_margin = float(self.cfg.get('rule_based_feasible_margin', 25.0))
            min_keep = int(self.cfg.get('rule_based_min_keep',
                                        min(8, len(rule_based_scores)) if len(rule_based_scores) > 0 else 0))

            feasible_mask = np.ones_like(rule_based_scores, dtype=bool)
            if not neural_only and len(rule_based_scores) > 0:
                best_rule = float(np.max(rule_based_scores))
                feasible_mask = rule_based_scores >= (best_rule - feasible_margin)
                hard_fail_mask = getattr(evaluator, "last_hard_fail_mask", None)
                if hard_fail_mask is not None and (~hard_fail_mask).any():
                    feasible_mask &= ~hard_fail_mask
                if feasible_mask.sum() < max(1, min_keep):
                    eligible = (
                        ~hard_fail_mask
                        if hard_fail_mask is not None and (~hard_fail_mask).any()
                        else np.ones_like(feasible_mask)
                    )
                    ranking_pool = np.where(
                        eligible,
                        rule_based_scores,
                        -np.inf,
                    )
                    keep_count = min(max(1, min_keep), int(eligible.sum()))
                    top_rule_idx = np.argsort(-ranking_pool)[:keep_count]
                    feasible_mask = np.zeros_like(rule_based_scores, dtype=bool)
                    feasible_mask[top_rule_idx] = True

            final_scores = np.full_like(sorted_probability, -1e9, dtype=np.float64)
            rule_norm_scores = np.zeros_like(sorted_probability, dtype=np.float64)
            prob_norm_scores = np.zeros_like(sorted_probability, dtype=np.float64)

            if neural_only:
                if not np.isfinite(sorted_probability).all():
                    raise ValueError('Non-finite neural trajectory probabilities')
                final_scores = sorted_probability.astype(np.float64)
                prob_norm_scores = final_scores.copy()
                best_idx = int(np.argmax(final_scores))
            elif feasible_mask.any():
                feasible_rule = rule_based_scores[feasible_mask].astype(np.float64)
                feasible_prob = sorted_probability[feasible_mask].astype(np.float64)

                rule_min = float(np.min(feasible_rule))
                rule_max = float(np.max(feasible_rule))
                if rule_max - rule_min > 1e-6:
                    rule_norm = (feasible_rule - rule_min) / (rule_max - rule_min)
                else:
                    rule_norm = np.ones_like(feasible_rule, dtype=np.float64)

                prob_sum = float(np.sum(feasible_prob))
                if prob_sum > 1e-8:
                    prob_norm = feasible_prob / prob_sum
                else:
                    prob_norm = np.full_like(feasible_prob, 1.0 / max(len(feasible_prob), 1), dtype=np.float64)

                rule_norm_scores[feasible_mask] = rule_norm
                prob_norm_scores[feasible_mask] = prob_norm
                final_scores[feasible_mask] = rule_weight * rule_norm + learning_weight * prob_norm
                best_idx = int(np.argmax(final_scores))
            else:
                final_scores = sorted_probability.astype(np.float64)
                prob_norm_scores = sorted_probability.astype(np.float64)
                best_idx = int(np.argmax(final_scores))

            # If every candidate collides in the conservative circle model,
            # choose the trajectory with the largest predicted clearance. In
            # dense moving traffic this is safer than either accelerating into
            # a side conflict or stopping abruptly in front of a rear vehicle.
            hard_fail_mask = getattr(evaluator, "last_hard_fail_mask", None)
            collision_mask = getattr(evaluator, "last_collision_mask", None)
            collision_clearance = getattr(
                evaluator, "last_collision_clearance", None
            )
            if (
                collision_mask is not None
                and len(collision_mask) == len(sorted_candidate_trajectories)
                and collision_mask.all()
                and collision_clearance is not None
            ):
                best_idx = int(np.argmax(collision_clearance))
            elif (
                hard_fail_mask is not None
                and len(hard_fail_mask) == len(sorted_candidate_trajectories)
                and hard_fail_mask.all()
            ):
                emergency_candidates = (
                    sorted_control_candidates
                    if sorted_control_candidates is not None
                    else sorted_candidate_trajectories
                )
                path_lengths = np.linalg.norm(
                    np.diff(emergency_candidates[..., :2], axis=1), axis=-1
                ).sum(axis=-1)
                best_idx = int(np.argmin(path_lengths))

            selected_control_traj = None
            if neural_only:
                assert best_idx == 0, 'Neural-only selection must preserve network top-1'
                print('[Pluto neural-only] network_rank=0 rules_called=0', flush=True)
            if sorted_control_candidates is not None:
                selected_control_traj = sorted_control_candidates[best_idx]
                if selected_control_traj.shape[-1] == 3:
                    heading = selected_control_traj[..., 2]
                    vx = np.gradient(selected_control_traj[..., 0], 0.1)
                    vy = np.gradient(selected_control_traj[..., 1], 0.1)
                    selected_control_traj = np.concatenate(
                        [
                            selected_control_traj[..., :2],
                            np.cos(heading)[..., None],
                            np.sin(heading)[..., None],
                            vx[..., None],
                            vy[..., None],
                        ],
                        axis=-1,
                    )
                elif selected_control_traj.shape[-1] >= 4 and selected_control_traj.shape[-1] < 6:
                    vel_xy = np.gradient(selected_control_traj[..., :2], 0.1, axis=0)
                    selected_control_traj = np.concatenate(
                        [selected_control_traj[..., :4], vel_xy[..., :2]], axis=-1
                    )

            if selected_control_traj is None:
                # Final fallback: synthesize richer control reference from planner trajectory.
                planner_selected = sorted_candidate_trajectories[best_idx]
                planner_xy = planner_selected
                if planner_xy.shape[-1] >= 3:
                    heading = planner_xy[..., 2]
                else:
                    diff = np.gradient(planner_xy[..., :2], axis=0)
                    heading = np.arctan2(diff[..., 1], diff[..., 0])
                vel_xy = np.gradient(planner_xy[..., :2], 0.1, axis=0)
                selected_control_traj = np.concatenate(
                    [
                        planner_xy[..., :2],
                        np.cos(heading)[..., None],
                        np.sin(heading)[..., None],
                        vel_xy[..., :2],
                    ],
                    axis=-1,
                )

            T_out = selected_control_traj.shape[0]
            C_out = min(selected_control_traj.shape[-1], 6)
            pred_traj = np.zeros((16, min(T_out, 80), 6), dtype=np.float32)
            pred_len = min(80, T_out)
            pred_traj[0, :pred_len, :C_out] = selected_control_traj[:pred_len, :C_out]

            info = {}
            if 'agent_tokens' in pluto_feature:
                info['agent_tokens'] = pluto_feature['agent_tokens']
            if controlled_agent_id is not None:
                info['controlled_agent_id'] = str(controlled_agent_id)
                if self.cfg["rl_finetuning_mode"]:
                    cbv_rollout_sample = self._build_cbv_rollout_sample(
                        current_state=current_state,
                        pluto_feature=pluto_feature,
                        out=out,
                        pred_local=pred_local,
                        timestep=timestep,
                        controlled_agent_id=controlled_agent_id,
                    )
                    if cbv_rollout_sample is not None:
                        info['cbv_rollout_samples'] = [cbv_rollout_sample]
            info['prediction'] = pred
            info['best_idx'] = best_idx
            info['trajectory_selection_mode'] = 'neural_only' if neural_only else 'hybrid'
            info['selected_network_rank'] = best_idx

            t6 = time.time()
            # print(f"[Pluto Profiling] step: {timestep}, process_scenario: {t2-t1:.3f}s, collate&to_tensor: {t3-t2:.3f}s, to_device: {t4-t3:.3f}s, model_forward: {t5-t4:.3f}s, postprocess: {t6-t5:.3f}s")
        return pred_traj, ref_lines, sorted_candidate_trajectories[:, :80], info
