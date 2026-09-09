"""Ego-only MetaDrive adapter for the user-provided evaluation v2 protocol.

TTC/comfort definitions are shared; collision/off-road/completion remain native
MetaDrive events, not PufferDrive's native score or official nuPlan metrics.
"""
import numpy as np

from evaluation.closed_loop_metrics import (
    STATE_FIELDS, compute_closed_loop_metric_rows, metric_metadata,
)
from evaluation.compute_puffer_nuplan_style_scores import FORMULA_VERSION, score_scenario


class EvaluationV2:
    def __init__(self):
        self.frames = []
        self.identities = {}
        self.flags = dict(collision=False, out_of_road=False, arrive_dest=False)
        self.warmup_frames = 0
        self.dt = None
        self.scenario_id = None
        self.warmup_flags = dict(self.flags)
        self.route_progress = None

    def step(self, info, env):
        policy = env.engine.get_policy(env.vehicle.id)
        manager = getattr(env.engine, "data_manager", None)
        if manager is not None:
            self.scenario_id = str(manager.current_scenario["id"])
        warmup = policy.action_info.get("closed_loop_stage") == "warmup"
        flags = self.warmup_flags if warmup else self.flags
        for flag in flags:
            if flag == "collision":
                value = any(info.get(k, False) for k in (
                    "crash_vehicle", "crash_object", "crash_human", "crash_building", "crash_sidewalk"
                ))
            else:
                value = info.get(flag, False)
            flags[flag] |= bool(value)
        if warmup:
            self.warmup_frames += 1
            return
        self.dt = float(env.config["physics_world_step_size"] * env.config["decision_repeat"])
        if info.get("route_completion") is not None:
            self.route_progress = float(info["route_completion"])
        from metadrive.component.vehicle.base_vehicle import BaseVehicle
        from metadrive.component.traffic_participants.base_traffic_participant import BaseTrafficParticipant
        from metadrive.component.static_object.traffic_object import TrafficObject

        frame = {}
        for obj in env.engine.get_objects().values():
            if not isinstance(obj, (BaseVehicle, BaseTrafficParticipant, TrafficObject)):
                continue
            # Stable slots: absent objects are padded invalid at finalize time.
            key = "ego" if obj is env.vehicle else str(obj.id)
            if key not in self.identities:
                self.identities[key] = len(self.identities)
            x, y = np.asarray(obj.position)[:2]
            vx, vy = np.asarray(obj.velocity)[:2]
            frame[key] = dict(x=x, y=y, vx=vx, vy=vy, heading=obj.heading_theta,
                              length=obj.LENGTH, width=obj.WIDTH,
                              id=self.identities[key], valid=1, stopped=0,
                              respawn_count=0, type=1 if isinstance(obj, BaseVehicle) else 2)
        if "ego" not in frame:
            raise RuntimeError("evaluation v2 did not find the controlled ego")
        # stopped is the simulator lifecycle flag, not simply speed==0.
        self.frames.append(frame)

    def finish(self):
        metadata = metric_metadata(self.dt or 0.1)
        metadata.update(focal_scope="controlled_ego_only", warmup_excluded=True,
                        scoring_source="evaluation.compute_puffer_nuplan_style_scores.score_scenario",
                        formula_version=FORMULA_VERSION,
                        native_event_protocol="head_metadrive_v1",
                        stopped_mapping="no_lifecycle_stop_or_respawn_in_this_adapter")
        if not self.frames:
            return dict(available=False, reason="no post-warmup frames", metadata=metadata,
                        scenario_id=self.scenario_id, evaluated_frames=0,
                        warmup_frames=self.warmup_frames, warmup_events=self.warmup_flags,
                        validity=dict(events=False, ttc=False, comfort=False))
        keys = list(self.identities)
        frames = [
            {field: np.asarray([frame.get(key, {}).get(field, 0) for key in keys])
             for field in STATE_FIELDS} for frame in self.frames
        ]
        row = compute_closed_loop_metric_rows(
            frames, [0, len(keys)], self.dt, focal_mask=[key == "ego" for key in keys]
        )[0]
        row.update(self.flags)
        # Auxiliary continuous simulator progress, never substituted into the
        # evaluation package's binary completion_rate or composite score.
        row["aux_metadrive_route_progress"] = self.route_progress
        row["completion"] = float(self.flags["arrive_dest"])
        row.update(completion_rate=row["completion"], collision_rate=float(self.flags["collision"]),
                   offroad_rate=float(self.flags["out_of_road"]))
        valid = row["comfort_metric_frames"] > 0 and row["ttc_metric_frames"] > 0
        for name in ("strict", "frame"):
            row[f"head_nuplan_style_{name}_score"] = (
                score_scenario(row, ttc_comfort_mode=name)[FORMULA_VERSION]
                if valid else None
            )
        return dict(available=valid, scenario_id=self.scenario_id, metrics=row,
                    evaluated_frames=len(frames), warmup_frames=self.warmup_frames,
                    warmup_events=self.warmup_flags,
                    validity=dict(events=True, ttc=row["ttc_metric_frames"] > 0,
                                  comfort=row["comfort_metric_frames"] > 0),
                    metadata=metadata)
