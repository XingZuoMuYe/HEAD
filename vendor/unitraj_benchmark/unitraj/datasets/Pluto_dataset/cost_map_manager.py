import numpy as np
import cv2
from shapely.geometry import Polygon
from typing import Optional, List, Tuple, Dict, Any
from scipy import ndimage

from unitraj.datasets.Pluto_dataset.pluto_utils import _is_lane_like, _is_crosswalk


class CostMapManager:
    def __init__(self, origin, angle, height=600, width=600, resolution=0.2):
        self.height = height
        self.width = width
        self.resolution = resolution
        self.resolution_hw = np.array([resolution, -resolution], dtype=np.float32)
        # origin must be 2D (x, y). Some SD/track sources may provide (x, y, z).
        origin = np.asarray(origin, dtype=np.float64).reshape(-1)
        self.origin = origin[:2].copy() if origin.shape[0] >= 2 else np.array([0.0, 0.0], dtype=np.float64)
        self.angle = angle
        self.offset = np.array([height / 2, width / 2], dtype=np.float32)
        self.rot_mat = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float64,
        )

    def _clip_pts(self, pts: np.ndarray) -> np.ndarray:
        """Clip pixel coordinates to image bounds to reduce OpenCV errors."""
        if pts.size == 0:
            return pts
        out = pts.copy()
        out[:, 0] = np.clip(out[:, 0], 0, self.width - 1)
        out[:, 1] = np.clip(out[:, 1], 0, self.height - 1)
        return out

    @staticmethod
    def _close_ring(poly: np.ndarray) -> np.ndarray:
        if poly.shape[0] < 3:
            return poly
        if not np.allclose(poly[0, :2], poly[-1, :2]):
            poly = np.concatenate([poly, poly[0:1]], axis=0)
        return poly

    def _rasterize_polygon(self, mask: np.ndarray, polygon_xy: np.ndarray, value: int = 1) -> None:
        if polygon_xy is None:
            return
        poly = np.asarray(polygon_xy, dtype=np.float64)
        if poly.ndim != 2 or poly.shape[0] < 3:
            return
        poly = self._close_ring(poly)
        pts = self.global_to_pixel(poly[:, :2])
        pts = np.round(pts).astype(np.int32)
        pts = self._clip_pts(pts.astype(np.float64)).astype(np.int32)
        try:
            cv2.fillPoly(mask, [pts], value)
        except Exception:
            # ignore rare self-intersection / invalid polygons
            pass

    def _rasterize_polyline(self, mask: np.ndarray, polyline_xy: np.ndarray, value: int = 1, thickness: int = 1) -> None:
        if polyline_xy is None:
            return
        line = np.asarray(polyline_xy, dtype=np.float64)
        if line.ndim != 2 or line.shape[0] < 2:
            return
        pts = self.global_to_pixel(line[:, :2])
        pts = np.round(pts).astype(np.int32)
        pts = self._clip_pts(pts.astype(np.float64)).astype(np.int32)
        try:
            cv2.polylines(
                mask,
                [pts.reshape(-1, 1, 2)],
                isClosed=False,
                color=value,
                thickness=int(thickness),
            )
        except Exception:
            pass

    def build_cost_maps(
        self,
        static_objects,
        agents,
        map_features_list,
        agents_polygon: Optional[List[Polygon]] = None,
        traffic_light_status: Optional[Dict[str, Any]] = None,
        present_idx: Optional[int] = None,
        future_steps: int = 20,
        dynamic_obstacle_types: Optional[Tuple[int, ...]] = None,
        dynamic_dilation_radius_m: float = 1.0,
        raster_lane_boundary: bool = False,
        lane_boundary_thickness_px: int = 1,
    ):
        """Build a signed distance field (SDF) over local raster.

        Conventions:
          - drivable_area_mask==1 means drivable/free space
          - drivable_area_mask==0 means non-drivable/obstacle
          - SDF > 0 inside drivable area, < 0 outside
        """
        drivable_area_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        dynamic_obstacle_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        crosswalk_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        red_light_mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # 1) static raster layers from SD map_features
        # - drivable_area_mask: LANE / LANE_CONNECTOR polygons
        # - crosswalk_mask: CROSSWALK polygons
        for _map_id, map_feat in (map_features_list or {}).items():
            mtype = str(map_feat.get('type', '')).upper()

            if _is_lane_like(mtype):
                polygon = map_feat.get('polygon', None)
                if polygon is not None and len(polygon) >= 3:
                    self._rasterize_polygon(drivable_area_mask, np.asarray(polygon)[:, :2], value=1)
                else:
                    # fallback: buffer polyline by drawing thick line
                    polyline = map_feat.get('polyline', None)
                    if polyline is not None and len(polyline) >= 2:
                        thickness = int(max(1, round(3.6 / max(self.resolution, 1e-3))))
                        self._rasterize_polyline(
                            drivable_area_mask,
                            np.asarray(polyline)[:, :2],
                            value=1,
                            thickness=thickness,
                        )

                if raster_lane_boundary:
                    poly = map_feat.get('polyline', None)
                    if poly is not None and len(poly) >= 2:
                        self._rasterize_polyline(
                            drivable_area_mask,
                            np.asarray(poly)[:, :2],
                            value=1,
                            thickness=int(lane_boundary_thickness_px),
                        )

            elif _is_crosswalk(mtype):
                polygon = map_feat.get('polygon', None)
                if polygon is not None and len(polygon) >= 3:
                    self._rasterize_polygon(crosswalk_mask, np.asarray(polygon)[:, :2], value=1)

        # 2) burn static objects into mask as obstacles
        if static_objects is not None and len(static_objects) > 0:
            for obj in np.asarray(static_objects):
                # expected obj layout in this dataset: [x,y,heading,width,length,cat]
                if obj.shape[0] < 5:
                    continue
                pos = obj[:2]
                heading = float(obj[2])
                w, l = float(obj[3]), float(obj[4])
                corners = self.get_box_corners(pos, heading, w, l)
                pts = self.global_to_pixel(corners)
                pts = np.round(pts).astype(np.int32)
                pts = self._clip_pts(pts.astype(np.float64)).astype(np.int32)
                try:
                    cv2.fillConvexPoly(drivable_area_mask, pts, 0)
                except Exception:
                    pass

        # 3) optional: parked agents as static obstacles (RIFT logic)
        if agents is not None and agents_polygon is not None:
            try:
                position = np.asarray(agents["position"], dtype=np.float64)
                valid_mask = np.asarray(agents["valid_mask"], dtype=bool)
                for pos_seq, mask_seq, poly in zip(position, valid_mask, agents_polygon):
                    if poly is None:
                        continue
                    if mask_seq.sum() < 50:
                        continue
                    pts = pos_seq[mask_seq]
                    if pts.shape[0] < 2:
                        continue
                    displacement = float(np.linalg.norm(pts[-1, :2] - pts[0, :2]))
                    if displacement < 1.0:
                        # rasterize polygon directly in world coords
                        coords = np.asarray(poly.exterior.coords, dtype=np.float64)
                        self._rasterize_polygon(drivable_area_mask, coords[:, :2], value=0)
            except Exception:
                pass

        # 4) red light obstacles (SD dynamic_map_states)
        # NOTE: The traffic light's lane_id/lane_connector_id lets us find the *controlled lane connector*,
        # but its polygon is typically the whole connector area and should NOT be fully blocked.
        # Instead, we rasterize a local stop region around stop_point.
        # Optionally, if the lane polyline is present, we also rasterize a short stopline segment
        # (a thin rectangle orthogonal to lane direction) near stop_point.
        if traffic_light_status is not None:
            tl_t = int(present_idx) if present_idx is not None else None

            def _is_red(val) -> bool:
                if val is None:
                    return False
                if isinstance(val, (bytes, str)):
                    s = str(val).upper()
                    # accept common encodings / names
                    # - strings like 'TRAFFIC_LIGHT_RED'/'RED'
                    # - numeric-like strings '3'
                    return ("RED" in s) or (s.strip() == "3")
                if isinstance(val, (np.integer, int)):
                    # numeric encoding used in UniTraj VBD: 0 unknown, 1 green, 2 yellow, 3 red
                    return int(val) == 3
                if isinstance(val, dict):
                    return _is_red(val.get('status', val.get('state')))
                return False

            for _lane_id, info in traffic_light_status.items():
                obj_state = None
                if 'state' in info and isinstance(info['state'], dict):
                    obj_state = info['state'].get('object_state', None)

                st = obj_state
                if tl_t is not None and isinstance(obj_state, (list, tuple, np.ndarray)):
                    if len(obj_state) > tl_t:
                        st = obj_state[tl_t]
                if not _is_red(st):
                    continue

                stop_pt = info.get('stop_point', None)
                if stop_pt is None:
                    continue

                stop_pt = np.asarray(stop_pt, dtype=np.float64).reshape(-1)
                if stop_pt.shape[0] < 2:
                    continue

                # Burn a small disk obstacle at stop_point (also record into red_light_mask).
                pt_px = self.global_to_pixel(stop_pt[:2][None, :])[0]
                cx, cy = int(round(pt_px[0])), int(round(pt_px[1]))
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    r_px = int(max(1, round(1.5 / max(self.resolution, 1e-3))))
                    try:
                        cv2.circle(red_light_mask, (cx, cy), r_px, 1, thickness=-1)
                        cv2.circle(drivable_area_mask, (cx, cy), r_px, 0, thickness=-1)
                    except Exception:
                        pass

                # Optional: derive a short stopline rectangle from lane polyline near stop_point.
                lane_key = str(_lane_id)
                lane_feat = None
                if isinstance(map_features_list, dict):
                    lane_feat = map_features_list.get(lane_key, None)

                polyline = lane_feat.get('polyline', None)
                polyline = np.asarray(polyline)[:, :2] if polyline is not None else None
                if polyline is not None and polyline.ndim == 2 and polyline.shape[0] >= 2:
                    # pick nearest segment to stop point
                    p = stop_pt[:2].astype(np.float64)
                    d = polyline[1:] - polyline[:-1]
                    seg_mid = 0.5 * (polyline[1:] + polyline[:-1])
                    j = int(np.argmin(np.linalg.norm(seg_mid - p[None, :], axis=1)))
                    direction = d[j]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction = direction / norm
                        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
                        # stopline size (meters)
                        half_len = 4.0  # across lane
                        half_w = 0.6     # along lane
                        c = p
                        rect = np.array([
                            c + normal * half_len + direction * half_w,
                            c - normal * half_len + direction * half_w,
                            c - normal * half_len - direction * half_w,
                            c + normal * half_len - direction * half_w,
                        ])
                        self._rasterize_polygon(red_light_mask, rect, value=1)
                        self._rasterize_polygon(drivable_area_mask, rect, value=0)

        # red_light_mask has already been burned into drivable_area_mask above.

        # 5) dynamic obstacles mask from future agent states (Pluto-like)
        # Mark dynamic agents' footprints for a short horizon (default 2s @ 10Hz => 20).
        if agents is not None and present_idx is not None:
            try:
                pos = np.asarray(agents.get('position'), dtype=np.float64)  # (N,T,2)
                hdg = np.asarray(agents.get('heading'), dtype=np.float64)  # (N,T)
                shp = np.asarray(agents.get('shape'), dtype=np.float64)  # (N,T,2)
                cat = np.asarray(agents.get('category'), dtype=np.int64)  # (N,)
                vld = np.asarray(agents.get('valid_mask'), dtype=bool)  # (N,T)

                if dynamic_obstacle_types is None:
                    # dataset category indices: [EGO, VEHICLE, PEDESTRIAN, BICYCLE]
                    dynamic_obstacle_types = (1, 2, 3)

                t0 = int(present_idx)
                t1 = min(pos.shape[1], t0 + int(future_steps) + 1)
                r_px = int(max(1, round(dynamic_dilation_radius_m / max(self.resolution, 1e-3))))

                for i in range(pos.shape[0]):
                    if int(cat[i]) not in dynamic_obstacle_types:
                        continue
                    for t in range(t0, t1):
                        if not bool(vld[i, t]):
                            continue
                        center = pos[i, t, :2]
                        heading = float(hdg[i, t])
                        # shape is constant over time in this dataset implementation
                        width = float(shp[i, t, 0]) if shp.ndim == 3 else float(shp[i, 0])
                        length = float(shp[i, t, 1]) if shp.ndim == 3 else float(shp[i, 1])
                        if not np.isfinite(center).all():
                            continue
                        corners = self.get_box_corners(center, heading, width, length)
                        pts = self.global_to_pixel(corners)
                        pts = np.round(pts).astype(np.int32)
                        pts = self._clip_pts(pts.astype(np.float64)).astype(np.int32)
                        try:
                            cv2.fillConvexPoly(dynamic_obstacle_mask, pts, 1)
                        except Exception:
                            pass

                if dynamic_obstacle_mask.any() and r_px > 0:
                    try:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_px + 1, 2 * r_px + 1))
                        dynamic_obstacle_mask = cv2.dilate(dynamic_obstacle_mask, kernel, iterations=1)
                    except Exception:
                        pass
            except Exception:
                pass

        # 6) compute signed distance transform for drivable area
        distance = ndimage.distance_transform_edt(drivable_area_mask)
        inv_distance = ndimage.distance_transform_edt(1 - drivable_area_mask)
        drivable_area_sdf = (distance - inv_distance) * self.resolution

        # 7) dynamic obstacle SDF (positive far away, negative inside obstacle)
        if dynamic_obstacle_mask.any():
            d_free = ndimage.distance_transform_edt(1 - dynamic_obstacle_mask)
            d_occ = ndimage.distance_transform_edt(dynamic_obstacle_mask)
            dynamic_sdf = (d_free - d_occ) * self.resolution
        else:
            dynamic_sdf = np.ones_like(drivable_area_sdf, dtype=np.float64) * 1e3

        # Combine (conservative): near dynamic obstacles, lower the SDF.
        combined_sdf = np.minimum(drivable_area_sdf, dynamic_sdf)

        return {
            "cost_maps": combined_sdf[:, :, None].astype(np.float16),
            "drivable_mask": drivable_area_mask.astype(np.uint8),
            "dynamic_obstacle_mask": dynamic_obstacle_mask.astype(np.uint8),
            "crosswalk_mask": crosswalk_mask.astype(np.uint8),
            "red_light_mask": red_light_mask.astype(np.uint8),
        }

    def get_box_corners(self, center, heading, width, length):
        dx = length / 2
        dy = width / 2
        corners = np.array([
            [dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]
        ])
        rot = np.array([
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)]
        ])
        return center + corners @ rot.T

    def global_to_pixel(self, coord):
        coord = np.asarray(coord, dtype=np.float64)
        if coord.ndim == 1:
            coord = coord[None, :]
        coord_xy = coord[:, :2]
        # origin_xy = self.origin[:2]
        coord = np.matmul(coord_xy - self.origin, self.rot_mat)
        coord = coord / self.resolution_hw + self.offset
        return coord
