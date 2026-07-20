#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import copy
from collections import deque
from itertools import chain

from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
import sensor_msgs.point_cloud2 as pc2
import message_filters

# ------------------------- Helpers: fields -------------------------


def fields_signature(fields):
    return [(f.name, f.offset, f.datatype, f.count) for f in fields]


def field_names_from_msg(msg):
    return [f.name for f in msg.fields]


def pick_canonical_fields(msg, prefer_intensity=True):
    names = set(field_names_from_msg(msg))
    if prefer_intensity and all(k in names for k in ("x", "y", "z", "intensity")):
        return ["x", "y", "z", "intensity"]
    if all(k in names for k in ("x", "y", "z")):
        return ["x", "y", "z"]
    return [n for n in field_names_from_msg(msg)]


def select_fields_by_names(fields, keep_names):
    field_map = {f.name: f for f in fields}
    out = []
    for n in keep_names:
        if n in field_map:
            out.append(copy.deepcopy(field_map[n]))
    return out


def read_points_as_list(msg, field_names):
    return list(pc2.read_points(msg, field_names=field_names, skip_nans=True))


# ------------------------- Helpers: optional voxel downsample -------------------------


def voxel_downsample(points, leaf, use_intensity=False):
    if leaf is None or leaf <= 0.0:
        return points
    inv = 1.0 / leaf
    vox = {}
    if use_intensity:
        for x, y, z, i in points:
            k = (int(x * inv), int(y * inv), int(z * inv))
            if k not in vox:
                vox[k] = (x, y, z, i)
    else:
        for x, y, z in points:
            k = (int(x * inv), int(y * inv), int(z * inv))
            if k not in vox:
                vox[k] = (x, y, z)
    return list(vox.values())


# ------------------------- Terrain merge -------------------------


def merge_clouds(msg_a, msg_b, out_frame_id=None):
    if msg_a is None or msg_b is None:
        return None
    if not msg_a.data or not msg_b.data:
        return None

    frame_a = msg_a.header.frame_id
    frame_b = msg_b.header.frame_id
    if out_frame_id is None:
        out_frame_id = frame_a

    # 不做 TF：frame 不同直接不合
    if frame_a != frame_b:
        rospy.logwarn_throttle(
            2.0,
            f"[merge_clouds] frame_id mismatch: A={frame_a}, B={frame_b}. Skip merge (no TF).",
        )
        return None

    sig_a = fields_signature(msg_a.fields)
    sig_b = fields_signature(msg_b.fields)

    if sig_a == sig_b:
        field_names = field_names_from_msg(msg_a)
        out_fields = copy.deepcopy(msg_a.fields)
    else:
        names_a = field_names_from_msg(msg_a)
        names_b = set(field_names_from_msg(msg_b))
        field_names = [n for n in names_a if n in names_b]
        if not field_names:
            rospy.logerr_throttle(2.0, "[merge_clouds] No common fields. Cannot merge.")
            return None
        out_fields = select_fields_by_names(msg_a.fields, field_names)
        rospy.logwarn_throttle(
            2.0, f"[merge_clouds] Field mismatch. Use common fields: {field_names}"
        )

    pts_a = read_points_as_list(msg_a, field_names)
    pts_b = read_points_as_list(msg_b, field_names)

    header = copy.deepcopy(msg_a.header)
    header.frame_id = out_frame_id
    header.stamp = (
        msg_a.header.stamp
        if msg_a.header.stamp >= msg_b.header.stamp
        else msg_b.header.stamp
    )

    merged_pts = pts_a + pts_b
    return pc2.create_cloud(header, out_fields, merged_pts)


# ------------------------- Node -------------------------


class CloudMergerAndAccumulator:
    def __init__(self):
        self.started = False
        self.processing_inited = False
        self.start_sub = None

        # 这些等启动后再 init
        self.pub_terrain_merged = None
        self.pub_scan_accum = None
        self.sub_scan = None
        self.sub_terrain_a = None
        self.sub_terrain_b = None
        self.ats = None

        # 缓存/状态
        self.scan_buffer = None
        self.scan_count = 0
        self.base_frame_id = None
        self.base_field_names = None
        self.base_out_fields = None

        # 启动方式。默认直接运行；需要外部触发时再在 launch 中打开。
        self.wait_for_start = bool(rospy.get_param("~wait_for_start", False))
        self.start_topic = rospy.get_param("~start_topic", "/start_exploration")

        # 点云处理参数
        self.window_size = int(rospy.get_param("~window_size", 10))
        self.publish_every_n = int(rospy.get_param("~publish_every_n", 5))
        self.voxel_leaf = float(rospy.get_param("~voxel_leaf", 0.1))
        self.prefer_intensity = bool(rospy.get_param("~prefer_intensity", True))

        self.topic_terrain_map = rospy.get_param("~terrain_map", "terrain_map")
        self.topic_terrain_map_ext = rospy.get_param(
            "~terrain_map_ext", "terrain_map_ext"
        )
        self.topic_registered_scan = rospy.get_param(
            "~registered_scan", "registered_scan_filted"
        )

        self.topic_terrain_merged = rospy.get_param(
            "~terrain_merged_out", "terrain_map_merged"
        )
        self.topic_scan_accum = rospy.get_param(
            "~registered_scan_accum_out", "registered_scan_accum_10"
        )

        self.sync_slop = float(rospy.get_param("~sync_slop", 0.10))
        self.sync_queue = int(rospy.get_param("~sync_queue", 10))

        if self.wait_for_start:
            self.start_sub = rospy.Subscriber(
                self.start_topic, Bool, self.cb_start, queue_size=5
            )
            rospy.logwarn(
                f"[CloudMergerAndAccumulator] Waiting for '{self.start_topic}' "
                "(std_msgs/Bool, data=True)."
            )
        else:
            self.started = True
            rospy.loginfo(
                "[CloudMergerAndAccumulator] wait_for_start=false; start processing immediately."
            )
            self._init_processing()

    def cb_start(self, msg: Bool):
        if msg is None:
            return
        if not msg.data:
            return

        if self.started:
            return

        self.started = True
        rospy.logwarn(
            "[CloudMergerAndAccumulator] ✅ exploration_start=True received. Init & start processing now."
        )

        # 启动后才初始化主体
        self._init_processing()

        # 如果你不希望后续再触发，可注销 start 订阅
        if self.start_sub is not None:
            try:
                self.start_sub.unregister()
            except Exception:
                pass
            self.start_sub = None

    def _init_processing(self):
        if self.processing_inited:
            return
        self.processing_inited = True

        # Publishers
        self.pub_terrain_merged = rospy.Publisher(
            self.topic_terrain_merged, PointCloud2, queue_size=1
        )
        self.pub_scan_accum = rospy.Publisher(
            self.topic_scan_accum, PointCloud2, queue_size=1
        )

        # Terrain map merge (approx sync)
        self.sub_terrain_a = message_filters.Subscriber(
            self.topic_terrain_map, PointCloud2
        )
        self.sub_terrain_b = message_filters.Subscriber(
            self.topic_terrain_map_ext, PointCloud2
        )
        self.ats = message_filters.ApproximateTimeSynchronizer(
            [self.sub_terrain_a, self.sub_terrain_b],
            queue_size=self.sync_queue,
            slop=self.sync_slop,
            allow_headerless=False,
        )
        self.ats.registerCallback(self.cb_terrain_pair)

        # Registered scan accumulation
        self.scan_buffer = deque(maxlen=self.window_size)
        self.sub_scan = rospy.Subscriber(
            self.topic_registered_scan,
            PointCloud2,
            self.cb_registered_scan,
            queue_size=5,
        )

        rospy.loginfo(
            f"[CloudMergerAndAccumulator] window_size={self.window_size}, publish_every_n={self.publish_every_n}, voxel_leaf={self.voxel_leaf}"
        )
        rospy.loginfo(
            f"[Terrain Merge] {self.topic_terrain_map} + {self.topic_terrain_map_ext} -> {self.topic_terrain_merged}"
        )
        rospy.loginfo(
            f"[Scan Accum] {self.topic_registered_scan} (x{self.window_size}) -> {self.topic_scan_accum}"
        )

    def cb_terrain_pair(self, msg_map, msg_ext):
        merged = merge_clouds(msg_map, msg_ext, out_frame_id=msg_map.header.frame_id)
        if merged is not None:
            self.pub_terrain_merged.publish(merged)

    def cb_registered_scan(self, msg):
        if msg is None or not msg.data:
            return

        # 初始化 canonical fields
        if self.base_field_names is None:
            self.base_field_names = pick_canonical_fields(
                msg, prefer_intensity=self.prefer_intensity
            )
            self.base_out_fields = select_fields_by_names(
                msg.fields, self.base_field_names
            )
            self.base_frame_id = msg.header.frame_id
            rospy.loginfo(
                f"[Scan Accum] canonical fields={self.base_field_names}, base_frame_id={self.base_frame_id}"
            )

        # frame_id 变化：清 buffer 避免乱图
        if msg.header.frame_id != self.base_frame_id:
            rospy.logwarn_throttle(
                2.0,
                f"[Scan Accum] frame_id changed: {self.base_frame_id} -> {msg.header.frame_id}. Reset buffer.",
            )
            self.scan_buffer.clear()
            self.base_frame_id = msg.header.frame_id
            self.base_field_names = pick_canonical_fields(
                msg, prefer_intensity=self.prefer_intensity
            )
            self.base_out_fields = select_fields_by_names(
                msg.fields, self.base_field_names
            )

        # 必要字段缺失：重置
        curr_names = set(field_names_from_msg(msg))
        if any(n not in curr_names for n in self.base_field_names):
            rospy.logwarn_throttle(
                2.0, "[Scan Accum] missing fields. Reset canonical fields & buffer."
            )
            self.scan_buffer.clear()
            self.base_field_names = pick_canonical_fields(
                msg, prefer_intensity=self.prefer_intensity
            )
            self.base_out_fields = select_fields_by_names(
                msg.fields, self.base_field_names
            )

        pts = read_points_as_list(msg, self.base_field_names)
        if not pts:
            return

        use_intensity = (
            len(self.base_field_names) == 4 and self.base_field_names[-1] == "intensity"
        )
        if self.voxel_leaf > 0:
            pts = voxel_downsample(pts, self.voxel_leaf, use_intensity=use_intensity)

        self.scan_buffer.append(pts)
        self.scan_count += 1

        # 每 N 帧发布一次
        if self.publish_every_n > 1 and (self.scan_count % self.publish_every_n) != 0:
            return

        all_pts = list(chain.from_iterable(self.scan_buffer))
        if not all_pts:
            return

        header = copy.deepcopy(msg.header)
        header.frame_id = self.base_frame_id
        header.stamp = msg.header.stamp

        out_msg = pc2.create_cloud(header, self.base_out_fields, all_pts)
        self.pub_scan_accum.publish(out_msg)


def main():
    rospy.init_node("cloud_merger_and_accumulator", anonymous=False)
    CloudMergerAndAccumulator()
    rospy.spin()


if __name__ == "__main__":
    main()
