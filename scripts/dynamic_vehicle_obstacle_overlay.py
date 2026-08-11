#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Overlay other vehicles as dynamic obstacles on a traversability cloud.

Run this node inside each vehicle namespace.  For example, the vehicle0
instance reads /vehicle0/local_traversability_cloud_raw and publishes
/vehicle0/local_traversability_cloud.  Points covered by the oriented,
inflated footprint of vehicle1/vehicle2 are assigned intensity 1.0.
"""

import math
import struct
import threading

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField


class DynamicVehicleObstacleOverlay:
    def __init__(self):
        namespace = rospy.get_namespace().strip("/")
        inferred_self = namespace.split("/")[-1] if namespace else ""

        self.self_vehicle = rospy.get_param("~self_vehicle", inferred_self)
        self.vehicle_names = rospy.get_param(
            "~vehicle_names", ["vehicle0", "vehicle1", "vehicle2"]
        )
        self.vehicle_names = [str(name).strip("/") for name in self.vehicle_names]

        self.vehicle_lengths = self._expand_numeric_param(
            rospy.get_param("~vehicle_lengths", [1.0, 1.0, 1.0]),
            len(self.vehicle_names),
            1.0,
        )
        self.vehicle_widths = self._expand_numeric_param(
            rospy.get_param("~vehicle_widths", [0.7, 0.7, 0.7]),
            len(self.vehicle_names),
            0.7,
        )

        # The requested 0.8 m is applied independently to front, rear,
        # left and right: total length/width therefore each grow by 1.6 m.
        self.inflation = max(0.0, float(rospy.get_param("~inflation", 0.8)))
        self.dynamic_intensity = float(
            rospy.get_param("~dynamic_obstacle_intensity", 1.0)
        )
        self.pose_timeout = max(0.0, float(rospy.get_param("~pose_timeout", 0.5)))
        self.strict_frame_check = bool(
            rospy.get_param("~strict_frame_check", True)
        )

        input_topic = rospy.get_param(
            "~input_topic", "local_traversability_cloud_raw"
        )
        output_topic = rospy.get_param(
            "~output_topic", "local_traversability_cloud"
        )
        odom_suffix = str(rospy.get_param("~odom_topic", "state_estimation")).strip("/")

        self._lock = threading.Lock()
        self._poses = {}
        self._dimensions = {}
        self._odom_subscribers = []

        for index, vehicle_name in enumerate(self.vehicle_names):
            self._dimensions[vehicle_name] = (
                self.vehicle_lengths[index],
                self.vehicle_widths[index],
            )
            topic = "/{}/{}".format(vehicle_name, odom_suffix)
            subscriber = rospy.Subscriber(
                topic,
                Odometry,
                self._odom_callback,
                callback_args=vehicle_name,
                queue_size=1,
                tcp_nodelay=True,
            )
            self._odom_subscribers.append(subscriber)

        self._publisher = rospy.Publisher(output_topic, PointCloud2, queue_size=1)
        self._cloud_subscriber = rospy.Subscriber(
            input_topic,
            PointCloud2,
            self._cloud_callback,
            queue_size=1,
            buff_size=8 * 1024 * 1024,
            tcp_nodelay=True,
        )

        if not self.self_vehicle:
            rospy.logfatal(
                "Cannot infer the current vehicle. Launch in /vehicleX or set ~self_vehicle."
            )
            rospy.signal_shutdown("self vehicle is unknown")
            return

        rospy.loginfo(
            "Dynamic vehicle overlay: self=%s, input=%s, output=%s, "
            "inflation=%.2f m on every side",
            self.self_vehicle,
            rospy.resolve_name(input_topic),
            rospy.resolve_name(output_topic),
            self.inflation,
        )

    @staticmethod
    def _expand_numeric_param(value, count, default):
        if isinstance(value, (int, float)):
            values = [float(value)] * count
        else:
            try:
                values = [float(item) for item in value]
            except (TypeError, ValueError):
                values = []

        if not values:
            values = [float(default)]
        if len(values) < count:
            values.extend([values[-1]] * (count - len(values)))
        return values[:count]

    @staticmethod
    def _normalize_frame(frame_id):
        return str(frame_id).strip("/")

    @staticmethod
    def _yaw_from_quaternion(quaternion):
        sin_yaw = 2.0 * (
            quaternion.w * quaternion.z + quaternion.x * quaternion.y
        )
        cos_yaw = 1.0 - 2.0 * (
            quaternion.y * quaternion.y + quaternion.z * quaternion.z
        )
        return math.atan2(sin_yaw, cos_yaw)

    def _odom_callback(self, message, vehicle_name):
        position = message.pose.pose.position
        yaw = self._yaw_from_quaternion(message.pose.pose.orientation)
        stamp = message.header.stamp
        if stamp == rospy.Time(0):
            stamp = rospy.Time.now()

        pose = (
            float(position.x),
            float(position.y),
            yaw,
            stamp,
            rospy.Time.now(),
            self._normalize_frame(message.header.frame_id),
        )
        with self._lock:
            self._poses[vehicle_name] = pose

    def _active_other_vehicle_rectangles(self, cloud_frame):
        now = rospy.Time.now()
        rectangles = []
        with self._lock:
            pose_snapshot = dict(self._poses)

        for vehicle_name in self.vehicle_names:
            if vehicle_name == self.self_vehicle:
                continue

            pose = pose_snapshot.get(vehicle_name)
            if pose is None:
                continue

            x, y, yaw, _pose_stamp, receipt_stamp, pose_frame = pose
            if self.pose_timeout > 0.0:
                age = max(0.0, (now - receipt_stamp).to_sec())
                if age > self.pose_timeout:
                    rospy.logwarn_throttle(
                        2.0,
                        "Ignore stale pose of {}: age={:.3f} s".format(
                            vehicle_name, age
                        ),
                    )
                    continue

            if (
                self.strict_frame_check
                and cloud_frame
                and pose_frame
                and cloud_frame != pose_frame
            ):
                rospy.logwarn_throttle(
                    2.0,
                    "Skip {}: cloud frame '{}' differs from pose frame '{}'".format(
                        vehicle_name, cloud_frame, pose_frame
                    ),
                )
                continue

            length, width = self._dimensions[vehicle_name]
            rectangles.append(
                (
                    vehicle_name,
                    x,
                    y,
                    math.cos(yaw),
                    math.sin(yaw),
                    0.5 * length + self.inflation,
                    0.5 * width + self.inflation,
                )
            )
        return rectangles

    @staticmethod
    def _point_is_inside_rectangle(point_x, point_y, rectangle):
        _name, center_x, center_y, cos_yaw, sin_yaw, half_length, half_width = rectangle
        dx = point_x - center_x
        dy = point_y - center_y

        # World/map frame -> vehicle heading frame.
        longitudinal = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        return abs(longitudinal) <= half_length and abs(lateral) <= half_width

    def _cloud_callback(self, message):
        cloud_frame = self._normalize_frame(message.header.frame_id)
        rectangles = self._active_other_vehicle_rectangles(cloud_frame)
        if not rectangles:
            self._publisher.publish(message)
            return

        fields = {field.name: field for field in message.fields}
        required = ("x", "y", "intensity")
        if any(name not in fields for name in required):
            rospy.logerr_throttle(
                2.0, "Input cloud must contain x, y and intensity fields."
            )
            self._publisher.publish(message)
            return

        if any(fields[name].datatype != PointField.FLOAT32 for name in required):
            rospy.logerr_throttle(
                2.0, "x, y and intensity fields must all be FLOAT32."
            )
            self._publisher.publish(message)
            return

        endian = ">" if message.is_bigendian else "<"
        float_format = endian + "f"
        x_offset = fields["x"].offset
        y_offset = fields["y"].offset
        intensity_offset = fields["intensity"].offset
        data = bytearray(message.data)

        changed_count = 0
        for row in range(message.height):
            row_base = row * message.row_step
            for column in range(message.width):
                point_base = row_base + column * message.point_step
                try:
                    point_x = struct.unpack_from(
                        float_format, data, point_base + x_offset
                    )[0]
                    point_y = struct.unpack_from(
                        float_format, data, point_base + y_offset
                    )[0]
                except struct.error:
                    rospy.logerr_throttle(2.0, "Malformed PointCloud2 data buffer.")
                    self._publisher.publish(message)
                    return

                if not (math.isfinite(point_x) and math.isfinite(point_y)):
                    continue

                if any(
                    self._point_is_inside_rectangle(point_x, point_y, rectangle)
                    for rectangle in rectangles
                ):
                    struct.pack_into(
                        float_format,
                        data,
                        point_base + intensity_offset,
                        self.dynamic_intensity,
                    )
                    changed_count += 1

        output = PointCloud2()
        output.header = message.header
        output.height = message.height
        output.width = message.width
        output.fields = message.fields
        output.is_bigendian = message.is_bigendian
        output.point_step = message.point_step
        output.row_step = message.row_step
        output.data = bytes(data)
        output.is_dense = message.is_dense
        self._publisher.publish(output)

        rospy.loginfo_throttle(
            1.0,
            "Dynamic vehicle overlay: active_other_vehicles={}, obstacle_points={}".format(
                len(rectangles), changed_count
            ),
        )


def main():
    rospy.init_node("dynamic_vehicle_obstacle_overlay")
    DynamicVehicleObstacleOverlay()
    rospy.spin()


if __name__ == "__main__":
    main()
