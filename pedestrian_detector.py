#!/usr/bin/env python3
"""CPU-only YOLOv8s + ByteTrack station-room personnel control.

功能：
- 兼容本地 MP4 文件和 RTSP 地址，默认读取 001.mp4
- YOLOv8s 纯 CPU、固定 imgsz=640、默认显示阈值 conf=0.55
- ByteTrack 自定义配置，CLAHE 暗光增强，小框过滤，二次 NMS
- 基于人体框与站房 ROI 的重叠面积占比判定是否在场
- 连续帧防抖：入场、离场、多人违规均需达到连续帧阈值
- 字典轻量记录人员轨迹，输出入场、离场、超时、多人四类告警
- 每帧绘制检测框、ID、置信度、ROI、人数、停留时长，并保存 MP4
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
DEFAULT_VIDEO = "001.mp4"
DEFAULT_MODEL = "yolov8s.pt"
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.55
DEFAULT_TRACK_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_TRACKER_CONFIG = "bytetrack_station.yaml"

# The following constants are default CLI values. They are intentionally exposed as
# --min-box-area/--roi-overlap/--enter-frames/--exit-frames/--multi-frames/
# --timeout-minutes/--max-persons so station-specific tuning does not require code edits.
DEFAULT_MIN_BOX_AREA = 800
DEFAULT_ROI_OVERLAP = 0.3
DEFAULT_ENTER_FRAMES = 5
DEFAULT_EXIT_FRAMES = 30
DEFAULT_MULTI_FRAMES = 5
DEFAULT_TIMEOUT_MINUTES = 30.0
DEFAULT_MAX_PERSONS = 1
WINDOW_NAME = "Station Personnel Control - YOLOv8s CPU"


@dataclass
class Detection:
    """A filtered pedestrian detection in one frame."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float


@dataclass
class TrackedDetection(Detection):
    """A pedestrian detection with a ByteTrack ID and ROI state."""

    track_id: int
    in_roi: bool = False
    roi_overlap: float = 0.0


@dataclass
class PersonState:
    """Lightweight business state for one tracked person."""

    track_id: int
    entered: bool = False
    entry_time: float | None = None
    last_seen_time: float | None = None
    exit_time: float | None = None
    inside_frames: int = 0
    outside_frames: int = 0
    timeout_alerted: bool = False
    total_stay_seconds: float = 0.0


class StationBusinessController:
    """Entry/exit, dwell-time, and occupancy business logic with frame debounce."""

    def __init__(
        self,
        timeout_seconds: float,
        max_persons: int,
        enter_frames: int,
        exit_frames: int,
        multi_frames: int,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_persons = max_persons
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.multi_frames = multi_frames
        self.people: dict[int, PersonState] = {}
        self.multi_violation_frames = 0
        self.multi_alert_active = False

    def update(self, detections: list[TrackedDetection], timestamp: float, frame_index: int) -> list[str]:
        """Update person states and return console/overlay alert messages."""
        messages: list[str] = []
        current_inside_ids = {det.track_id for det in detections if det.in_roi and det.track_id > 0}
        current_track_ids = {det.track_id for det in detections if det.track_id > 0}

        for track_id in current_track_ids:
            state = self.people.setdefault(track_id, PersonState(track_id=track_id))
            state.last_seen_time = timestamp
            if track_id in current_inside_ids:
                state.inside_frames += 1
                state.outside_frames = 0
                if not state.entered and state.inside_frames >= self.enter_frames:
                    state.entered = True
                    state.entry_time = timestamp
                    state.exit_time = None
                    messages.append(format_event(frame_index, timestamp, f"入场 ID={track_id}"))
            else:
                state.outside_frames += 1
                state.inside_frames = 0

        for track_id, state in list(self.people.items()):
            if track_id not in current_track_ids and state.entered:
                state.outside_frames += 1
                state.inside_frames = 0

            if state.entered and state.entry_time is not None:
                state.total_stay_seconds = max(0.0, timestamp - state.entry_time)
                if state.total_stay_seconds >= self.timeout_seconds and not state.timeout_alerted:
                    state.timeout_alerted = True
                    messages.append(
                        format_event(
                            frame_index,
                            timestamp,
                            f"超时 ID={track_id} stay={format_duration(state.total_stay_seconds)}",
                        )
                    )

            if state.entered and state.outside_frames >= self.exit_frames:
                state.entered = False
                state.exit_time = timestamp
                messages.append(
                    format_event(
                        frame_index,
                        timestamp,
                        f"离场 ID={track_id} total={format_duration(state.total_stay_seconds)}",
                    )
                )

        inside_count = self.inside_count()
        if inside_count > self.max_persons:
            self.multi_violation_frames += 1
            if self.multi_violation_frames >= self.multi_frames and not self.multi_alert_active:
                self.multi_alert_active = True
                messages.append(
                    format_event(
                        frame_index,
                        timestamp,
                        f"多人违规 count={inside_count} limit={self.max_persons}",
                    )
                )
        else:
            self.multi_violation_frames = 0
            self.multi_alert_active = False

        return messages

    def inside_count(self) -> int:
        """Return debounced count of people currently considered inside the station."""
        return sum(1 for state in self.people.values() if state.entered)

    def active_states(self) -> list[PersonState]:
        """Return states that are currently considered inside."""
        return [state for state in self.people.values() if state.entered]


def format_event(frame_index: int, timestamp: float, message: str) -> str:
    return f"[Frame {frame_index:06d} | {format_duration(timestamp)}] {message}"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_roi(roi_text: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    """Parse ROI as x1,y1,x2,y2 in pixels or normalized 0-1 coordinates."""
    if not roi_text:
        return 0, 0, width, height

    values = [float(part.strip()) for part in roi_text.split(",")]
    if len(values) != 4:
        raise SystemExit("ROI 格式错误，应为 x1,y1,x2,y2，例如 0.1,0.1,0.9,0.9 或 100,80,1180,650")

    if all(0.0 <= value <= 1.0 for value in values):
        x1, y1, x2, y2 = values
        values = [x1 * width, y1 * height, x2 * width, y2 * height]

    x1, y1, x2, y2 = [int(round(value)) for value in values]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise SystemExit("ROI 范围无效，必须满足 x2>x1 且 y2>y1")
    return x1, y1, x2, y2


def box_overlap_ratio(box: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> float:
    """Return intersection area divided by the person box area."""
    x1, y1, x2, y2 = box
    rx1, ry1, rx2, ry2 = roi
    inter_x1 = max(x1, rx1)
    inter_y1 = max(y1, ry1)
    inter_x2 = min(x2, rx2)
    inter_y2 = min(y2, ry2)
    intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    return intersection / box_area if box_area else 0.0


def enhance_low_light(frame: np.ndarray) -> np.ndarray:
    """Use CLAHE on the L channel to improve pedestrian detection in dim scenes."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def collect_detections_from_results(
    results: list,
    conf: float,
    iou: float,
    min_box_area: int,
) -> list[tuple[Detection, int | None]]:
    """Collect person detections, keep optional tracker IDs, filter small boxes, and apply NMS."""
    boxes: list[list[int]] = []
    scores: list[float] = []
    track_ids: list[int | None] = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf[0])
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if width * height < min_box_area:
                continue

            track_id = int(box.id[0]) if box.id is not None else None
            boxes.append([x1, y1, width, height])
            scores.append(score)
            track_ids.append(track_id)

    if not boxes:
        return []

    keep_indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf, nms_threshold=iou)
    detections: list[tuple[Detection, int | None]] = []
    for index in np.array(keep_indices).flatten():
        x, y, width, height = boxes[int(index)]
        detection = Detection(x, y, x + width, y + height, scores[int(index)])
        detections.append((detection, track_ids[int(index)]))

    return detections


def track_pedestrians_with_bytetrack(
    model: YOLO,
    frame: np.ndarray,
    imgsz: int,
    display_conf: float,
    track_conf: float,
    iou: float,
    min_box_area: int,
    tracker_config: str,
    roi: tuple[int, int, int, int],
    roi_overlap_threshold: float,
) -> list[TrackedDetection]:
    """Detect and track pedestrians with Ultralytics ByteTrack on CPU."""
    enhanced_frame = enhance_low_light(frame)
    results = model.track(
        source=enhanced_frame,
        imgsz=imgsz,
        conf=track_conf,
        iou=iou,
        classes=[PERSON_CLASS_ID],
        device="cpu",
        tracker=tracker_config,
        persist=True,
        verbose=False,
    )

    tracked: list[TrackedDetection] = []
    for detection, track_id in collect_detections_from_results(results, display_conf, iou, min_box_area):
        overlap = box_overlap_ratio((detection.x1, detection.y1, detection.x2, detection.y2), roi)
        tracked.append(
            TrackedDetection(
                x1=detection.x1,
                y1=detection.y1,
                x2=detection.x2,
                y2=detection.y2,
                score=detection.score,
                track_id=track_id if track_id is not None else 0,
                in_roi=overlap >= roi_overlap_threshold,
                roi_overlap=overlap,
            )
        )
    return tracked


def draw_overlay(
    frame: np.ndarray,
    detections: list[TrackedDetection],
    controller: StationBusinessController,
    roi: tuple[int, int, int, int],
    alerts: list[str],
) -> None:
    """Draw ROI, pedestrian boxes, IDs, confidence scores, counts, dwell times, and alerts."""
    cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 0, 0), 2)
    for detection in detections:
        color = (0, 255, 0) if detection.in_roi else (0, 200, 255)
        cv2.rectangle(frame, (detection.x1, detection.y1), (detection.x2, detection.y2), color, 2)
        label = f"ID {detection.track_id} {detection.score:.2f} ROI {detection.roi_overlap:.2f}"
        cv2.putText(
            frame,
            label,
            (detection.x1, max(20, detection.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    inside_count = controller.inside_count()
    cv2.putText(
        frame,
        f"Inside: {inside_count}/{controller.max_persons}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255) if inside_count > controller.max_persons else (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    y = 75
    for state in controller.active_states()[:6]:
        cv2.putText(
            frame,
            f"ID {state.track_id} stay {format_duration(state.total_stay_seconds)}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255) if state.timeout_alerted else (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 28

    for alert in alerts[-3:]:
        cv2.putText(frame, alert[-70:], (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        y += 26


def build_default_output_path(video_path: Path) -> Path:
    """Build an MP4 output path beside the source video."""
    return video_path.with_name(f"{video_path.stem}_station_control.mp4")


def create_video_writer(capture: cv2.VideoCapture, output_path: Path) -> cv2.VideoWriter:
    """Create an MP4 video writer matching the input video's size and FPS."""
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for codec in ("avc1", "H264", "mp4v"):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if writer.isOpened():
            return writer
    raise SystemExit(f"无法创建输出视频: {output_path}")


def resolve_tracker_config(config_path: str) -> str:
    """Resolve the ByteTrack YAML path, including the script directory default."""
    path = Path(config_path).expanduser()
    if path.is_file():
        return str(path.resolve())

    script_relative_path = Path(__file__).resolve().parent / config_path
    if script_relative_path.is_file():
        return str(script_relative_path)

    raise SystemExit(f"ByteTrack 参数文件不存在: {config_path}")


def open_capture(source: str) -> cv2.VideoCapture:
    """Open a local video file or RTSP/HTTP stream."""
    source_path = Path(source).expanduser()
    if source_path.is_file():
        capture = cv2.VideoCapture(str(source_path.resolve()))
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"无法打开视频源: {source}")
    return capture


def print_runtime_config(args: argparse.Namespace, roi: tuple[int, int, int, int], output_path: Path) -> None:
    """Print effective tunable parameters so default constants are visible at runtime."""
    print(f"开始检测: {args.video}")
    print(f"输出视频: {output_path}")
    print(f"ROI: {roi}, overlap>={args.roi_overlap}")
    print(
        "参数: "
        f"imgsz={args.imgsz}, conf={args.conf}, track_conf={args.track_conf}, "
        f"min_box_area={args.min_box_area}, enter_frames={args.enter_frames}, "
        f"exit_frames={args.exit_frames}, multi_frames={args.multi_frames}, "
        f"timeout_minutes={args.timeout_minutes}, max_persons={args.max_persons}"
    )


def run_detection(args: argparse.Namespace) -> None:
    capture = open_capture(args.video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        source_path = Path(args.video).expanduser()
        output_path = build_default_output_path(source_path if source_path.suffix else Path("rtsp_station_control.mp4")).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    roi = parse_roi(args.roi, width, height)
    model = YOLO(args.model)
    writer = create_video_writer(capture, output_path)
    tracker_config = resolve_tracker_config(args.tracker_config)
    controller = StationBusinessController(
        timeout_seconds=args.timeout_minutes * 60.0,
        max_persons=args.max_persons,
        enter_frames=args.enter_frames,
        exit_frames=args.exit_frames,
        multi_frames=args.multi_frames,
    )

    print_runtime_config(args, roi, output_path)
    if not args.no_display:
        print("按 q 或 Esc 退出。")

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        timestamp = frame_index / fps
        detections = track_pedestrians_with_bytetrack(
            model=model,
            frame=frame,
            imgsz=args.imgsz,
            display_conf=args.conf,
            track_conf=args.track_conf,
            iou=args.iou,
            min_box_area=args.min_box_area,
            tracker_config=tracker_config,
            roi=roi,
            roi_overlap_threshold=args.roi_overlap,
        )
        alerts = controller.update(detections, timestamp, frame_index)
        for alert in alerts:
            print(alert)

        draw_overlay(frame, detections, controller, roi, alerts)
        writer.write(frame)
        print(f"Frame {frame_index}: inside={controller.inside_count()} detected={len(detections)}")

        if not args.no_display:
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

        frame_index += 1

    capture.release()
    writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()
    print("检测结束。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLOv8s CPU 站房人员停留时长与人数合规管控",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="本地 MP4 或 RTSP 地址，默认 001.mp4")
    parser.add_argument("--output", default=None, help="输出 MP4 路径，默认与输入同目录并追加 _station_control")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="YOLOv8 模型权重，默认 yolov8s.pt")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="固定推理尺寸，默认 640")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="最终显示/输出置信度阈值，默认 0.55")
    parser.add_argument("--track-conf", type=float, default=DEFAULT_TRACK_CONF, help="ByteTrack 输入检测阈值，默认 0.25")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU, help="NMS IoU 阈值，默认 0.45")
    parser.add_argument("--tracker-config", default=DEFAULT_TRACKER_CONFIG, help="ByteTrack 参数 YAML")
    parser.add_argument("--min-box-area", type=int, default=DEFAULT_MIN_BOX_AREA, help="最小检测框面积过滤阈值")
    parser.add_argument("--roi", default=None, help="站房 ROI，格式 x1,y1,x2,y2；支持像素或 0-1 归一化，默认全画面")
    parser.add_argument("--roi-overlap", type=float, default=DEFAULT_ROI_OVERLAP, help="人体框与 ROI 重叠占比阈值")
    parser.add_argument("--enter-frames", type=int, default=DEFAULT_ENTER_FRAMES, help="入场连续帧防抖")
    parser.add_argument("--exit-frames", type=int, default=DEFAULT_EXIT_FRAMES, help="离场连续帧防抖")
    parser.add_argument("--multi-frames", type=int, default=DEFAULT_MULTI_FRAMES, help="多人违规连续帧防抖")
    parser.add_argument("--timeout-minutes", type=float, default=DEFAULT_TIMEOUT_MINUTES, help="停留超时阈值，默认 30 分钟")
    parser.add_argument("--max-persons", type=int, default=DEFAULT_MAX_PERSONS, help="合规最大在场人数，默认 1")
    parser.add_argument("--no-display", action="store_true", help="不弹出实时窗口，仅输出标注后 MP4")
    return parser.parse_args()


def main() -> None:
    run_detection(parse_args())


if __name__ == "__main__":
    main()
