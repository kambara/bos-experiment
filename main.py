import argparse
from dataclasses import dataclass
from typing import Tuple, Union

import cv2
import ffmpeg
import numpy as np
from cv2 import VideoWriter


# -----------------------------------------------------------------------------
# Point
#
@dataclass
class Point:
    x: float
    y: float

    def to_list(self) -> list:
        return [self.x, self.y]

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"


# -----------------------------------------------------------------------------
# マーカー
#
class Marker:
    def __init__(self, corners) -> None:
        def get_corner(index):
            return corners[0][index]

        def create_point(index):
            corner = get_corner(index)
            return Point(int(corner[0]), int(corner[1]))

        self.top_left = create_point(0)
        self.top_right = create_point(1)
        self.bottom_right = create_point(2)
        self.bottom_left = create_point(3)

    @property
    def center(self) -> Point:
        cx = (
            self.top_left.x
            + self.top_right.x
            + self.bottom_right.x
            + self.bottom_left.x
        ) / 4
        cy = (
            self.top_left.y
            + self.top_right.y
            + self.bottom_right.y
            + self.bottom_left.y
        ) / 4
        return Point(cx, cy)

    def __str__(self) -> str:
        return (
            f"TL {self.top_left}, "
            f"TR {self.top_right}, "
            f"BR {self.bottom_right}, "
            f"BL {self.bottom_left}"
        )


# -----------------------------------------------------------------------------
# 4頂点ポリゴン
#
@dataclass
class QuadPoints:
    top_left: Union[Point, None] = None
    top_right: Union[Point, None] = None
    bottom_right: Union[Point, None] = None
    bottom_left: Union[Point, None] = None

    def __str__(self) -> str:
        return (
            f"TL {self.top_left}, "
            f"TR {self.top_right}, "
            f"BR {self.bottom_right}, "
            f"BL {self.bottom_left}"
        )


# -----------------------------------------------------------------------------
# 背景の四隅のマーカー
#
class QuadMarkers:
    def __init__(self) -> None:
        self.top_left: Union[Marker, None] = None
        self.top_right: Union[Marker, None] = None
        self.bottom_right: Union[Marker, None] = None
        self.bottom_left: Union[Marker, None] = None

    def get_quad_points(self) -> QuadPoints:
        return QuadPoints(
            self.top_left.center if self.top_left else None,
            self.top_right.center if self.top_right else None,
            self.bottom_right.center if self.bottom_right else None,
            self.bottom_left.center if self.bottom_left else None,
        )

    def __str__(self) -> str:
        return (
            f"(\n"
            f"  TL Marker: {self.top_left},\n"
            f"  TR Marker: {self.top_right},\n"
            f"  BR Marker: {self.bottom_right},\n"
            f"  BL Marker: {self.bottom_left}\n"
            f")"
        )


# -----------------------------------------------------------------------------
# マーカーで囲まれたエリアを抽出
#
def extract_marker_area(frame, dictionary, video_size):
    # マーカー位置検出
    quad_markers = detect_quad_markers(frame, dictionary)
    if (
        quad_markers.top_left is None
        or quad_markers.top_right is None
        or quad_markers.bottom_right is None
        or quad_markers.bottom_left is None
    ):
        print("Error: Markers are not detected")
        print(quad_markers)
        raise Exception
    quad_corners = quad_markers.get_quad_points()

    # TODO: マーカーが見つからなかった場合は前のフレームのマーカーを使用したい

    # 射影変換
    src = np.float32(
        [
            quad_corners.top_left.to_list(),
            quad_corners.top_right.to_list(),
            quad_corners.bottom_right.to_list(),
            quad_corners.bottom_left.to_list(),
        ]
    )
    dest_width = int(video_size[0])
    dest_height = int(video_size[1])
    dest = np.float32(
        [
            [0, 0],
            [dest_width, 0],
            [dest_width, dest_height],
            [0, dest_height],
        ]
    )
    mat = cv2.getPerspectiveTransform(src, dest)
    frame = cv2.warpPerspective(frame, mat, (dest_width, dest_height))
    return frame


def detect_quad_markers(frame, dictionary) -> QuadMarkers:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    res, frame = cv2.threshold(frame, 70, 255, cv2.THRESH_TOZERO)
    corners, ids, rejects = cv2.aruco.detectMarkers(frame, dictionary)
    quad_markers = QuadMarkers()
    for index, id in enumerate(ids):
        marker = Marker(corners[index])
        if id[0] == 0:
            quad_markers.top_left = marker
        elif id[0] == 1:
            quad_markers.top_right = marker
        elif id[0] == 2:
            quad_markers.bottom_right = marker
        elif id[0] == 3:
            quad_markers.bottom_left = marker
    return quad_markers


# CLAHE
def applyClahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return clahe.apply(image)


def showImage(image) -> bool:
    cv2.imshow("frame", image)
    key = cv2.waitKey(1)
    return key == 27  # Esc key


def create_video_writer(video, temp_video_path: str) -> VideoWriter:
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    size = get_video_size(video)
    return cv2.VideoWriter(temp_video_path, fourcc, round(fps), size)


def get_video_size(video) -> Tuple[int, int]:
    return (
        int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


# Motion JPGをH.264のmp4に変換
def convert_video(input: str, output: str) -> None:
    ffmpeg.input(input).output(output, vcodec="libx264").run(
        overwrite_output=True
    )


# コマンド引数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Output video path")
    parser.add_argument("input", help="Input video path")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# main
#
def main():
    args = parse_args()

    # Load video
    video = cv2.VideoCapture(args.input)
    video_size = get_video_size(video)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Video writer
    temp_video_path = "_temp.avi"
    video_writer = create_video_writer(video, temp_video_path)

    # ArUco
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # 最初のフレームを背景として保存
    success, frame = video.read()
    # frame = applyClahe(frame)
    background = extract_marker_area(frame, dictionary, video_size)

    count = 1
    while True:
        # Next frame
        success, frame = video.read()
        if not success:
            break
        count += 1
        print(f"Frame {count} / {frame_count}")
        # frame = applyClahe(frame)
        try:
            frame = extract_marker_area(frame, dictionary, video_size)
        except Exception:
            continue

        # Diff
        frame = cv2.absdiff(frame, background)

        # コントラストを上げる
        # ret, frame = cv2.threshold(frame, 10, 255, cv2.THRESH_TOZERO)
        ret, frame = cv2.threshold(frame, 240, 255, cv2.THRESH_TOZERO_INV)
        frame = frame * 3

        # Blur
        # frame = cv2.GaussianBlur(frame, (3, 3), 0)
        # frame = cv2.medianBlur(frame, 5)
        # frame = cv2.blur(frame, (3, 3))
        # frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

        # Write
        if args.output is not None:
            video_writer.write(frame)

        # Show
        escape = showImage(frame)
        if escape:
            break

    video.release()
    cv2.destroyAllWindows()
    video_writer.release()
    if args.output is not None:
        convert_video(temp_video_path, args.output)


main()
