from pathlib import Path
from collections import OrderedDict
import math
import time


class Track:
    """
    Represents one tracked object.

    Stores:
        - persistent object ID
        - class
        - bounding box
        - center position
        - velocity
        - track history
        - confidence
    """

    def __init__(self, track_id, detection, frame_number):
        self.track_id = track_id

        self.class_name = detection["class"]
        self.class_id = detection.get("class_id", -1)
        self.confidence = detection["confidence"]

        self.bbox = detection["bbox"]

        self.center = self._get_center(self.bbox)

        self.previous_center = self.center

        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.frame_number = frame_number

        self.frames_seen = 1
        self.missed_frames = 0

        self.history = [self.center]

    @staticmethod
    def _get_center(bbox):
        x1, y1, x2, y2 = bbox

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        return center_x, center_y

    def update(self, detection, frame_number, fps=30.0):
        """
        Update an existing track with a new detection.
        """

        new_bbox = detection["bbox"]
        new_center = self._get_center(new_bbox)

        old_center = self.center

        self.previous_center = old_center
        self.center = new_center

        self.bbox = new_bbox

        self.confidence = detection["confidence"]

        self.frame_number = frame_number

        self.frames_seen += 1
        self.missed_frames = 0

        # ---------------------------------------------------------
        # Pixel displacement
        # ---------------------------------------------------------

        dx = new_center[0] - old_center[0]
        dy = new_center[1] - old_center[1]

        # ---------------------------------------------------------
        # Velocity in pixels / second
        # ---------------------------------------------------------

        if fps > 0:

            self.velocity_x = dx * fps
            self.velocity_y = dy * fps

        else:

            self.velocity_x = dx
            self.velocity_y = dy

        # ---------------------------------------------------------
        # History
        # ---------------------------------------------------------

        self.history.append(new_center)

        # Keep only recent history
        if len(self.history) > 30:
            self.history.pop(0)

    def mark_missed(self):
        self.missed_frames += 1

    def speed(self):
        """
        Magnitude of velocity.
        """

        return math.sqrt(
            self.velocity_x ** 2
            +
            self.velocity_y ** 2
        )


class TrackerAgent:
    """
    Lightweight multi-object tracking agent.

    Input:
        YOLO detections from consecutive frames.

    Output:
        Persistent object IDs
        Bounding boxes
        Center positions
        Velocity vectors
        Track history

    The tracker uses a lightweight centroid-based association
    mechanism. It is intentionally simple and explainable.
    """

    def __init__(
        self,
        max_distance=100.0,
        max_missed_frames=10
    ):

        print()
        print("=" * 65)
        print("INITIALIZING TRACKER AGENT")
        print("=" * 65)

        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames

        self.next_track_id = 1

        self.tracks = OrderedDict()

        self.frame_number = 0

        print("Tracking module loaded.")
        print(f"Maximum association distance : {max_distance} pixels")
        print(f"Maximum missed frames        : {max_missed_frames}")

        print("=" * 65)

    # =============================================================
    # DISTANCE
    # =============================================================

    @staticmethod
    def centroid_distance(center_a, center_b):

        dx = center_a[0] - center_b[0]
        dy = center_a[1] - center_b[1]

        return math.sqrt(
            dx * dx + dy * dy
        )

    # =============================================================
    # CENTER
    # =============================================================

    @staticmethod
    def get_center(bbox):

        x1, y1, x2, y2 = bbox

        return (
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0
        )

    # =============================================================
    # CREATE TRACK
    # =============================================================

    def _create_track(
        self,
        detection
    ):

        track = Track(
            track_id=self.next_track_id,
            detection=detection,
            frame_number=self.frame_number
        )

        self.tracks[
            self.next_track_id
        ] = track

        self.next_track_id += 1

    # =============================================================
    # MATCH DETECTION TO TRACK
    # =============================================================

    def _find_best_track(
        self,
        detection,
        available_track_ids
    ):

        detection_center = self.get_center(
            detection["bbox"]
        )

        best_track_id = None
        best_distance = float("inf")

        for track_id in available_track_ids:

            track = self.tracks[track_id]

            # -----------------------------------------------------
            # Do not match different object classes
            # -----------------------------------------------------

            if track.class_name != detection["class"]:
                continue

            distance = self.centroid_distance(
                detection_center,
                track.center
            )

            if (
                distance < best_distance
                and
                distance <= self.max_distance
            ):

                best_distance = distance
                best_track_id = track_id

        return best_track_id

    # =============================================================
    # UPDATE TRACKS
    # =============================================================

    def update(
        self,
        detections,
        fps=30.0
    ):

        self.frame_number += 1

        # ---------------------------------------------------------
        # First frame
        # ---------------------------------------------------------

        if not self.tracks:

            for detection in detections:

                self._create_track(
                    detection
                )

            return self.get_active_tracks()

        # ---------------------------------------------------------
        # Existing tracks
        # ---------------------------------------------------------

        available_track_ids = list(
            self.tracks.keys()
        )

        matched_tracks = set()

        # ---------------------------------------------------------
        # Associate detections
        # ---------------------------------------------------------

        for detection in detections:

            best_track_id = self._find_best_track(
                detection,
                [
                    track_id
                    for track_id in available_track_ids
                    if track_id not in matched_tracks
                ]
            )

            if best_track_id is not None:

                track = self.tracks[
                    best_track_id
                ]

                track.update(
                    detection,
                    self.frame_number,
                    fps
                )

                matched_tracks.add(
                    best_track_id
                )

            else:

                self._create_track(
                    detection
                )

        # ---------------------------------------------------------
        # Mark unmatched tracks
        # ---------------------------------------------------------

        for track_id in list(
            self.tracks.keys()
        ):

            if track_id not in matched_tracks:

                self.tracks[
                    track_id
                ].mark_missed()

        # ---------------------------------------------------------
        # Remove old tracks
        # ---------------------------------------------------------

        for track_id in list(
            self.tracks.keys()
        ):

            if (
                self.tracks[
                    track_id
                ].missed_frames
                >
                self.max_missed_frames
            ):

                del self.tracks[
                    track_id
                ]

        return self.get_active_tracks()

    # =============================================================
    # ACTIVE TRACKS
    # =============================================================

    def get_active_tracks(self):

        active_tracks = []

        for track_id, track in self.tracks.items():

            if track.missed_frames == 0:

                active_tracks.append(
                    self._track_to_dict(track)
                )

        return active_tracks

    # =============================================================
    # TRACK → DICTIONARY
    # =============================================================

    @staticmethod
    def _track_to_dict(track):

        return {

            "track_id":
                track.track_id,

            "class_id":
                track.class_id,

            "class":
                track.class_name,

            "confidence":
                round(
                    track.confidence,
                    4
                ),

            "bbox":
                [
                    round(value, 2)
                    for value in track.bbox
                ],

            "center":
                [
                    round(
                        track.center[0],
                        2
                    ),

                    round(
                        track.center[1],
                        2
                    )
                ],

            "previous_center":
                [
                    round(
                        track.previous_center[0],
                        2
                    ),

                    round(
                        track.previous_center[1],
                        2
                    )
                ],

            "velocity":
                {
                    "x":
                        round(
                            track.velocity_x,
                            2
                        ),

                    "y":
                        round(
                            track.velocity_y,
                            2
                        )
                },

            "speed":
                round(
                    track.speed(),
                    2
                ),

            "frames_seen":
                track.frames_seen,

            "history":
                [
                    [
                        round(point[0], 2),
                        round(point[1], 2)
                    ]
                    for point in track.history
                ]
        }

    # =============================================================
    # DISPLAY
    # =============================================================

    def display(
        self,
        tracks
    ):

        print()
        print("=" * 65)
        print("TRACKER AGENT RESULTS")
        print("=" * 65)

        print(
            f"Frame number : {self.frame_number}"
        )

        print(
            f"Active tracks: {len(tracks)}"
        )

        print()

        for track in tracks:

            velocity = track[
                "velocity"
            ]

            print(
                f"ID {track['track_id']:03d} | "
                f"{track['class']:<20} | "
                f"confidence={track['confidence']:.2f} | "
                f"center=({track['center'][0]:.1f}, "
                f"{track['center'][1]:.1f}) | "
                f"velocity=("
                f"{velocity['x']:.1f}, "
                f"{velocity['y']:.1f}) | "
                f"speed={track['speed']:.1f}"
            )

        print("=" * 65)

    # =============================================================
    # RESET
    # =============================================================

    def reset(self):

        self.tracks.clear()

        self.next_track_id = 1

        self.frame_number = 0


# =================================================================
# TEST TRACKER WITH SYNTHETIC CONSECUTIVE DETECTIONS
# =================================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("TRACKER AGENT TEST")
    print("=" * 65)

    tracker = TrackerAgent(
        max_distance=120,
        max_missed_frames=5
    )

    # -------------------------------------------------------------
    # Simulated consecutive frames
    # -------------------------------------------------------------

    frame_1 = [

        {
            "class_id": 2,
            "class": "car",
            "confidence": 0.91,
            "bbox": [
                100,
                300,
                200,
                400
            ]
        },

        {
            "class_id": 6,
            "class": "pedestrian",
            "confidence": 0.88,
            "bbox": [
                500,
                300,
                540,
                390
            ]
        }
    ]

    frame_2 = [

        {
            "class_id": 2,
            "class": "car",
            "confidence": 0.92,
            "bbox": [
                110,
                300,
                210,
                400
            ]
        },

        {
            "class_id": 6,
            "class": "pedestrian",
            "confidence": 0.89,
            "bbox": [
                490,
                300,
                530,
                390
            ]
        }
    ]

    frame_3 = [

        {
            "class_id": 2,
            "class": "car",
            "confidence": 0.93,
            "bbox": [
                120,
                300,
                220,
                400
            ]
        },

        {
            "class_id": 6,
            "class": "pedestrian",
            "confidence": 0.90,
            "bbox": [
                480,
                300,
                520,
                390
            ]
        }
    ]

    # -------------------------------------------------------------
    # Process frames
    # -------------------------------------------------------------

    print()
    print("FRAME 1")
    tracks = tracker.update(
        frame_1,
        fps=30
    )
    tracker.display(tracks)

    print()
    print("FRAME 2")
    tracks = tracker.update(
        frame_2,
        fps=30
    )
    tracker.display(tracks)

    print()
    print("FRAME 3")
    tracks = tracker.update(
        frame_3,
        fps=30
    )
    tracker.display(tracks)

    print()
    print("=" * 65)
    print("TRACKER TEST COMPLETE")
    print("=" * 65)