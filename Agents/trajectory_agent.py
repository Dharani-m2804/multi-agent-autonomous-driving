from math import sqrt


class TrajectoryPredictionAgent:
    """
    Predicts future object positions using a constant-velocity model.

    Prediction horizons:
        0.5 s
        1.0 s
        1.5 s
        2.0 s

    Input:
        TrackerAgent output

    Output:
        Current position
        Velocity
        Predicted future positions
    """

    def __init__(
        self,
        prediction_horizons=None
    ):

        if prediction_horizons is None:
            prediction_horizons = [
                0.5,
                1.0,
                1.5,
                2.0
            ]

        self.prediction_horizons = prediction_horizons

        print()
        print("=" * 65)
        print("INITIALIZING TRAJECTORY PREDICTION AGENT")
        print("=" * 65)

        print(
            "Prediction horizons:",
            ", ".join(
                f"{t:.1f}s"
                for t in self.prediction_horizons
            )
        )

        print("=" * 65)

    # =============================================================
    # PREDICT ONE POSITION
    # =============================================================

    @staticmethod
    def predict_position(
        center,
        velocity,
        delta_t
    ):
        """
        Constant-velocity prediction:

            x(t) = x0 + vx * t
            y(t) = y0 + vy * t
        """

        x = center[0]
        y = center[1]

        vx = velocity["x"]
        vy = velocity["y"]

        predicted_x = x + vx * delta_t
        predicted_y = y + vy * delta_t

        return [
            round(predicted_x, 2),
            round(predicted_y, 2)
        ]

    # =============================================================
    # PREDICT ONE TRACK
    # =============================================================

    def predict_track(
        self,
        track
    ):

        center = track["center"]

        velocity = track["velocity"]

        predictions = []

        for delta_t in self.prediction_horizons:

            position = self.predict_position(
                center,
                velocity,
                delta_t
            )

            predictions.append(
                {
                    "time_seconds":
                        delta_t,

                    "position":
                        position
                }
            )

        return {

            "track_id":
                track["track_id"],

            "class":
                track["class"],

            "confidence":
                track["confidence"],

            "current_position":
                center,

            "velocity":
                velocity,

            "speed":
                track["speed"],

            "predictions":
                predictions
        }

    # =============================================================
    # PREDICT ALL TRACKS
    # =============================================================

    def predict(
        self,
        tracks
    ):

        predictions = []

        for track in tracks:

            prediction = self.predict_track(
                track
            )

            predictions.append(
                prediction
            )

        return predictions

    # =============================================================
    # DISPLAY
    # =============================================================

    def display(
        self,
        predictions
    ):

        print()
        print("=" * 65)
        print("TRAJECTORY PREDICTION RESULTS")
        print("=" * 65)

        print(
            f"Objects predicted: {len(predictions)}"
        )

        print()

        for prediction in predictions:

            print(
                f"ID {prediction['track_id']:03d} | "
                f"{prediction['class']:<20}"
            )

            print(
                f"  Current position : "
                f"{prediction['current_position']}"
            )

            print(
                f"  Velocity         : "
                f"({prediction['velocity']['x']:.2f}, "
                f"{prediction['velocity']['y']:.2f}) px/s"
            )

            print(
                f"  Speed            : "
                f"{prediction['speed']:.2f} px/s"
            )

            for future in prediction["predictions"]:

                print(
                    f"  +{future['time_seconds']:.1f}s"
                    f" → {future['position']}"
                )

            print()

        print("=" * 65)


# =================================================================
# TEST
# =================================================================

if __name__ == "__main__":

    print()
    print("=" * 65)
    print("TRAJECTORY PREDICTION AGENT TEST")
    print("=" * 65)

    # -------------------------------------------------------------
    # Import tracker
    # -------------------------------------------------------------

    from tracker_agent import TrackerAgent

    tracker = TrackerAgent(
        max_distance=120,
        max_missed_frames=5
    )

    predictor = TrajectoryPredictionAgent()

    # -------------------------------------------------------------
    # Frame 1
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

    # -------------------------------------------------------------
    # Frame 2
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Frame 3
    # -------------------------------------------------------------

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
    # Update tracker
    # -------------------------------------------------------------

    tracker.update(
        frame_1,
        fps=30
    )

    tracker.update(
        frame_2,
        fps=30
    )

    tracks = tracker.update(
        frame_3,
        fps=30
    )

    tracker.display(tracks)

    # -------------------------------------------------------------
    # Predict trajectories
    # -------------------------------------------------------------

    predictions = predictor.predict(
        tracks
    )

    predictor.display(
        predictions
    )

    print()
    print("=" * 65)
    print("TRAJECTORY TEST COMPLETE")
    print("=" * 65)