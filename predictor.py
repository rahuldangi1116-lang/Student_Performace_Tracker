
import joblib
import pandas as pd
from pathlib import Path


class StudentPerformancePredictor:

    # Categorical features
    categorical_options = {

        "Parental_Involvement": [
            "Low", "Medium", "High"
        ],

        "Access_to_Resources": [
            "Low", "Medium", "High"
        ],

        "Extracurricular_Activities": [
            "Yes", "No"
        ],

        "Motivation_Level": [
            "Low", "Medium", "High"
        ],

        "Internet_Access": [
            "Yes", "No"
        ],

        "Family_Income": [
            "Low", "Medium", "High"
        ],

        "Teacher_Quality": [
            "Low", "Medium", "High"
        ],

        "School_Type": [
            "Public", "Private"
        ],

        "Peer_Influence": [
            "Positive", "Neutral", "Negative"
        ],

        "Learning_Disabilities": [
            "Yes", "No"
        ],

        "Parental_Education_Level": [
            "High School", "College", "Postgraduate"
        ],

        "Distance_from_Home": [
            "Near", "Moderate", "Far"
        ],

        "Gender": [
            "Male", "Female"
        ]
    }

    # Numeric features:
    # (minimum, maximum, default, step)

    numeric_ranges = {

        "Hours_Studied": (0, 24, 5, 1),

        "Attendance": (0, 100, 85, 1),

        "Sleep_Hours": (0, 24, 7, 1),

        "Previous_Scores": (0, 100, 70, 1),

        "Tutoring_Sessions": (0, 20, 2, 1),

        "Physical_Activity": (0, 10, 3, 1)
    }

    def __init__(self, model_path):

        model_path = Path(model_path)

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        # Load trained ML model
        self.model = joblib.load(model_path)

        # Get original training feature names
        self.feature_names = getattr(
            self.model,
            "feature_names_in_",
            None
        )

        if self.feature_names is None:

            raise ValueError(
                "Model does not contain feature_names_in_. "
                "Retrain your model using a pandas DataFrame."
            )

        self.feature_names = list(
            self.feature_names
        )

    def predict(self, input_values):

        # Check missing features
        missing_features = [
            feature
            for feature in self.feature_names
            if feature not in input_values
        ]

        if missing_features:

            raise ValueError(
                f"Missing features: {missing_features}"
            )

        # Create DataFrame in exact training order
        input_data = pd.DataFrame(
            [
                {
                    feature: input_values[feature]
                    for feature in self.feature_names
                }
            ]
        )

        # Generate prediction
        prediction = self.model.predict(
            input_data
        )[0]

        # Keep score within 0-100
        prediction = max(
            0,
            min(100, float(prediction))
        )

        return prediction
