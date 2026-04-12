import math
import pandas as pd

class DataPreprocessor:
    """Preprocessor for industrial predictive maintenance dataset."""

    def __init__(self, data_path: str):
        """Initialize with data path.
        
        Args:
            data_path: Path to the CSV file containing the dataset
        """
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None

    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded successfully. Shape: {self.df.shape}")
        return self.df

    def preprocess_data(self) -> tuple:
        """Preprocess the dataset by adding features and balancing classes."""
        print("Preprocessing data...")
        
        # Feature engineering
        self.df['temperature_difference [k]'] = (
            self.df['Process temperature [K]'] - self.df['Air temperature [K]']
        )
        self.df['Power consumption [W]'] = (
            self.df['Torque [Nm]'] * ((2 * math.pi) * self.df["Rotational speed [rpm]"].astype(float) / 60)
        )
        self.df['Power to tempreture ratio [W/K]'] = (
            self.df['Power consumption [W]'] / self.df['temperature_difference [k]']
        )
        self.df['PWF proximity [W]'] = abs(self.df["Power consumption [W]"] - 6250)
        self.df['HDF proximity [W]'] = (
            (self.df["Temperature difference [K]"] - 8.6) * (self.df["Rotational speed [rpm]"] - 1320)
        )
        self.df['Failure type'] = self.df.apply(self._map_failure_type, axis=1)
        
        # Encode categorical features
        self.df['Type'] = self.df['Type'].map({"L": 1, "M": 2, "H": 3})

        # Normalize column names
        self.df.columns = self.df.columns.str.replace('[', '').str.replace(']', '')

        # Data balancing
        df_failures = self.df[self.df['Machine failure'] == 1]
        df_no_failures = self.df[self.df['Machine failure'] == 0]

        failure_count = len(df_failures)
        target_no_failure_count = failure_count * 4
        actual_sample_size = min(len(df_no_failures), target_no_failure_count)

        df_no_failures_sampled = df_no_failures.sample(n=actual_sample_size, random_state=42)
        balanced_df = pd.concat([df_failures, df_no_failures_sampled], ignore_index=True)

        # Prepare features and target
        drop_columns = ['UID', 'Product ID', 'Machine failure', 'Failure type', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        self.X = balanced_df.drop(columns=drop_columns)
        self.y = balanced_df['Failure type']

        return self.X, self.y

    @staticmethod
    def _map_failure_type(row) -> int:
        """Map failure indicators to failure type.
        Args:
            row: A row of the DataFrame
        """
        if row["Machine failure"] == 0:
            return 0
        elif row["TWF"] == 1:
            return 1
        elif row["HDF"] == 1:
            return 2
        elif row["PWF"] == 1:
            return 3
        elif row["OSF"] == 1:
            return 4
        return 0
        

        
