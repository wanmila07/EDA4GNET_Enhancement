"""
Data loading utilities for EDA4GNET Framework Enhancement
"""

import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

class DataLoader:
    def __init__(self):
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.sample_dir = self.data_dir / "sample"
    
    def load_dataset(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load dataset from CSV file
        
        Args:
            file_path: Path to CSV file
            sample_size: Optional sample size for testing
            
        Returns:
            Loaded DataFrame
        """
        start_time = time.time()
        
        try:
            if sample_size:
                # Load only specified number of rows
                df = pd.read_csv(file_path, nrows=sample_size)
                print(f"Loaded sample of {len(df):,} rows from {file_path}")
            else:
                # Load entire file
                df = pd.read_csv(file_path)
                print(f"Loaded {len(df):,} rows from {file_path}")
            
            loading_time = time.time() - start_time
            print(f"Loading time: {loading_time:.3f} seconds")
            
            return df
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            raise
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for analysis
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Create a copy to avoid modifying original
        processed_df = df.copy()
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Add row indices as IDs if not present
        if 'id' not in processed_df.columns:
            processed_df['id'] = range(len(processed_df))
        
        print(f"Preprocessed dataset: {len(processed_df):,} rows, {len(processed_df.columns)} columns")
        
        return processed_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # Fill missing numeric values with 0
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
        
        # Fill missing categorical values with 'unknown'
        categorical_columns = df.select_dtypes(include=['object']).columns
        df[categorical_columns] = df[categorical_columns].fillna('unknown')
        
        return df
    
    def get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get information about the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'data_types': df.dtypes.value_counts().to_dict(),
            'missing_values': df.isnull().sum().sum(),
            'column_names': list(df.columns)
        }
        
        # Check for target column
        if 'delay' in df.columns:
            info['target_distribution'] = df['delay'].value_counts().to_dict()
        
        return info
    
    def create_balanced_sample(self, df: pd.DataFrame, target_col: str = 'delay', 
                              normal_samples: int = 5000, attack_samples: int = 1000) -> pd.DataFrame:
        """
        Create a balanced sample from the dataset
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            normal_samples: Number of normal samples
            attack_samples: Number of samples per attack type
            
        Returns:
            Balanced sample DataFrame
        """
        samples = []
        
        # Get normal traffic samples
        normal_df = df[df[target_col] == 'normal']
        if len(normal_df) >= normal_samples:
            normal_sample = normal_df.sample(normal_samples, random_state=42)
        else:
            normal_sample = normal_df
        samples.append(normal_sample)
        
        # Get attack samples
        attack_types = df[df[target_col] != 'normal'][target_col].unique()
        
        for attack in attack_types:
            attack_df = df[df[target_col] == attack]
            sample_size = min(attack_samples, len(attack_df))
            if sample_size > 0:
                attack_sample = attack_df.sample(sample_size, random_state=42)
                samples.append(attack_sample)
        
        # Combine all samples
        balanced_df = pd.concat(samples, ignore_index=True)
        
        print(f"Created balanced sample with {len(balanced_df):,} records")
        print("Sample distribution:")
        print(balanced_df[target_col].value_counts())
        
        return balanced_df
    
    def create_stratified_samples(self, train_df: pd.DataFrame, test_df: pd.DataFrame, 
                                 target_col: str = 'delay') -> Dict[str, str]:
        """
        Create stratified samples of different sizes
        
        Args:
            train_df: Training DataFrame
            test_df: Testing DataFrame
            target_col: Target column for stratification
            
        Returns:
            Dictionary with sample file paths
        """
        # Ensure sample directory exists
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        sample_sizes = [10000, 100000]
        sample_paths = {}
        
        # Function for stratified sampling
        def create_stratified_sample(df: pd.DataFrame, size: int) -> pd.DataFrame:
            if size >= len(df):
                return df.copy()
            
            # Calculate proportional sample sizes for each class
            value_counts = df[target_col].value_counts()
            sample_dict = {}
            
            for class_name, count in value_counts.items():
                class_sample_size = max(1, int(size * count / len(df)))
                class_df = df[df[target_col] == class_name]
                
                if len(class_df) >= class_sample_size:
                    sample_dict[class_name] = class_df.sample(class_sample_size, random_state=42)
                else:
                    sample_dict[class_name] = class_df
            
            return pd.concat(list(sample_dict.values()), ignore_index=True)
        
        # Create training samples
        for size in sample_sizes:
            train_sample = create_stratified_sample(train_df, size)
            train_sample_path = self.sample_dir / f"sample_train_{size//1000}k.csv"
            train_sample.to_csv(train_sample_path, index=False)
            sample_paths[f'train_{size//1000}k'] = str(train_sample_path)
            print(f"Created training sample: {train_sample_path} ({len(train_sample):,} records)")
        
        # Create testing samples
        for size in sample_sizes:
            test_sample = create_stratified_sample(test_df, size)
            test_sample_path = self.sample_dir / f"sample_test_{size//1000}k.csv"
            test_sample.to_csv(test_sample_path, index=False)
            sample_paths[f'test_{size//1000}k'] = str(test_sample_path)
            print(f"Created testing sample: {test_sample_path} ({len(test_sample):,} records)")
        
        # Create balanced sample for development
        balanced_sample = self.create_balanced_sample(train_df, target_col)
        balanced_sample_path = self.sample_dir / "balanced_sample.csv"
        balanced_sample.to_csv(balanced_sample_path, index=False)
        sample_paths['balanced'] = str(balanced_sample_path)
        print(f"Created balanced sample: {balanced_sample_path} ({len(balanced_sample):,} records)")
        
        return sample_paths
    
    def get_sample_paths(self) -> Dict[str, str]:
        """
        Get paths to sample files
        
        Returns:
            Dictionary with sample file paths
        """
        return {
            'balanced_sample': str(self.sample_dir / 'balanced_sample.csv'),
            'train_10k': str(self.sample_dir / 'sample_train_10k.csv'),
            'test_10k': str(self.sample_dir / 'sample_test_10k.csv'),
            'train_100k': str(self.sample_dir / 'sample_train_100k.csv'),
            'test_100k': str(self.sample_dir / 'sample_test_100k.csv'),
            'full_train': str(self.raw_dir / 'train.csv'),
            'full_test': str(self.raw_dir / 'test.csv')
        }
    
    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate dataset for EDA4GNET processing
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        
        # Check basic requirements
        if len(df) == 0:
            validation_results['errors'].append("Dataset is empty")
            validation_results['is_valid'] = False
        
        if len(df.columns) < 10:
            validation_results['warnings'].append(f"Dataset has only {len(df.columns)} columns, expected more for GOOSE analysis")
        
        # Check for key columns
        key_columns = ['delay', 'StNum', 'sqDiff', 'timeFromLastChange']
        missing_key_columns = [col for col in key_columns if col not in df.columns]
        
        if missing_key_columns:
            validation_results['warnings'].append(f"Missing key columns: {missing_key_columns}")
        
        # Check target column
        if 'delay' not in df.columns:
            validation_results['errors'].append("Target column 'delay' not found")
            validation_results['is_valid'] = False
        else:
            # Check class distribution
            class_dist = df['delay'].value_counts()
            normal_pct = (df['delay'] == 'normal').mean() * 100
            
            if normal_pct > 98:
                validation_results['warnings'].append(f"Dataset is highly imbalanced ({normal_pct:.1f}% normal traffic)")
            
            if len(class_dist) < 2:
                validation_results['warnings'].append("Dataset contains only one class")
        
        # Check data quality
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 10:
            validation_results['warnings'].append(f"High percentage of missing values: {missing_pct:.1f}%")
        
        # Memory usage check
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        if memory_mb > 1000:  # 1GB
            validation_results['recommendations'].append(f"Large dataset ({memory_mb:.1f} MB), consider using samples for development")
        
        # Performance recommendations
        if len(df) > 100000:
            validation_results['recommendations'].append("Large dataset detected, parallel processing will provide significant benefits")
        
        return validation_results
    
    def analyze_computational_profile(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze the computational profile of operations on the dataset
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with operation timing results
        """
        def measure_operation(operation_name: str, operation_func):
            start_time = time.time()
            try:
                result = operation_func()
                end_time = time.time()
                return end_time - start_time
            except Exception as e:
                print(f"Error in {operation_name}: {e}")
                return 0.0
        
        # Get numeric columns for testing
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:10]  # Limit for performance
        
        results = {}
        
        print("Analyzing computational profile...")
        
        # CPU-bound operations
        if len(numeric_cols) >= 3:
            results['calculate_differences'] = measure_operation(
                "Calculate differences",
                lambda: df[numeric_cols[:3]].diff()
            )
            
            results['calculate_correlation'] = measure_operation(
                "Calculate correlation matrix",
                lambda: df[numeric_cols].corr()
            )
        
        if 'delay' in df.columns and len(numeric_cols) >= 2:
            results['groupby_operation'] = measure_operation(
                "Group by attack type",
                lambda: df.groupby('delay')[numeric_cols[:2]].mean()
            )
        
        # I/O-bound operations (test with sample)
        sample_df = df.sample(min(1000, len(df)))
        temp_file = self.sample_dir / "temp_profile_test.csv"
        
        results['write_csv'] = measure_operation(
            "Write CSV file",
            lambda: sample_df.to_csv(temp_file, index=False)
        )
        
        results['read_csv'] = measure_operation(
            "Read CSV file",
            lambda: pd.read_csv(temp_file)
        )
        
        # Clean up
        if temp_file.exists():
            temp_file.unlink()
        
        # Statistical operations
        if len(numeric_cols) > 0:
            results['statistical_summary'] = measure_operation(
                "Statistical summary",
                lambda: df[numeric_cols].describe()
            )
        
        print("Computational profile analysis completed")
        
        return results
    
    def print_dataset_summary(self, df: pd.DataFrame):
        """
        Print a comprehensive summary of the dataset
        
        Args:
            df: Input DataFrame
        """
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        
        # Basic info
        info = self.get_dataset_info(df)
        print(f"Rows: {info['rows']:,}")
        print(f"Columns: {info['columns']}")
        print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        print(f"Missing Values: {info['missing_values']:,}")
        
        # Data types
        print(f"\nData Types:")
        for dtype, count in info['data_types'].items():
            print(f"  {dtype}: {count} columns")
        
        # Target distribution
        if 'target_distribution' in info:
            print(f"\nTarget Distribution:")
            total = sum(info['target_distribution'].values())
            for class_name, count in info['target_distribution'].items():
                pct = (count / total) * 100
                print(f"  {class_name}: {count:,} ({pct:.1f}%)")
        
        # Validation
        validation = self.validate_dataset(df)
        if not validation['is_valid']:
            print(f"\n⚠️  VALIDATION ERRORS:")
            for error in validation['errors']:
                print(f"  • {error}")
        
        if validation['warnings']:
            print(f"\n⚠️  WARNINGS:")
            for warning in validation['warnings']:
                print(f"  • {warning}")
        
        if validation['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in validation['recommendations']:
                print(f"  • {rec}")
        
        print("=" * 60)