#!/usr/bin/env python3
"""
Direct File Processor - Optimized implementation for EDA4GNET Framework Enhancement
"""

import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Tuple
import threading
from queue import Queue


class DirectFileProcessor:
    def __init__(self, num_threads: int = 4, chunk_size: int = 10000):
        """
        Initialize Direct File Processor
        
        Args:
            num_threads: Number of threads for parallel processing
            chunk_size: Size of chunks for processing
        """
        self.num_threads = num_threads
        self.chunk_size = chunk_size
        self.results_queue = Queue()
    
    def process_file(self, file_path: str, attack_detection: bool = True) -> Dict[str, Any]:
        """
        Process file directly with optimized approach
        
        Args:
            file_path: Path to CSV file
            attack_detection: Whether to perform attack detection
            
        Returns:
            Processing results dictionary
        """
        start_time = time.time()
        
        print(f"Processing file: {file_path}")
        print(f"Using {self.num_threads} threads with chunk size {self.chunk_size}")
        
        # Get file metadata without loading entire file
        try:
            with open(file_path, 'r') as f:
                header = f.readline().strip().split(',')
                
            # Count total rows
            total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract header
            
        except Exception as e:
            print(f"Error reading file metadata: {e}")
            return {'error': str(e)}
        
        print(f"File contains {total_rows:,} rows with {len(header)} columns")
        
        # Divide file into chunks
        chunks = []
        for i in range(0, total_rows, self.chunk_size):
            chunk_end = min(i + self.chunk_size, total_rows)
            chunks.append((i, chunk_end))
        
        print(f"Processing {len(chunks)} chunks...")
        
        # Process chunks in parallel
        processing_start = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all chunk processing tasks
            future_to_chunk = {
                executor.submit(self._process_chunk, file_path, chunk_start, chunk_end, idx, attack_detection): idx
                for idx, (chunk_start, chunk_end) in enumerate(chunks)
            }
            
            # Collect results
            chunk_results = []
            detected_attacks = []
            rows_processed = 0
            
            for future in as_completed(future_to_chunk):
                chunk_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_results.append(result)
                    rows_processed += result['rows_processed']
                    detected_attacks.extend(result['detected_attacks'])
                    
                    if chunk_idx % 10 == 0:  # Progress update every 10 chunks
                        print(f"Completed chunk {chunk_idx + 1}/{len(chunks)}")
                        
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx}: {e}")
        
        processing_time = time.time() - processing_start
        total_time = time.time() - start_time
        throughput = rows_processed / total_time if total_time > 0 else 0
        
        # Prepare results
        results = {
            'file_name': Path(file_path).name,
            'processor_type': 'DirectFileProcessor',
            'num_threads': self.num_threads,
            'chunk_size': self.chunk_size,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'records_processed': rows_processed,
            'processing_time': processing_time,
            'total_time': total_time,
            'throughput': throughput,
            'detected_attacks': len(detected_attacks),
            'chunks_processed': len(chunk_results),
            'attack_details': detected_attacks[:100] if attack_detection else []  # Limit to first 100
        }
        
        print(f"\nProcessing completed:")
        print(f"Records processed: {rows_processed:,}")
        print(f"Processing time: {processing_time:.3f} seconds")
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Throughput: {throughput:,.2f} records/second")
        print(f"Attacks detected: {len(detected_attacks)}")
        
        return results
    
    def _process_chunk(self, file_path: str, chunk_start: int, chunk_end: int, 
                      thread_id: int, attack_detection: bool) -> Dict[str, Any]:
        """
        Process a single chunk of the file
        
        Args:
            file_path: Path to CSV file
            chunk_start: Starting row index
            chunk_end: Ending row index
            thread_id: Thread identifier
            attack_detection: Whether to perform attack detection
            
        Returns:
            Chunk processing results
        """
        try:
            # Calculate rows to skip and read
            skiprows = list(range(1, chunk_start + 1)) if chunk_start > 0 else None
            nrows = chunk_end - chunk_start
            
            # Read chunk
            chunk_df = pd.read_csv(file_path, skiprows=skiprows, nrows=nrows, low_memory=False)
            
            # Process the chunk
            processed_chunk = self._process_data(chunk_df, attack_detection)
            
            # Detect attacks if enabled
            detected_attacks = []
            if attack_detection and 'delay' in chunk_df.columns:
                attack_rows = chunk_df[chunk_df['delay'] != 'normal']
                for idx, row in attack_rows.iterrows():
                    detected_attacks.append({
                        'row_index': chunk_start + idx,
                        'attack_type': row['delay'],
                        'thread_id': thread_id
                    })
            
            return {
                'thread_id': thread_id,
                'chunk_start': chunk_start,
                'chunk_end': chunk_end,
                'rows_processed': len(chunk_df),
                'detected_attacks': detected_attacks,
                'processing_successful': True
            }
            
        except Exception as e:
            return {
                'thread_id': thread_id,
                'chunk_start': chunk_start,
                'chunk_end': chunk_end,
                'rows_processed': 0,
                'detected_attacks': [],
                'processing_successful': False,
                'error': str(e)
            }
    
    def _process_data(self, df: pd.DataFrame, attack_detection: bool) -> pd.DataFrame:
        """
        Process data chunk with optimized operations
        
        Args:
            df: Input DataFrame chunk
            attack_detection: Whether to perform attack detection
            
        Returns:
            Processed DataFrame
        """
        # Get numeric columns for processing (limit to first 10 for performance)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:10]
        
        # Calculate differences for numeric columns
        for col in numeric_cols:
            try:
                df[f'{col}_diff'] = df[col].diff().fillna(0)
            except Exception:
                continue
        
        # Perform attack detection if enabled
        if attack_detection:
            # Calculate z-scores for anomaly detection
            for col in numeric_cols:
                try:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    
                    if std_val > 0:
                        df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
                    else:
                        df[f'{col}_zscore'] = 0
                except Exception:
                    continue
            
            # Focus on key features identified in data understanding
            key_features = ['StNum', 'sqDiff', 'timeFromLastChange']
            
            for feature in key_features:
                if feature in df.columns:
                    try:
                        # Flag anomalies where z-score > 3 standard deviations
                        z_col = f'{feature}_zscore'
                        if z_col in df.columns:
                            df[f'{feature}_anomaly'] = np.abs(df[z_col]) > 3
                    except Exception:
                        continue
        
        return df
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save processing results to JSON file
        
        Args:
            results: Results dictionary
            output_path: Output file path
        """
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            print(f"Error saving results: {e}")


def test_processor(file_path: str, num_threads: int = 4, chunk_size: int = 10000):
    """
    Test the DirectFileProcessor with given parameters
    
    Args:
        file_path: Path to test file
        num_threads: Number of threads to use
        chunk_size: Chunk size for processing
        
    Returns:
        Processing results
    """
    print(f"Testing DirectFileProcessor")
    print(f"File: {file_path}")
    print(f"Threads: {num_threads}")
    print(f"Chunk size: {chunk_size}")
    print("-" * 50)
    
    # Create processor
    processor = DirectFileProcessor(num_threads=num_threads, chunk_size=chunk_size)
    
    # Process file
    results = processor.process_file(file_path, attack_detection=True)
    
    # Save results
    output_path = f"direct_file_results_{num_threads}t_{chunk_size}c.json"
    processor.save_results(results, output_path)
    
    return results


def main():
    """Main function for testing DirectFileProcessor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Direct File Processor')
    parser.add_argument('--file', default='data/sample/balanced_sample.csv', help='Input file path')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--chunk-size', type=int, default=3000, help='Chunk size')
    
    args = parser.parse_args()
    
    # Test with provided parameters
    results = test_processor(args.file, args.threads, args.chunk_size)
    
    print("\n" + "=" * 60)
    print("DIRECT FILE PROCESSOR TEST COMPLETED")
    print("=" * 60)


if __name__ == '__main__':
    main()
