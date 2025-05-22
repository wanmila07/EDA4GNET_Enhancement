"""
Experiment runner for EDA4GNET Framework Enhancement
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

from src.eda4gnet.java_bridge import JavaBridge
from src.eda4gnet.data_loader import DataLoader


class ExperimentRunner:
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.data_loader = DataLoader()
        
        # Create results directories
        for impl in ['baseline', 'threading', 'multiprocessing', 'hybrid', 'comparison']:
            (self.results_dir / impl).mkdir(parents=True, exist_ok=True)
    
    def run_baseline_experiment(self, dataset_path: str, save_results: bool = True) -> Dict[str, Any]:
        """
        Run baseline (sequential) experiment
        
        Args:
            dataset_path: Path to dataset file
            save_results: Whether to save results to file
            
        Returns:
            Results dictionary
        """
        print("Running Baseline (Sequential) Experiment")
        print("=" * 50)
        
        # Load data
        df = self.data_loader.load_dataset(dataset_path)
        df = self.data_loader.preprocess_data(df)
        
        # Run experiment
        with JavaBridge() as bridge:
            start_time = time.time()
            results = bridge.process_sequential(df)
            total_time = time.time() - start_time
        
        # Add metadata
        results.update({
            'dataset_path': dataset_path,
            'dataset_size': len(df),
            'total_time': total_time,
            'experiment_type': 'baseline',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        if save_results:
            self._save_results(results, 'baseline', 'performance_summary.json')
        
        self._print_results(results)
        return results
    
    def run_threading_experiment(self, dataset_path: str, num_threads: int = 4, save_results: bool = True) -> Dict[str, Any]:
        """
        Run threading experiment
        
        Args:
            dataset_path: Path to dataset file
            num_threads: Number of threads to use
            save_results: Whether to save results to file
            
        Returns:
            Results dictionary
        """
        print("Running Threading Experiment")
        print("=" * 50)
        
        # Load data
        df = self.data_loader.load_dataset(dataset_path)
        df = self.data_loader.preprocess_data(df)
        
        # Run experiment
        with JavaBridge() as bridge:
            start_time = time.time()
            results = bridge.process_threaded(df, num_threads)
            total_time = time.time() - start_time
        
        # Add metadata
        results.update({
            'dataset_path': dataset_path,
            'dataset_size': len(df),
            'total_time': total_time,
            'num_threads': num_threads,
            'experiment_type': 'threading',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        if save_results:
            self._save_results(results, 'threading', 'performance_summary.json')
        
        self._print_results(results)
        return results
    
    def run_multiprocessing_experiment(self, dataset_path: str, num_processes: int = 4, save_results: bool = True) -> Dict[str, Any]:
        """
        Run multiprocessing experiment
        
        Args:
            dataset_path: Path to dataset file
            num_processes: Number of processes to use
            save_results: Whether to save results to file
            
        Returns:
            Results dictionary
        """
        print("Running Multiprocessing Experiment")
        print("=" * 50)
        
        # Load data
        df = self.data_loader.load_dataset(dataset_path)
        df = self.data_loader.preprocess_data(df)
        
        # Run experiment
        with JavaBridge() as bridge:
            start_time = time.time()
            results = bridge.process_multiprocess(df, num_processes)
            total_time = time.time() - start_time
        
        # Add metadata
        results.update({
            'dataset_path': dataset_path,
            'dataset_size': len(df),
            'total_time': total_time,
            'num_processes': num_processes,
            'experiment_type': 'multiprocessing',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        if save_results:
            self._save_results(results, 'multiprocessing', 'performance_summary.json')
        
        self._print_results(results)
        return results
    
    def run_hybrid_experiment(self, dataset_path: str, num_processes: int = 4, num_threads: int = 8, save_results: bool = True) -> Dict[str, Any]:
        """
        Run hybrid experiment
        
        Args:
            dataset_path: Path to dataset file
            num_processes: Number of processes to use
            num_threads: Number of threads to use
            save_results: Whether to save results to file
            
        Returns:
            Results dictionary
        """
        print("Running Hybrid Experiment")
        print("=" * 50)
        
        # Load data
        df = self.data_loader.load_dataset(dataset_path)
        df = self.data_loader.preprocess_data(df)
        
        # Run experiment
        with JavaBridge() as bridge:
            start_time = time.time()
            results = bridge.process_hybrid(df, num_processes, num_threads)
            total_time = time.time() - start_time
        
        # Add metadata
        results.update({
            'dataset_path': dataset_path,
            'dataset_size': len(df),
            'total_time': total_time,
            'num_processes': num_processes,
            'num_threads': num_threads,
            'experiment_type': 'hybrid',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        if save_results:
            self._save_results(results, 'hybrid', 'performance_summary.json')
        
        self._print_results(results)
        return results
    
    def run_direct_file_experiment(self, dataset_path: str, num_threads: int = 4, save_results: bool = True) -> Dict[str, Any]:
        """
        Run direct file access experiment
        
        Args:
            dataset_path: Path to dataset file
            num_threads: Number of threads to use
            save_results: Whether to save results to file
            
        Returns:
            Results dictionary
        """
        print("Running Direct File Access Experiment")
        print("=" * 50)
        
        # Run experiment
        with JavaBridge() as bridge:
            start_time = time.time()
            results = bridge.process_file_direct(dataset_path, num_threads)
            total_time = time.time() - start_time
        
        # Add metadata
        results.update({
            'dataset_path': dataset_path,
            'total_time': total_time,
            'num_threads': num_threads,
            'experiment_type': 'direct_file_access',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        if save_results:
            self._save_results(results, 'baseline', 'direct_file_performance.json')
        
        self._print_results(results)
        return results
    
    def run_all_experiments(self, dataset_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Run all experiments and compare results
        
        Args:
            dataset_path: Path to dataset file
            
        Returns:
            Dictionary with all experiment results
        """
        print("Running All Experiments")
        print("=" * 60)
        
        all_results = {}
        
        # Run all experiments
        all_results['baseline'] = self.run_baseline_experiment(dataset_path)
        all_results['threading'] = self.run_threading_experiment(dataset_path)
        all_results['multiprocessing'] = self.run_multiprocessing_experiment(dataset_path)
        all_results['hybrid'] = self.run_hybrid_experiment(dataset_path)
        all_results['direct_file'] = self.run_direct_file_experiment(dataset_path)
        
        # Generate comparison
        comparison = self._generate_comparison(all_results)
        self._save_results(comparison, 'comparison', 'summary_report.json')
        
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPARISON SUMMARY")
        print("=" * 60)
        self._print_comparison(comparison)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any], experiment_type: str, filename: str):
        """Save results to JSON file"""
        filepath = self.results_dir / experiment_type / filename
        
        # Convert any non-serializable objects to strings
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to: {filepath}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _print_results(self, results: Dict[str, Any]):
        """Print experiment results"""
        print(f"\nExperiment: {results.get('experiment_type', 'Unknown')}")
        print(f"Dataset: {results.get('dataset_path', 'Unknown')}")
        print(f"Records processed: {results.get('dataset_size', results.get('recordsProcessed', 'Unknown')):,}")
        print(f"Processing time: {results.get('processingTime', 0):.3f} seconds")
        print(f"Total time: {results.get('total_time', 0):.3f} seconds")
        print(f"Throughput: {results.get('throughput', 0):,.2f} records/second")
        print(f"Detections: {results.get('detectionCount', results.get('detections', 0))}")
        
        if 'numThreads' in results:
            print(f"Threads used: {results['numThreads']}")
        if 'numProcesses' in results:
            print(f"Processes used: {results['numProcesses']}")
    
    def _generate_comparison(self, all_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparison of all experiment results"""
        comparison = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'experiments': {},
            'ranking': {}
        }
        
        # Extract key metrics for each experiment
        for exp_name, results in all_results.items():
            comparison['experiments'][exp_name] = {
                'throughput': results.get('throughput', 0),
                'processing_time': results.get('processingTime', 0),
                'total_time': results.get('total_time', 0),
                'detections': results.get('detectionCount', results.get('detections', 0))
            }
        
        # Rank by throughput
        throughput_ranking = sorted(
            comparison['experiments'].items(),
            key=lambda x: x[1]['throughput'],
            reverse=True
        )
        comparison['ranking']['by_throughput'] = [name for name, _ in throughput_ranking]
        
        return comparison
    
    def _print_comparison(self, comparison: Dict[str, Any]):
        """Print comparison results"""
        print("\nThroughput Ranking:")
        for i, exp_name in enumerate(comparison['ranking']['by_throughput'], 1):
            metrics = comparison['experiments'][exp_name]
            print(f"{i}. {exp_name}: {metrics['throughput']:,.2f} records/second")


def main():
    parser = argparse.ArgumentParser(description='Run EDA4GNET experiments')
    parser.add_argument('--experiment', choices=['baseline', 'threading', 'multiprocessing', 'hybrid', 'direct_file', 'all'], 
                       default='all', help='Experiment type to run')
    parser.add_argument('--dataset', default='data/sample/balanced_sample.csv', help='Dataset file path')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    parser.add_argument('--processes', type=int, default=4, help='Number of processes')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.experiment == 'baseline':
        runner.run_baseline_experiment(args.dataset)
    elif args.experiment == 'threading':
        runner.run_threading_experiment(args.dataset, args.threads)
    elif args.experiment == 'multiprocessing':
        runner.run_multiprocessing_experiment(args.dataset, args.processes)
    elif args.experiment == 'hybrid':
        runner.run_hybrid_experiment(args.dataset, args.processes, args.threads)
    elif args.experiment == 'direct_file':
        runner.run_direct_file_experiment(args.dataset, args.threads)
    elif args.experiment == 'all':
        runner.run_all_experiments(args.dataset)


if __name__ == '__main__':
    main()
    