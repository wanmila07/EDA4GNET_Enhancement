"""
Java-Python bridge module for EDA4GNET Framework Enhancement
"""

import time
import pandas as pd
from py4j.java_gateway import JavaGateway, GatewayParameters
from typing import List, Dict, Any, Optional

class JavaBridge:
    def __init__(self, gateway_port: int = 25333):
        """
        Initialize Java Bridge
        
        Args:
            gateway_port: Port number for Java Gateway (default: 25333)
        """
        self.port = gateway_port
        self.gateway = None
        self.java_gateway = None
        self.is_connected = False
    
    def connect(self) -> bool:
        """
        Connect to Java Gateway
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.gateway = JavaGateway(
                gateway_parameters=GatewayParameters(port=self.port)
            )
            self.java_gateway = self.gateway.entry_point
            
            # Test connection
            test_result = self.java_gateway.testConnection()
            print(f"Java Gateway connection test: {test_result}")
            
            self.is_connected = True
            return True
            
        except Exception as e:
            print(f"Failed to connect to Java Gateway on port {self.port}: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Java Gateway"""
        if self.gateway:
            try:
                self.gateway.shutdown()
                print("Java Gateway disconnected")
            except Exception as e:
                print(f"Error disconnecting from gateway: {e}")
        
        self.gateway = None
        self.java_gateway = None
        self.is_connected = False
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get Java system information
        
        Returns:
            Dict containing system information
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Java Gateway")
        
        return dict(self.java_gateway.getSystemInfo())
    
    def convert_dataframe_to_java_list(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Convert pandas DataFrame to Java-compatible list of maps
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of dictionaries compatible with Java
        """
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Convert numpy types to Python native types for Java compatibility
        java_records = []
        for record in records:
            java_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    java_record[key] = None
                elif isinstance(value, (int, float, str, bool)):
                    java_record[key] = value
                else:
                    # Convert numpy types to Python native types
                    java_record[key] = value.item() if hasattr(value, 'item') else str(value)
            java_records.append(java_record)
        
        return java_records
    
    def process_sequential(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Process DataFrame using sequential implementation
        
        Args:
            df: Input DataFrame
            
        Returns:
            Processing results dictionary
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Java Gateway")
        
        # Convert DataFrame to Java format
        java_data = self.convert_dataframe_to_java_list(df)
        
        # Process using Java sequential implementation
        results = self.java_gateway.processSequential(java_data)
        
        return dict(results)
    
    def process_threaded(self, df: pd.DataFrame, num_threads: int = 4) -> Dict[str, Any]:
        """
        Process DataFrame using threaded implementation
        
        Args:
            df: Input DataFrame
            num_threads: Number of threads to use
            
        Returns:
            Processing results dictionary
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Java Gateway")
        
        # Convert DataFrame to Java format
        java_data = self.convert_dataframe_to_java_list(df)
        
        # Process using Java threaded implementation
        results = self.java_gateway.processThreaded(java_data, num_threads)
        
        return dict(results)
    
    def process_multiprocess(self, df: pd.DataFrame, num_processes: int = 4) -> Dict[str, Any]:
        """
        Process DataFrame using multiprocessing implementation
        
        Args:
            df: Input DataFrame
            num_processes: Number of processes to use
            
        Returns:
            Processing results dictionary
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Java Gateway")
        
        # Convert DataFrame to Java format
        java_data = self.convert_dataframe_to_java_list(df)
        
        # Process using Java multiprocessing implementation
        results = self.java_gateway.processMultiprocess(java_data, num_processes)
        
        return dict(results)
    
    def process_hybrid(self, df: pd.DataFrame, num_processes: int = 4, num_threads: int = 8) -> Dict[str, Any]:
        """
        Process DataFrame using hybrid implementation
        
        Args:
            df: Input DataFrame
            num_processes: Number of processes to use
            num_threads: Number of threads to use
            
        Returns:
            Processing results dictionary
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Java Gateway")
        
        # Convert DataFrame to Java format
        java_data = self.convert_dataframe_to_java_list(df)
        
        # Process using Java hybrid implementation
        results = self.java_gateway.processHybrid(java_data, num_processes, num_threads)
        
        return dict(results)
    
    def process_file_direct(self, file_path: str, num_threads: int = 4) -> Dict[str, Any]:
        """
        Process file directly using optimized Java implementation
        
        Args:
            file_path: Path to CSV file
            num_threads: Number of threads to use
            
        Returns:
            Processing results dictionary
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Java Gateway")
        
        # Process file directly in Java
        results = self.java_gateway.processFileDirect(file_path, num_threads)
        
        return dict(results)
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()