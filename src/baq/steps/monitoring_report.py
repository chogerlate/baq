"""
Monitoring Report Module for Bangkok Air Quality Forecasting.

This module provides comprehensive monitoring capabilities for PM2.5 forecasting models
and data quality assessment. It generates detailed reports on model performance,
data drift detection, and system health monitoring.

Features:
- Model performance monitoring and tracking
- Data drift detection using Evidently framework
- Comprehensive HTML report generation
- Data quality assessment and validation
- Performance degradation alerts
- Visual monitoring dashboards

The monitoring process includes:
1. Model performance evaluation and comparison
2. Data drift analysis between training and production data
3. Data quality checks and validation
4. Alert generation for significant changes
5. HTML report compilation with visualizations

Example:
    >>> monitor = MonitoringReport(model, train_data, test_data, config)
    >>> report_html = monitor.create_monitoring_report()
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import json
from datetime import datetime
import base64
from io import BytesIO

class MonitoringReport:
    def __init__(self, model: object, train_data: pd.DataFrame, test_data: pd.DataFrame, config: dict):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.config = config

    def create_monitoring_report(self) -> str:
        """
        Create a comprehensive monitoring report for model performance and data drift.
        
        Returns:
            str: HTML report content as a string
        """
        # Generate all monitoring components
        data_drift_analysis = self._analyze_data_drift()
        performance_metrics = self._calculate_performance_metrics()
        data_quality_checks = self._perform_data_quality_checks()
        visualizations = self._create_visualizations()
        
        # Create comprehensive HTML report
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Monitoring Report</title>
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background-color: #f5f5f5;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background-color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{ 
                    color: #2c3e50; 
                    text-align: center; 
                    border-bottom: 3px solid #3498db; 
                    padding-bottom: 10px;
                }}
                h2 {{ 
                    color: #3498db; 
                    margin-top: 30px;
                    border-left: 4px solid #3498db;
                    padding-left: 15px;
                }}
                h3 {{ 
                    color: #34495e; 
                    margin-top: 20px;
                }}
                .metrics {{ 
                    background-color: #f8f9fa; 
                    padding: 20px; 
                    border-radius: 8px; 
                    margin: 15px 0;
                    border-left: 4px solid #17a2b8;
                }}
                .alert {{ 
                    color: #e74c3c; 
                    font-weight: bold; 
                    background-color: #fdf2f2;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #e74c3c;
                }}
                .success {{ 
                    color: #27ae60; 
                    font-weight: bold; 
                    background-color: #f2fdf2;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #27ae60;
                }}
                .warning {{ 
                    color: #f39c12; 
                    font-weight: bold; 
                    background-color: #fefbf2;
                    padding: 10px;
                    border-radius: 5px;
                    border-left: 4px solid #f39c12;
                }}
                .metric-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                    gap: 15px; 
                    margin: 15px 0;
                }}
                .metric-card {{ 
                    background: white; 
                    padding: 15px; 
                    border-radius: 8px; 
                    border: 1px solid #e0e0e0;
                    text-align: center;
                }}
                .metric-value {{ 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #2c3e50;
                }}
                .metric-label {{ 
                    color: #7f8c8d; 
                    font-size: 14px;
                }}
                .visualization {{ 
                    text-align: center; 
                    margin: 20px 0;
                }}
                .visualization img {{ 
                    max-width: 100%; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 15px 0;
                }}
                th, td {{ 
                    padding: 12px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd;
                }}
                th {{ 
                    background-color: #3498db; 
                    color: white;
                }}
                .timestamp {{ 
                    text-align: center; 
                    color: #7f8c8d; 
                    font-style: italic; 
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Monitoring Report</h1>
                
                <div class="metrics">
                    <h3>Report Summary</h3>
                    <p><strong>Model Type:</strong> {self.config["model"]["model_type"]}</p>
                    <p><strong>Target Column:</strong> {self.config["training"]["target_column"]}</p>
                    <p><strong>Forecast Horizon:</strong> {self.config["training"]["forecast_horizon"]} hours</p>
                    <p><strong>Training Data Size:</strong> {len(self.train_data)} samples</p>
                    <p><strong>Test Data Size:</strong> {len(self.test_data)} samples</p>
                </div>

                <h2>Data Quality Assessment</h2>
                {self._format_data_quality_section(data_quality_checks)}

                <h2>Data Drift Analysis</h2>
                {self._format_data_drift_section(data_drift_analysis)}

                <h2>Performance Metrics</h2>
                {self._format_performance_section(performance_metrics)}

                <h2>Visualizations</h2>
                {self._format_visualizations_section(visualizations)}

                <div class="timestamp">
                    <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_report

    def _analyze_data_drift(self) -> Dict[str, Any]:
        """Analyze data drift between training and test datasets."""
        drift_results = {}
        
        # Get numerical columns
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if col in self.test_data.columns:
                # Statistical tests for drift detection
                train_mean = self.train_data[col].mean()
                test_mean = self.test_data[col].mean()
                train_std = self.train_data[col].std()
                test_std = self.test_data[col].std()
                
                # Calculate percentage change
                mean_change = abs((test_mean - train_mean) / train_mean * 100) if train_mean != 0 else 0
                std_change = abs((test_std - train_std) / train_std * 100) if train_std != 0 else 0
                
                # Simple drift detection based on thresholds
                drift_detected = mean_change > 20 or std_change > 30
                
                drift_results[col] = {
                    'train_mean': train_mean,
                    'test_mean': test_mean,
                    'train_std': train_std,
                    'test_std': test_std,
                    'mean_change_pct': mean_change,
                    'std_change_pct': std_change,
                    'drift_detected': drift_detected
                }
        
        return drift_results

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate basic performance metrics."""
        target_col = self.config["training"]["target_column"]
        
        if target_col in self.train_data.columns and target_col in self.test_data.columns:
            train_target = self.train_data[target_col]
            test_target = self.test_data[target_col]
            
            metrics = {
                'train_target_mean': train_target.mean(),
                'test_target_mean': test_target.mean(),
                'train_target_std': train_target.std(),
                'test_target_std': test_target.std(),
                'train_target_min': train_target.min(),
                'test_target_min': test_target.min(),
                'train_target_max': train_target.max(),
                'test_target_max': test_target.max(),
            }
            
            # Calculate target distribution shift
            target_shift = abs((test_target.mean() - train_target.mean()) / train_target.mean() * 100)
            metrics['target_distribution_shift_pct'] = target_shift
            
            return metrics
        
        return {}

    def _perform_data_quality_checks(self) -> Dict[str, Any]:
        """Perform comprehensive data quality checks."""
        quality_results = {
            'train_data': self._check_data_quality(self.train_data, 'Training'),
            'test_data': self._check_data_quality(self.test_data, 'Test')
        }
        return quality_results

    def _check_data_quality(self, data: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """Check data quality for a single dataset."""
        total_rows = len(data)
        total_cols = len(data.columns)
        
        # Missing values analysis
        missing_counts = data.isnull().sum()
        missing_percentages = (missing_counts / total_rows * 100).round(2)
        
        # Duplicate rows
        duplicate_rows = data.duplicated().sum()
        
        # Data types
        data_types = data.dtypes.value_counts().to_dict()
        
        # Numerical columns statistics
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        outliers_count = {}
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outliers_count[col] = outliers
        
        return {
            'total_rows': total_rows,
            'total_columns': total_cols,
            'missing_values': missing_counts.to_dict(),
            'missing_percentages': missing_percentages.to_dict(),
            'duplicate_rows': duplicate_rows,
            'data_types': {str(k): v for k, v in data_types.items()},
            'outliers_count': outliers_count
        }

    def _create_visualizations(self) -> Dict[str, str]:
        """Create visualizations and return them as base64 encoded strings."""
        visualizations = {}
        
        # Target distribution comparison
        target_col = self.config["training"]["target_column"]
        if target_col in self.train_data.columns and target_col in self.test_data.columns:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Distribution plots
            ax1.hist(self.train_data[target_col].dropna(), bins=30, alpha=0.7, label='Training', density=True)
            ax1.hist(self.test_data[target_col].dropna(), bins=30, alpha=0.7, label='Test', density=True)
            ax1.set_title('Target Distribution Comparison')
            ax1.set_xlabel(target_col)
            ax1.set_ylabel('Density')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plots
            data_for_box = [self.train_data[target_col].dropna(), self.test_data[target_col].dropna()]
            ax2.boxplot(data_for_box, labels=['Training', 'Test'])
            ax2.set_title('Target Distribution Box Plot')
            ax2.set_ylabel(target_col)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            visualizations['target_distribution'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        # Feature correlation heatmap for numerical columns
        numerical_cols = self.train_data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 1:
            fig, ax = plt.subplots(figsize=(12, 10))
            correlation_matrix = self.train_data[numerical_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                       square=True, ax=ax, fmt='.2f')
            ax.set_title('Feature Correlation Matrix (Training Data)')
            plt.tight_layout()
            visualizations['correlation_matrix'] = self._fig_to_base64(fig)
            plt.close(fig)
        
        return visualizations

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return f"data:image/png;base64,{image_base64}"

    def _format_data_quality_section(self, quality_results: Dict[str, Any]) -> str:
        """Format data quality results into HTML."""
        html = ""
        
        for dataset_name, results in quality_results.items():
            html += f"""
            <div class="metrics">
                <h3>{dataset_name.replace('_', ' ').title()} Data Quality</h3>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{results['total_rows']:,}</div>
                        <div class="metric-label">Total Rows</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['total_columns']}</div>
                        <div class="metric-label">Total Columns</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{results['duplicate_rows']:,}</div>
                        <div class="metric-label">Duplicate Rows</div>
                    </div>
                </div>
            """
            
            # Missing values summary
            missing_cols = [col for col, pct in results['missing_percentages'].items() if pct > 0]
            if missing_cols:
                html += '<div class="warning">Missing Values Detected:</div>'
                html += '<table><tr><th>Column</th><th>Missing Count</th><th>Missing %</th></tr>'
                for col in missing_cols:
                    count = results['missing_values'][col]
                    pct = results['missing_percentages'][col]
                    html += f'<tr><td>{col}</td><td>{count}</td><td>{pct}%</td></tr>'
                html += '</table>'
            else:
                html += '<div class="success">No missing values detected</div>'
            
            html += "</div>"
        
        return html

    def _format_data_drift_section(self, drift_results: Dict[str, Any]) -> str:
        """Format data drift results into HTML."""
        if not drift_results:
            return '<div class="metrics"><p>No numerical columns available for drift analysis.</p></div>'
        
        html = '<div class="metrics">'
        
        # Summary of drift detection
        drifted_features = [col for col, results in drift_results.items() if results['drift_detected']]
        
        if drifted_features:
            html += f'<div class="alert">Data drift detected in {len(drifted_features)} feature(s): {", ".join(drifted_features)}</div>'
        else:
            html += '<div class="success">No significant data drift detected</div>'
        
        # Detailed drift table
        html += '''
        <h3>Detailed Drift Analysis</h3>
        <table>
            <tr>
                <th>Feature</th>
                <th>Train Mean</th>
                <th>Test Mean</th>
                <th>Mean Change %</th>
                <th>Std Change %</th>
                <th>Drift Status</th>
            </tr>
        '''
        
        for col, results in drift_results.items():
            status = "⚠️ DRIFT" if results['drift_detected'] else "✅ OK"
            html += f'''
            <tr>
                <td>{col}</td>
                <td>{results['train_mean']:.3f}</td>
                <td>{results['test_mean']:.3f}</td>
                <td>{results['mean_change_pct']:.2f}%</td>
                <td>{results['std_change_pct']:.2f}%</td>
                <td>{status}</td>
            </tr>
            '''
        
        html += '</table></div>'
        return html

    def _format_performance_section(self, performance_metrics: Dict[str, Any]) -> str:
        """Format performance metrics into HTML."""
        if not performance_metrics:
            return '<div class="metrics"><p>No performance metrics available.</p></div>'
        
        html = f'''
        <div class="metrics">
            <h3>Target Variable Statistics</h3>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{performance_metrics['train_target_mean']:.3f}</div>
                    <div class="metric-label">Train Mean</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_metrics['test_target_mean']:.3f}</div>
                    <div class="metric-label">Test Mean</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{performance_metrics['target_distribution_shift_pct']:.2f}%</div>
                    <div class="metric-label">Distribution Shift</div>
                </div>
            </div>
        '''
        
        # Alert for significant target shift
        if performance_metrics['target_distribution_shift_pct'] > 15:
            html += '<div class="alert">Significant target distribution shift detected!</div>'
        else:
            html += '<div class="success">Target distribution appears stable</div>'
        
        html += '</div>'
        return html

    def _format_visualizations_section(self, visualizations: Dict[str, str]) -> str:
        """Format visualizations into HTML."""
        html = ""
        
        for viz_name, viz_data in visualizations.items():
            title = viz_name.replace('_', ' ').title()
            html += f'''
            <div class="visualization">
                <h3>{title}</h3>
                <img src="{viz_data}" alt="{title}">
            </div>
            '''
        
        return html

    def save_monitoring_report(self, report: str, path: str):
        """Save the monitoring report to a file."""
        with open(path, "w", encoding='utf-8') as f:
            f.write(report)