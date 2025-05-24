import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataVisualization:
    """
    Comprehensive visualization utilities for brain tumor dataset
    """
    
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    def plot_class_distribution(self, overview: Dict, save_path: str = None):
        """
        Plot class distribution for training and testing sets
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training distribution
        train_counts = [overview['training'][class_name]['count'] 
                       for class_name in Config.CLASS_NAMES]
        ax1.pie(train_counts, labels=Config.CLASS_NAMES, autopct='%1.1f%%', 
                colors=self.colors)
        ax1.set_title('Training Set Distribution', fontsize=14, fontweight='bold')
        
        # Testing distribution
        test_counts = [overview['testing'][class_name]['count'] 
                      for class_name in Config.CLASS_NAMES]
        ax2.pie(test_counts, labels=Config.CLASS_NAMES, autopct='%1.1f%%', 
                colors=self.colors)
        ax2.set_title('Testing Set Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_sample_images(self, samples: Dict, save_path: str = None):
        """
        Plot sample images from each class
        """
        n_classes = len(Config.CLASS_NAMES)
        n_samples = len(samples[Config.CLASS_NAMES[0]])
        
        fig, axes = plt.subplots(n_classes, n_samples, figsize=(20, 16))
        
        for i, class_name in enumerate(Config.CLASS_NAMES):
            for j, sample in enumerate(samples[class_name]):
                ax = axes[i, j] if n_classes > 1 else axes[j]
                ax.imshow(sample['image'])
                ax.set_title(f"{class_name}\n{sample['filename']}", fontsize=10)
                ax.axis('off')
        
        plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_image_properties_analysis(self, df: pd.DataFrame, save_path: str = None):
        """
        Plot comprehensive image properties analysis
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Image dimensions distribution
        axes[0, 0].hist([df[df['class'] == cls]['width'] for cls in Config.CLASS_NAMES], 
                       label=Config.CLASS_NAMES, alpha=0.7, bins=20)
        axes[0, 0].set_title('Width Distribution')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].legend()
        
        axes[0, 1].hist([df[df['class'] == cls]['height'] for cls in Config.CLASS_NAMES], 
                       label=Config.CLASS_NAMES, alpha=0.7, bins=20)
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].legend()
        
        # File size distribution
        sns.boxplot(data=df, x='class', y='file_size_kb', ax=axes[0, 2])
        axes[0, 2].set_title('File Size Distribution by Class')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Intensity statistics
        sns.boxplot(data=df, x='class', y='mean_intensity', ax=axes[1, 0])
        axes[1, 0].set_title('Mean Intensity by Class')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        sns.boxplot(data=df, x='class', y='std_intensity', ax=axes[1, 1])
        axes[1, 1].set_title('Intensity Standard Deviation by Class')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Aspect ratio
        sns.boxplot(data=df, x='class', y='aspect_ratio', ax=axes[1, 2])
        axes[1, 2].set_title('Aspect Ratio by Class')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame):
        """
        Create interactive Plotly dashboard for data exploration
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Class Distribution', 'File Size vs Mean Intensity', 
                          'Image Dimensions', 'Intensity Distribution'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # Class distribution pie chart
        class_counts = df['class'].value_counts()
        fig.add_trace(
            go.Pie(labels=class_counts.index, values=class_counts.values,
                   name="Class Distribution"),
            row=1, col=1
        )
        
        # File size vs mean intensity scatter
        fig.add_trace(
            go.Scatter(x=df['file_size_kb'], y=df['mean_intensity'],
                      mode='markers', color=df['class'],
                      text=df['filename'], name="Size vs Intensity"),
            row=1, col=2
        )
        
        # Image dimensions scatter
        fig.add_trace(
            go.Scatter(x=df['width'], y=df['height'],
                      mode='markers', color=df['class'],
                      text=df['filename'], name="Dimensions"),
            row=2, col=1
        )
        
        # Intensity histogram
        for class_name in Config.CLASS_NAMES:
            class_data = df[df['class'] == class_name]
            fig.add_trace(
                go.Histogram(x=class_data['mean_intensity'], 
                           name=f"{class_name} Intensity",
                           opacity=0.7),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Brain Tumor Dataset Analysis Dashboard")
        return fig