
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from pathlib import Path
import warnings
import glob
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("-- Starting Comprehensive Analysis Workflow --")
print("=" * 50)

# ============================================================================
# CLEANUP: DELETE EXISTING OUTPUT FILES
# ============================================================================

print("\n[STEP] Cleaning up existing output files...")

# List of files to delete before starting
files_to_cleanup = [
    'analysis_dashboard.png',
    'analysis_dashboard.jpg',
    'analysis_dashboard.jpeg',
    'analysis_results.csv',
    'analysis_summary.txt',
    'temporal_analysis.csv',
    'correlation_matrix.csv',
    'performance_rankings.csv'
]

# Delete existing files
for file_path in files_to_cleanup:
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"- Deleted existing file: {file_path}")
        except Exception as e:
            print(f"- Could not delete {file_path}: {e}")

# Clean up any temporary analysis files
temp_files = glob.glob('temp_analysis_*.csv') + glob.glob('temp_analysis_*.png')
for temp_file in temp_files:
    try:
        os.remove(temp_file)
        print(f"- Deleted temporary file: {temp_file}")
    except Exception as e:
        print(f"- Could not delete {temp_file}: {e}")

# Clean up analysis directories if they exist
analysis_dirs = ['analysis_results', 'temp_analysis', 'output']
for dir_path in analysis_dirs:
    if os.path.exists(dir_path):
        try:
            import shutil
            shutil.rmtree(dir_path)
            print(f"- Deleted existing directory: {dir_path}")
        except Exception as e:
            print(f"- Could not delete directory {dir_path}: {e}")

# Clear any existing matplotlib figures to avoid memory issues
plt.close('all')

print("- Cleanup completed")

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n[STEP] Loading and preparing data...")

# Load File 2 (KPIs data)
try:
    file2_data = pd.read_csv("Data/Interview/Folder 2/File 2.csv")
    print(f"- File 2 loaded: {file2_data.shape[0]} rows, {file2_data.shape[1]} columns")
    
    # Clean column names
    file2_data.columns = file2_data.columns.str.strip()
    print(f"- File 2 columns: {list(file2_data.columns)}")
    
except Exception as e:
    print(f"[ERROR] Error loading File 2: {e}")
    file2_data = None

# Load File 3 (Excel file - convert to CSV if needed)
try:
    # Try to read Excel directly
    file3_data = pd.read_excel("Data/Interview/Folder 3/File3.xlsx")
    print(f"-  File 3 loaded: {file3_data.shape[0]} rows, {file3_data.shape[1]} columns")
    print(f"-  File 3 columns: {list(file3_data.columns)}")
    
except Exception as e:
    print(f"[ERROR] Error loading File 3: {e}")
    file3_data = None

# Load fused respondents data
try:
    # Load the clean fused data
    fused_data = pd.read_csv("fused_respondents_clean.csv")
    print(f"- Fused respondents loaded: {fused_data.shape[0]} rows, {fused_data.shape[1]} columns")
    
except Exception as e:
    print(f"[ERROR] Error loading fused data: {e}")
    # Try to load individual files
    try:
        fused_files = list(Path("fused_respondents").glob("*.csv"))
        if fused_files:
            fused_data = pd.read_csv(fused_files[0])
            print(f"- Loaded first fused file: {fused_files[0].name}")
        else:
            fused_data = None
    except Exception as e2:
        print(f"[ERROR] Error loading individual fused files: {e2}")
        fused_data = None

# ============================================================================
# STEP 2: DATA CLEANING AND PREPARATION
# ============================================================================

print("\n[STEP] Data cleaning and preparation...")

if file2_data is not None:
    # Clean File 2 data
    file2_data = file2_data.dropna(how='all')
    
    # Extract stimulus categories
    def categorize_stimulus(stimulus_name):
        if 'INP_' in stimulus_name:
            return 'Advertisement'
        elif 'PURCHINT_' in stimulus_name:
            return 'Purchase Intent'
        elif 'PURCHASE' in stimulus_name:
            return 'Purchase'
        elif 'PREFERENCE' in stimulus_name:
            return 'Preference'
        elif 'UNAIDED' in stimulus_name:
            return 'Unaided Recall'
        elif 'Stickiness' in stimulus_name:
            return 'Brand Stickiness'
        else:
            return 'Other'
    
    file2_data['category'] = file2_data['SourceStimuliName'].apply(categorize_stimulus)
    
    # Clean numeric columns
    numeric_columns = ['Engagement', 'Emotion', 'CognitiveLoad', 'Desire', 'Memorisation', 'Impact']
    for col in numeric_columns:
        if col in file2_data.columns:
            file2_data[col] = pd.to_numeric(file2_data[col], errors='coerce')
    
    print(f"- File 2 cleaned and categorized")
    print(f"- Categories found: {file2_data['category'].value_counts().to_dict()}")

if file3_data is not None:
    # Clean File 3 data
    file3_data = file3_data.dropna(how='all')
    print(f"- File 3 cleaned")

# ============================================================================
# STEP 3: DYNAMIC STATISTICAL ANALYSIS BY STIMULUS AND CATEGORY
# ============================================================================

print("\n[STEP] Dynamic statistical analysis...")

def analyze_by_category(df, metric_col):
    """Analyze metrics by category with statistical tests"""
    if metric_col not in df.columns:
        return None
    
    results = {}
    
    # Basic statistics by category
    category_stats = df.groupby('category')[metric_col].agg([
        'count', 'mean', 'std', 'min', 'max', 'median'
    ]).round(3)
    
    # ANOVA test between categories
    categories = df['category'].unique()
    if len(categories) > 1:
        category_groups = [df[df['category'] == cat][metric_col].dropna() for cat in categories]
        category_groups = [group for group in category_groups if len(group) > 0]
        
        if len(category_groups) > 1:
            f_stat, p_value = stats.f_oneway(*category_groups)
            results['anova'] = {'f_statistic': f_stat, 'p_value': p_value}
    
    results['category_stats'] = category_stats
    return results

if file2_data is not None:
    # Analyze key metrics
    metrics_to_analyze = ['Engagement', 'Emotion', 'CognitiveLoad', 'Desire', 'Memorisation', 'Impact']
    
    print("\n- Statistical Analysis by Category:")
    for metric in metrics_to_analyze:
        if metric in file2_data.columns:
            print(f"\n🔍 {metric} Analysis:")
            analysis = analyze_by_category(file2_data, metric)
            if analysis:
                print(analysis['category_stats'])
                if 'anova' in analysis:
                    print(f"   ANOVA p-value: {analysis['anova']['p_value']:.4f}")

# ============================================================================
# STEP 4: COMPLETE REPORTS FOR EACH STIMULUS AND CATEGORY
# ============================================================================

print("\n[STEP] Generating complete reports...")

def generate_stimulus_report(df, stimulus_id):
    """Generate detailed report for a specific stimulus"""
    stim_data = df[df['SourceStimuliName'] == stimulus_id]
    if stim_data.empty:
        return None
    
    report = {
        'stimulus_id': stimulus_id,
        'category': stim_data['category'].iloc[0] if 'category' in stim_data.columns else 'Unknown',
        'metrics': {}
    }
    
    # Calculate metrics
    metric_cols = ['Engagement', 'Emotion', 'CognitiveLoad', 'Desire', 'Memorisation', 'Impact']
    for col in metric_cols:
        if col in stim_data.columns:
            report['metrics'][col] = {
                'value': stim_data[col].iloc[0],
                'percentile': df[col].rank(pct=True)[stim_data.index[0]] * 100
            }
    
    return report

def generate_category_report(df, category):
    """Generate detailed report for a category"""
    cat_data = df[df['category'] == category]
    if cat_data.empty:
        return None
    
    report = {
        'category': category,
        'stimulus_count': len(cat_data),
        'metrics_summary': {},
        'top_performers': []
    }
    
    # Calculate summary statistics
    metric_cols = ['Engagement', 'Emotion', 'CognitiveLoad', 'Desire', 'Memorisation', 'Impact']
    for col in metric_cols:
        if col in cat_data.columns:
            report['metrics_summary'][col] = {
                'mean': cat_data[col].mean(),
                'std': cat_data[col].std(),
                'min': cat_data[col].min(),
                'max': cat_data[col].max(),
                'median': cat_data[col].median()
            }
    
    # Find top performers
    for col in metric_cols:
        if col in cat_data.columns:
            top_stim = cat_data.loc[cat_data[col].idxmax(), 'SourceStimuliName']
            report['top_performers'].append({
                'metric': col,
                'stimulus': top_stim,
                'value': cat_data[col].max()
            })
    
    return report

if file2_data is not None:
    # Generate reports for each stimulus
    print("\n- Individual Stimulus Reports:")
    unique_stimuli = file2_data['SourceStimuliName'].unique()
    
    for stim in unique_stimuli[:5]:  # Show first 5 for brevity
        report = generate_stimulus_report(file2_data, stim)
        if report:
            print(f"\n- {report['stimulus_id']} ({report['category']}):")
            for metric, data in report['metrics'].items():
                print(f"   {metric}: {data['value']:.2f} (Percentile: {data['percentile']:.1f}%)")
    
    # Generate reports for each category
    print("\n- Category Reports:")
    unique_categories = file2_data['category'].unique()
    
    for cat in unique_categories:
        report = generate_category_report(file2_data, cat)
        if report:
            print(f"\n🏷️  {report['category']} ({report['stimulus_count']} stimuli):")
            for metric, stats in report['metrics_summary'].items():
                print(f"   {metric}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")

# ============================================================================
# STEP 5: COMPARATIVE AND DETAILED ANALYSIS OF MULTIPLE ADS
# ============================================================================

print("\n[STEP] Comparative analysis of advertisements...")

def compare_advertisements(df):
    """Compare advertisements across multiple dimensions"""
    ad_data = df[df['category'] == 'Advertisement'].copy()
    
    if ad_data.empty:
        return None
    
    comparison = {
        'total_ads': len(ad_data),
        'performance_ranking': {},
        'correlation_analysis': {},
        'clustering_insights': {}
    }
    
    # Performance ranking
    metrics = ['Engagement', 'Emotion', 'CognitiveLoad', 'Desire', 'Memorisation', 'Impact']
    for metric in metrics:
        if metric in ad_data.columns:
            ranking = ad_data.nlargest(3, metric)[['SourceStimuliName', metric]]
            comparison['performance_ranking'][metric] = ranking.to_dict('records')
    
    # Correlation analysis
    if len(metrics) > 1:
        corr_matrix = ad_data[metrics].corr()
        comparison['correlation_analysis'] = corr_matrix.round(3).to_dict()
    
    return comparison

if file2_data is not None:
    ad_comparison = compare_advertisements(file2_data)
    if ad_comparison:
        print(f"\n- Advertisement Analysis ({ad_comparison['total_ads']} ads):")
        
        # Show top performers
        print("\n- Top Performers by Metric:")
        for metric, ranking in ad_comparison['performance_ranking'].items():
            print(f"\n   {metric}:")
            for i, item in enumerate(ranking, 1):
                print(f"     {i}. {item['SourceStimuliName']}: {item[metric]:.2f}")

# ============================================================================
# STEP 6: TEMPORAL PATTERNS AND TIME SERIES ANALYSIS
# ============================================================================

print("\n[STEP] Temporal patterns analysis...")

def analyze_temporal_patterns(fused_data):
    """Analyze temporal patterns in the fused data"""
    if fused_data is None:
        return None
    
    temporal_analysis = {
        'data_points': len(fused_data),
        'time_range': None,
        'temporal_features': {},
        'patterns': {}
    }
    
    # Check for temporal columns
    time_columns = [col for col in fused_data.columns if any(word in col.lower() 
                   for word in ['time', 'timestamp', 'duration', 'second'])]
    
    if time_columns:
        print(f"- Found temporal columns: {time_columns}")
        
        # Basic temporal analysis
        for col in time_columns[:3]:  # Analyze first 3 temporal columns
            if fused_data[col].dtype in ['int64', 'float64']:
                temporal_analysis['temporal_features'][col] = {
                    'min': fused_data[col].min(),
                    'max': fused_data[col].max(),
                    'mean': fused_data[col].mean(),
                    'std': fused_data[col].std()
                }
    
    return temporal_analysis

# Analyze temporal patterns
temporal_analysis = analyze_temporal_patterns(fused_data)
if temporal_analysis:
    print(f"\n- Temporal Analysis:")
    print(f"   Data points: {temporal_analysis['data_points']}")
    if temporal_analysis['temporal_features']:
        for col, stats in temporal_analysis['temporal_features'].items():
            print(f"   {col}: {stats['min']:.2f} - {stats['max']:.2f} (mean: {stats['mean']:.2f})")

# ============================================================================
# STEP 7: VISUALIZATION AND INSIGHTS
# ============================================================================

print("\n[STEP] Creating visualizations...")

if file2_data is not None:
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comprehensive Analysis Dashboard', fontsize=16)
    
    # 1. Engagement by Category
    if 'Engagement' in file2_data.columns and 'category' in file2_data.columns:
        sns.boxplot(data=file2_data, x='category', y='Engagement', ax=axes[0,0])
        axes[0,0].set_title('Engagement by Category')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # 2. Correlation Heatmap
    metrics = ['Engagement', 'Emotion', 'CognitiveLoad', 'Desire', 'Memorisation', 'Impact']
    available_metrics = [m for m in metrics if m in file2_data.columns]
    if len(available_metrics) > 1:
        corr_matrix = file2_data[available_metrics].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[0,1])
        axes[0,1].set_title('Metric Correlations')
    
    # 3. Top Performers
    if 'Engagement' in file2_data.columns:
        top_10 = file2_data.nlargest(10, 'Engagement')[['SourceStimuliName', 'Engagement']]
        axes[1,0].barh(range(len(top_10)), top_10['Engagement'])
        axes[1,0].set_yticks(range(len(top_10)))
        axes[1,0].set_yticklabels(top_10['SourceStimuliName'], fontsize=8)
        axes[1,0].set_title('Top 10 by Engagement')
        axes[1,0].set_xlabel('Engagement Score')
    
    # 4. Category Distribution
    if 'category' in file2_data.columns:
        category_counts = file2_data['category'].value_counts()
        axes[1,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Distribution by Category')
    
    plt.tight_layout()
    
    # Add timestamp to filename to avoid conflicts
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dashboard_filename = f'analysis_dashboard_{timestamp}.png'
    
    plt.savefig(dashboard_filename, dpi=300, bbox_inches='tight')
    print(f"- Dashboard saved as '{dashboard_filename}'")
    
    # Also save without timestamp for easy access
    plt.savefig('analysis_dashboard.png', dpi=300, bbox_inches='tight')
    print("- Dashboard also saved as 'analysis_dashboard.png'")

# ============================================================================
# STEP 8: SUMMARY AND RECOMMENDATIONS
# ============================================================================

print("\n[STEP] Summary and recommendations...")

def generate_summary(file2_data, file3_data, fused_data):
    """Generate comprehensive summary and recommendations"""
    summary = {
        'data_overview': {},
        'key_insights': [],
        'recommendations': []
    }
    
    if file2_data is not None:
        summary['data_overview']['stimuli_analyzed'] = len(file2_data)
        summary['data_overview']['categories'] = file2_data['category'].value_counts().to_dict()
        
        # Key insights
        if 'Engagement' in file2_data.columns:
            best_engagement = file2_data.loc[file2_data['Engagement'].idxmax()]
            summary['key_insights'].append(f"Best engagement: {best_engagement['SourceStimuliName']} ({best_engagement['Engagement']:.2f})")
        
        if 'Memorisation' in file2_data.columns:
            best_memory = file2_data.loc[file2_data['Memorisation'].idxmax()]
            summary['key_insights'].append(f"Best memorization: {best_memory['SourceStimuliName']} ({best_memory['Memorisation']:.2f})")
    
    if fused_data is not None:
        summary['data_overview']['temporal_data_points'] = len(fused_data)
    
    # Recommendations
    if file2_data is not None:
        ad_data = file2_data[file2_data['category'] == 'Advertisement']
        if not ad_data.empty:
            high_engagement = ad_data[ad_data['Engagement'] > ad_data['Engagement'].quantile(0.75)]
            summary['recommendations'].append(f"Focus on {len(high_engagement)} high-engagement advertisements")
    
    return summary

# Generate and display summary
summary = generate_summary(file2_data, file3_data, fused_data)

print("\n= ANALYSIS SUMMARY:")
print("=" * 50)
if summary['data_overview']:
    print("= Data Overview:")
    for key, value in summary['data_overview'].items():
        print(f"   {key}: {value}")

if summary['key_insights']:
    print("\n= Key Insights:")
    for insight in summary['key_insights']:
        print(f"   • {insight}")

if summary['recommendations']:
    print("\n= Recommendations:")
    for rec in summary['recommendations']:
        print(f"   • {rec}")

print("\n- Analysis workflow completed successfully!")
print("- Check 'analysis_dashboard.png' for visualizations")

# ============================================================================
# SAVE ANALYSIS SUMMARY TO FILE
# ============================================================================

print("\n[STEP] Saving analysis summary to file...")

# Create summary file with timestamp
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
summary_filename = f'analysis_summary_{timestamp}.txt'

# ============================================================================
# SAVE ALL WORKFLOW FILES AND DATA
# ============================================================================

print("\n[STEP] Saving all workflow files and data...")

# Create analysis results directory
analysis_dir = f'analysis_results_{timestamp}'
os.makedirs(analysis_dir, exist_ok=True)
print(f"- Created analysis directory: {analysis_dir}")

# Save detailed analysis data
if file2_data is not None:
    # Save cleaned data
    file2_clean_path = os.path.join(analysis_dir, 'file2_cleaned_data.csv')
    file2_data.to_csv(file2_clean_path, index=False)
    print(f"- Saved cleaned File 2 data: {file2_clean_path}")
    
    # Save category analysis
    category_analysis = file2_data.groupby('category').agg({
        'Engagement': ['mean', 'std', 'min', 'max'],
        'Emotion': ['mean', 'std', 'min', 'max'],
        'CognitiveLoad': ['mean', 'std', 'min', 'max'],
        'Desire': ['mean', 'std', 'min', 'max'],
        'Memorisation': ['mean', 'std', 'min', 'max'],
        'Impact': ['mean', 'std', 'min', 'max']
    }).round(3)
    
    category_path = os.path.join(analysis_dir, 'category_analysis.csv')
    category_analysis.to_csv(category_path)
    print(f"- Saved category analysis: {category_path}")
    
    # Save performance rankings
    metrics = ['Engagement', 'Emotion', 'CognitiveLoad', 'Desire', 'Memorisation', 'Impact']
    rankings_data = []
    
    for metric in metrics:
        if metric in file2_data.columns:
            top_5 = file2_data.nlargest(5, metric)[['SourceStimuliName', 'category', metric]]
            for idx, row in top_5.iterrows():
                rankings_data.append({
                    'Metric': metric,
                    'Rank': len([r for r in rankings_data if r['Metric'] == metric]) + 1,
                    'Stimulus': row['SourceStimuliName'],
                    'Category': row['category'],
                    'Value': row[metric]
                })
    
    if rankings_data:
        rankings_df = pd.DataFrame(rankings_data)
        rankings_path = os.path.join(analysis_dir, 'performance_rankings.csv')
        rankings_df.to_csv(rankings_path, index=False)
        print(f"- Saved performance rankings: {rankings_path}")

# Save File 3 data if available
if file3_data is not None:
    file3_path = os.path.join(analysis_dir, 'file3_data.csv')
    file3_data.to_csv(file3_path, index=False)
    print(f"- Saved File 3 data: {file3_path}")

# Save temporal analysis data
if fused_data is not None:
    # Save sample of temporal data (first 10000 rows to avoid huge files)
    temporal_sample = fused_data.head(10000)
    temporal_path = os.path.join(analysis_dir, 'temporal_data_sample.csv')
    temporal_sample.to_csv(temporal_path, index=False)
    print(f"- Saved temporal data sample: {temporal_path}")
    
    # Save temporal analysis summary
    temporal_summary = {
        'total_rows': len(fused_data),
        'total_columns': len(fused_data.columns),
        'memory_usage_mb': fused_data.memory_usage(deep=True).sum() / 1024 / 1024,
        'columns': list(fused_data.columns)
    }
    
    import json
    temporal_summary_path = os.path.join(analysis_dir, 'temporal_summary.json')
    with open(temporal_summary_path, 'w') as f:
        json.dump(temporal_summary, f, indent=2)
    print(f"- Saved temporal summary: {temporal_summary_path}")

# Save correlation matrix
if file2_data is not None:
    metrics = ['Engagement', 'Emotion', 'CognitiveLoad', 'Desire', 'Memorisation', 'Impact']
    available_metrics = [m for m in metrics if m in file2_data.columns]
    
    if len(available_metrics) > 1:
        corr_matrix = file2_data[available_metrics].corr()
        corr_path = os.path.join(analysis_dir, 'correlation_matrix.csv')
        corr_matrix.to_csv(corr_path)
        print(f"- Saved correlation matrix: {corr_path}")

# Save detailed stimulus reports
if file2_data is not None:
    stimulus_reports = []
    for stim in file2_data['SourceStimuliName'].unique():
        stim_data = file2_data[file2_data['SourceStimuliName'] == stim]
        if not stim_data.empty:
            report = {
                'stimulus_id': stim,
                'category': stim_data['category'].iloc[0] if 'category' in stim_data.columns else 'Unknown'
            }
            
            # Add metrics
            for metric in metrics:
                if metric in stim_data.columns:
                    report[f'{metric}_value'] = stim_data[metric].iloc[0]
                    report[f'{metric}_percentile'] = file2_data[metric].rank(pct=True)[stim_data.index[0]] * 100
            
            stimulus_reports.append(report)
    
    if stimulus_reports:
        reports_df = pd.DataFrame(stimulus_reports)
        reports_path = os.path.join(analysis_dir, 'detailed_stimulus_reports.csv')
        reports_df.to_csv(reports_path, index=False)
        print(f"- Saved detailed stimulus reports: {reports_path}")

# Save workflow configuration and metadata
workflow_metadata = {
    'analysis_timestamp': timestamp,
    'files_processed': {
        'file2_loaded': file2_data is not None,
        'file3_loaded': file3_data is not None,
        'fused_data_loaded': fused_data is not None
    },
    'analysis_parameters': {
        'metrics_analyzed': metrics,
        'categories_found': list(file2_data['category'].unique()) if file2_data is not None else [],
        'total_stimuli': len(file2_data['SourceStimuliName'].unique()) if file2_data is not None else 0
    }
}

metadata_path = os.path.join(analysis_dir, 'workflow_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(workflow_metadata, f, indent=2)
print(f"- Saved workflow metadata: {metadata_path}")

# Copy dashboard to analysis directory
if os.path.exists('analysis_dashboard.png'):
    import shutil
    dashboard_copy = os.path.join(analysis_dir, 'analysis_dashboard.png')
    shutil.copy2('analysis_dashboard.png', dashboard_copy)
    print(f"- Copied dashboard to analysis directory: {dashboard_copy}")

print(f"- All workflow files saved to: {analysis_dir}")

# Move all generated files to the analysis directory
files_to_move = [
    'analysis_dashboard.png',
    'analysis_summary.txt',
    f'analysis_summary_{timestamp}.txt',
    f'analysis_dashboard_{timestamp}.png'
]

for file_path in files_to_move:
    if os.path.exists(file_path):
        try:
            import shutil
            destination = os.path.join(analysis_dir, os.path.basename(file_path))
            shutil.move(file_path, destination)
            print(f"- Moved {file_path} to analysis directory")
        except Exception as e:
            print(f"- Could not move {file_path}: {e}")

# Create file index for easy navigation
file_index = {
    'analysis_directory': analysis_dir,
    'files_created': [
        'file2_cleaned_data.csv - Cleaned and categorized File 2 data',
        'category_analysis.csv - Statistical analysis by category',
        'performance_rankings.csv - Top performers for each metric',
        'file3_data.csv - File 3 data (if available)',
        'temporal_data_sample.csv - Sample of temporal data (first 10,000 rows)',
        'temporal_summary.json - Summary of temporal data characteristics',
        'correlation_matrix.csv - Correlation matrix between metrics',
        'detailed_stimulus_reports.csv - Detailed reports for each stimulus',
        'workflow_metadata.json - Analysis configuration and metadata',
        'analysis_dashboard.png - Visualization dashboard',
        'analysis_summary.txt - Text summary of analysis',
        f'analysis_summary_{timestamp}.txt - Timestamped summary',
        f'analysis_dashboard_{timestamp}.png - Timestamped dashboard'
    ],
    'file_descriptions': {
        'file2_cleaned_data.csv': 'Main cleaned data with categories and metrics',
        'category_analysis.csv': 'Statistical summary by stimulus category',
        'performance_rankings.csv': 'Ranked list of top performers for each metric',
        'correlation_matrix.csv': 'Correlation coefficients between all metrics',
        'detailed_stimulus_reports.csv': 'Individual stimulus performance with percentiles',
        'workflow_metadata.json': 'Analysis configuration and processing details',
        'analysis_dashboard.png': 'Comprehensive visualization dashboard',
        'analysis_summary.txt': 'Text summary of all analysis results'
    }
}

# Save file index
index_path = os.path.join(analysis_dir, 'file_index.json')
with open(index_path, 'w') as f:
    json.dump(file_index, f, indent=2)
print(f"- Saved file index: {index_path}")

# Create README for the analysis directory
readme_content = f"""# Analysis Results - {timestamp}

This directory contains all files generated by the comprehensive analysis workflow.

## 📁 Files Overview

### Data Files
- **file2_cleaned_data.csv**: Main cleaned data with categories and metrics
- **file3_data.csv**: File 3 data (if available)
- **temporal_data_sample.csv**: Sample of temporal data (first 10,000 rows)

### Analysis Files
- **category_analysis.csv**: Statistical summary by stimulus category
- **performance_rankings.csv**: Ranked list of top performers for each metric
- **correlation_matrix.csv**: Correlation coefficients between all metrics
- **detailed_stimulus_reports.csv**: Individual stimulus performance with percentiles

### Summary Files
- **temporal_summary.json**: Summary of temporal data characteristics
- **workflow_metadata.json**: Analysis configuration and processing details
- **file_index.json**: Complete index of all files

### Visualizations
- **analysis_dashboard.png**: Comprehensive visualization dashboard

## 📊 Key Metrics Analyzed
- Engagement
- Emotion
- Cognitive Load
- Desire
- Memorisation
- Impact

## 🎯 Categories Identified
- Advertisement
- Purchase Intent
- Purchase
- Preference
- Unaided Recall
- Brand Stickiness

## 📈 Analysis Features
- Dynamic statistical analysis by category
- Complete reports for each stimulus
- Comparative analysis of advertisements
- Temporal pattern analysis
- Performance rankings and correlations

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

readme_path = os.path.join(analysis_dir, 'README.md')
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"- Created README: {readme_path}")

with open(summary_filename, 'w', encoding='utf-8') as f:
    f.write("COMPREHENSIVE ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    
    if summary['data_overview']:
        f.write("DATA OVERVIEW:\n")
        for key, value in summary['data_overview'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
    
    if summary['key_insights']:
        f.write("KEY INSIGHTS:\n")
        for insight in summary['key_insights']:
            f.write(f"  • {insight}\n")
        f.write("\n")
    
    if summary['recommendations']:
        f.write("RECOMMENDATIONS:\n")
        for rec in summary['recommendations']:
            f.write(f"  • {rec}\n")
        f.write("\n")
    
    f.write(f"Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print(f"- Analysis summary saved as '{summary_filename}'")
print("- Analysis summary also saved as 'analysis_summary.txt'")

# Also save without timestamp
with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
    f.write("COMPREHENSIVE ANALYSIS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    
    if summary['data_overview']:
        f.write("DATA OVERVIEW:\n")
        for key, value in summary['data_overview'].items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")
    
    if summary['key_insights']:
        f.write("KEY INSIGHTS:\n")
        for insight in summary['key_insights']:
            f.write(f"  • {insight}\n")
        f.write("\n")
    
    if summary['recommendations']:
        f.write("RECOMMENDATIONS:\n")
        for rec in summary['recommendations']:
            f.write(f"  • {rec}\n")
        f.write("\n")
    
    f.write(f"Analysis completed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print("- All analysis files saved successfully!")

# Final summary of what was created
print(f"\n{'='*60}")
print(f"📁 COMPLETE ANALYSIS PACKAGE CREATED")
print(f"{'='*60}")
print(f"📂 Analysis Directory: {analysis_dir}")
print(f"📊 Total Files Created: {len(os.listdir(analysis_dir))}")
print(f"📈 Analysis Timestamp: {timestamp}")
print(f"🎯 All files are organized in the analysis directory")
print(f"📋 Check the README.md file for complete documentation")
print(f"{'='*60}")

# List all files in the directory
print(f"\n📁 Files in {analysis_dir}:")
for file in sorted(os.listdir(analysis_dir)):
    print(f"   📄 {file}")

print(f"\n✅ Analysis workflow completed successfully!")
print(f"📁 All results saved in: {analysis_dir}")

def analyze_time_series_patterns(cleaned_fused_path, output_dir):
    """Analyze time-series patterns in the data pipeline"""
    print("\n[STEP] Analyzing time-series patterns in data pipeline...")
    
    if not os.path.exists(cleaned_fused_path):
        print(f"[ERROR] {cleaned_fused_path} not found.")
        return
    
    # Load data with proper separator
    fused_data = pd.read_csv(cleaned_fused_path, sep=';')
    temporal_dir = os.path.join(output_dir, "time_series_analysis")
    os.makedirs(temporal_dir, exist_ok=True)
    
    print(f"Analyzing {len(fused_data)} data points for time-series patterns...")
    
    # 1. Find temporal columns
    temporal_columns = [col for col in fused_data.columns if any(x in col.lower() for x in ['timestamp', 'time', 'duration'])]
    
    if not temporal_columns:
        print("- No temporal columns found for time-series analysis")
        return
    
    # 2. Analyze time-series patterns
    print("Analyzing time-series patterns...")
    
    # Find main timestamp column
    main_timestamp = None
    for col in temporal_columns:
        if 'timestamp' in col.lower():
            main_timestamp = col
            break
    
    if main_timestamp:
        # Convert to numeric for analysis
        fused_data[main_timestamp] = pd.to_numeric(fused_data[main_timestamp], errors='coerce')
        valid_data = fused_data.dropna(subset=[main_timestamp])
        
        if len(valid_data) > 0:
            # Sort by timestamp
            valid_data = valid_data.sort_values(main_timestamp)
            
            # 3. Time-series characteristics
            time_series_analysis = {
                'total_data_points': len(valid_data),
                'time_span_seconds': valid_data[main_timestamp].max() - valid_data[main_timestamp].min(),
                'time_span_hours': (valid_data[main_timestamp].max() - valid_data[main_timestamp].min()) / 3600,
                'average_sampling_rate_hz': len(valid_data) / ((valid_data[main_timestamp].max() - valid_data[main_timestamp].min()) / 1000),
                'temporal_gaps_detected': 0,
                'data_continuity_score': 0
            }
            
            # 4. Detect temporal gaps and continuity
            time_diffs = valid_data[main_timestamp].diff().dropna()
            if len(time_diffs) > 0:
                median_gap = time_diffs.median()
                large_gaps = time_diffs[time_diffs > median_gap * 10]
                time_series_analysis['temporal_gaps_detected'] = len(large_gaps)
                time_series_analysis['data_continuity_score'] = 1 - (len(large_gaps) / len(time_diffs))
            
            # Save time-series analysis
            pd.DataFrame([time_series_analysis]).to_csv(os.path.join(temporal_dir, '01_time_series_characteristics.csv'), index=False)
            print(f"- Saved time-series characteristics")
            
            # 5. Temporal sampling analysis
            print("Performing temporal sampling analysis...")
            sampling_analysis = []
            
            # Analyze different time windows
            time_windows = [1000, 5000, 10000]  # seconds
            for window in time_windows:
                window_start = valid_data[main_timestamp].min()
                window_end = window_start + window
                
                window_data = valid_data[(valid_data[main_timestamp] >= window_start) & 
                                       (valid_data[main_timestamp] <= window_end)]
                
                if len(window_data) > 0:
                    sampling_analysis.append({
                        'time_window_seconds': window,
                        'data_points_in_window': len(window_data),
                        'sampling_rate_hz': len(window_data) / window,
                        'unique_respondents': window_data['respondent_id'].nunique() if 'respondent_id' in window_data.columns else 0,
                        'start_time': window_start,
                        'end_time': window_end
                    })
            
            pd.DataFrame(sampling_analysis).to_csv(os.path.join(temporal_dir, '02_temporal_sampling_analysis.csv'), index=False)
            print(f"- Saved temporal sampling analysis")
            
            # 6. Time-series patterns by respondent
            if 'respondent_id' in valid_data.columns:
                print("Analyzing respondent-specific time-series patterns...")
                respondent_patterns = []
                
                for respondent in valid_data['respondent_id'].unique():
                    resp_data = valid_data[valid_data['respondent_id'] == respondent]
                    if len(resp_data) > 1:
                        resp_time_span = resp_data[main_timestamp].max() - resp_data[main_timestamp].min()
                        resp_pattern = {
                            'respondent_id': respondent,
                            'data_points': len(resp_data),
                            'time_span_seconds': resp_time_span,
                            'time_span_minutes': resp_time_span / 60,
                            'sampling_rate_hz': len(resp_data) / resp_time_span if resp_time_span > 0 else 0,
                            'start_time': resp_data[main_timestamp].min(),
                            'end_time': resp_data[main_timestamp].max()
                        }
                        respondent_patterns.append(resp_pattern)
                
                pd.DataFrame(respondent_patterns).to_csv(os.path.join(temporal_dir, '03_respondent_time_series_patterns.csv'), index=False)
                print(f"- Saved respondent time-series patterns")
            
            # 7. Summary insights
            insights = [
                f"Time-series analysis completed on {len(valid_data)} data points",
                f"Data collection spanned {time_series_analysis['time_span_hours']:.2f} hours",
                f"Average sampling rate: {time_series_analysis['average_sampling_rate_hz']:.2f} Hz",
                f"Data continuity score: {time_series_analysis['data_continuity_score']:.2%}",
                f"Temporal gaps detected: {time_series_analysis['temporal_gaps_detected']}"
            ]
            
            pd.DataFrame({'Time_Series_Insight': insights}).to_csv(os.path.join(temporal_dir, '04_time_series_insights.csv'), index=False)
            print(f"- Saved time-series insights")
            
            print(f"✅ Time-series analysis completed and saved in: {temporal_dir}")
        else:
            print("- No valid temporal data found for analysis")
    else:
        print("- No suitable timestamp column found for time-series analysis")

# Copy fused data files to analysis directory if they exist
def copy_fused_data_to_analysis(analysis_dir):
    """Copy fused data files to the analysis directory"""
    print("\n[STEP] Copying fused data to analysis directory...")
    
    # Check if fusion_results directory exists
    fusion_results_dir = "fusion_results"
    if os.path.exists(fusion_results_dir):
        files_to_copy = [
            'fused_data.csv',
            'fused_respondents_clean.csv'
        ]
        
        for file_name in files_to_copy:
            source_path = os.path.join(fusion_results_dir, file_name)
            if os.path.exists(source_path):
                import shutil
                dest_path = os.path.join(analysis_dir, file_name)
                shutil.copy2(source_path, dest_path)
                print(f"- Copied {file_name} to analysis directory")
            else:
                print(f"- {file_name} not found in fusion_results")
    else:
        print("- fusion_results directory not found")

# Copy fused data files
copy_fused_data_to_analysis(analysis_dir)

# Analyze time-series patterns
cleaned_fused_path = os.path.join(analysis_dir, 'fused_respondents_clean.csv')
analyze_time_series_patterns(cleaned_fused_path, analysis_dir)
