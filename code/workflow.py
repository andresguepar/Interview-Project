
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
    # Cargar y combinar todos los archivos individuales de fused_respondents
    fused_files = glob.glob('fused_respondents/*.csv')
    if fused_files:
        df_list = [pd.read_csv(f) for f in fused_files]
        fused_data = pd.concat(df_list, ignore_index=True)
        print(f"- Fused respondents loaded: {fused_data.shape[0]} rows, {fused_data.shape[1]} columns")
    else:
        fused_data = None
        print("[ERROR] No fused respondent files found in 'fused_respondents/'")
except Exception as e:
    print(f"[ERROR] Error loading fused respondent files: {e}")

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
            print(f"\n- {metric} Analysis:")
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
            print(f"\n-  {report['category']} ({report['stimulus_count']} stimuli):")
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

# Eliminar la generación de archivos redundantes
# Mantener solo los outputs principales

# Mantener solo:
# - analysis_dashboard.png
# - category_analysis.csv
# - performance_rankings.csv
# - file2_cleaned_data.csv
# - file3_data.csv
# - correlation_matrix.csv

# Final summary of what was created
print(f"\n{'='*60}")
print(f"- COMPLETE ANALYSIS PACKAGE CREATED")
print(f"{'='*60}")
print(f"- Analysis Directory: {analysis_dir}")
print(f"- Total Files Created: {len(os.listdir(analysis_dir))}")
print(f"- Analysis Timestamp: {timestamp}")
print(f"- All files are organized in the analysis directory")
print(f"- Check the README.md file for complete documentation")
print(f"{'='*60}")

# List all files in the directory
print(f"\n- Files in {analysis_dir}:")
for file in sorted(os.listdir(analysis_dir)):
    print(f"   📄 {file}")

print(f"\n- Analysis workflow completed successfully!")
print(f"- All results saved in: {analysis_dir}")

# Eliminar la función analyze_time_series_patterns y su llamada
# Busca y elimina:
# def analyze_time_series_patterns(...):
#     ...
# Y la línea:
# analyze_time_series_patterns(fused_data, analysis_dir)
