import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_eeg_data(folder_path):
    """Load EEG data from the specified folder"""
    eeg_files = []
    eeg_folder = os.path.join(folder_path, "EEG - Folder 1(a)")
    
    if os.path.exists(eeg_folder):
        for file in os.listdir(eeg_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(eeg_folder, file)
                try:
                    df = pd.read_csv(file_path, sep=';')
                    df['source_file'] = file
                    df['data_type'] = 'EEG'
                    eeg_files.append(df)
                    print(f"Loaded EEG file: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    if eeg_files:
        return pd.concat(eeg_files, ignore_index=True)
    return None

def load_eye_tracking_data(folder_path):
    """Load Eye Tracking data from the specified folder"""
    et_files = []
    et_folder = os.path.join(folder_path, "Eye Tracking - Folder 1(b)")
    
    if os.path.exists(et_folder):
        for file in os.listdir(et_folder):
            if file.endswith('.csv'):
                file_path = os.path.join(et_folder, file)
                try:
                    df = pd.read_csv(file_path, sep=';')
                    df['source_file'] = file
                    df['data_type'] = 'Eye_Tracking'
                    et_files.append(df)
                    print(f"Loaded Eye Tracking file: {file}")
                except Exception as e:
                    print(f"Error loading {file}: {e}")
    
    if et_files:
        return pd.concat(et_files, ignore_index=True)
    return None

def fuse_data(eeg_data, et_data, method='timestamp'):
    """Fuse EEG and Eye Tracking data based on timestamp alignment"""
    if eeg_data is None or et_data is None:
        print("Error: Both EEG and Eye Tracking data are required for fusion")
        return None
    
    print("Starting data fusion...")
    
    # Ensure timestamp columns exist
    eeg_timestamp_col = None
    et_timestamp_col = None
    
    # Find timestamp columns
    for col in eeg_data.columns:
        if 'timestamp' in col.lower():
            eeg_timestamp_col = col
            break
    
    for col in et_data.columns:
        if 'timestamp' in col.lower():
            et_timestamp_col = col
            break
    
    if eeg_timestamp_col is None or et_timestamp_col is None:
        print("Warning: Timestamp columns not found, using index-based fusion")
        method = 'index'
    
    if method == 'timestamp':
        # Convert timestamps to numeric
        eeg_data[eeg_timestamp_col] = pd.to_numeric(eeg_data[eeg_timestamp_col], errors='coerce')
        et_data[et_timestamp_col] = pd.to_numeric(et_data[et_timestamp_col], errors='coerce')
        
        # Remove rows with invalid timestamps
        eeg_data = eeg_data.dropna(subset=[eeg_timestamp_col])
        et_data = et_data.dropna(subset=[et_timestamp_col])
        
        # Sort by timestamp
        eeg_data = eeg_data.sort_values(eeg_timestamp_col)
        et_data = et_data.sort_values(et_timestamp_col)
        
        # Merge on timestamp (nearest match)
        fused_data = pd.merge_asof(
            eeg_data, 
            et_data, 
            left_on=eeg_timestamp_col, 
            right_on=et_timestamp_col,
            direction='nearest',
            tolerance=1000  # 1 second tolerance
        )
        
        print(f"Fused data shape: {fused_data.shape}")
        return fused_data
    
    elif method == 'index':
        # Simple concatenation based on index
        min_length = min(len(eeg_data), len(et_data))
        eeg_subset = eeg_data.head(min_length).reset_index(drop=True)
        et_subset = et_data.head(min_length).reset_index(drop=True)
        
        # Add prefixes to avoid column conflicts
        eeg_subset = eeg_subset.add_prefix('eeg_')
        et_subset = et_subset.add_prefix('et_')
        
        fused_data = pd.concat([eeg_subset, et_subset], axis=1)
        print(f"Fused data shape: {fused_data.shape}")
        return fused_data

def process_individual_respondents(base_path):
    """Process individual respondent data and create fused files"""
    respondents_dir = os.path.join(base_path, "Data", "Interview", "Folder 1")
    
    if not os.path.exists(respondents_dir):
        print(f"Directory not found: {respondents_dir}")
        return
    
    # Create output directory
    output_dir = "fused_respondents"
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each respondent folder
    for folder in os.listdir(respondents_dir):
        folder_path = os.path.join(respondents_dir, folder)
        
        if os.path.isdir(folder_path):
            print(f"\nProcessing respondent: {folder}")
            
            # Load data
            eeg_data = load_eeg_data(folder_path)
            et_data = load_eye_tracking_data(folder_path)
            
            if eeg_data is not None and et_data is not None:
                # Fuse data
                fused_data = fuse_data(eeg_data, et_data)
                
                if fused_data is not None:
                    # Save fused data
                    output_file = os.path.join(output_dir, f"fused_{folder}.csv")
                    fused_data.to_csv(output_file, index=False)
                    print(f"Saved fused data: {output_file}")
                    
                    # Save summary
                    summary = {
                        'respondent': folder,
                        'eeg_rows': len(eeg_data),
                        'et_rows': len(et_data),
                        'fused_rows': len(fused_data),
                        'eeg_columns': len(eeg_data.columns),
                        'et_columns': len(et_data.columns),
                        'fused_columns': len(fused_data.columns)
                    }
                    print(f"Summary: {summary}")
            else:
                print(f"Could not load data for {folder}")

def create_combined_fusion():
    """Create a combined fusion of all respondent data"""
    print("\nCreating combined fusion...")
    
    fused_dir = "fused_respondents"
    if not os.path.exists(fused_dir):
        print("No fused respondent files found. Run process_individual_respondents first.")
        return
    
    # Create main results directory
    results_dir = "fusion_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load all fused files
    fused_files = []
    for file in os.listdir(fused_dir):
        if file.startswith('fused_') and file.endswith('.csv'):
            file_path = os.path.join(fused_dir, file)
            try:
                df = pd.read_csv(file_path)
                df['respondent_id'] = file.replace('fused_', '').replace('.csv', '')
                fused_files.append(df)
                print(f"Loaded: {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
    
    if fused_files:
        # Combine all data
        combined_data = pd.concat(fused_files, ignore_index=True)
        
        # Save combined data in results directory
        combined_path = os.path.join(results_dir, 'fused_data.csv')
        combined_data.to_csv(combined_path, index=False)
        print(f"Saved combined fusion: {combined_path} ({combined_data.shape})")
        
        # Create summary
        summary = {
            'total_respondents': len(fused_files),
            'total_rows': len(combined_data),
            'total_columns': len(combined_data.columns),
            'respondents': [f.replace('fused_', '').replace('.csv', '') for f in os.listdir(fused_dir) if f.startswith('fused_')]
        }
        print(f"Combined fusion summary: {summary}")
        
        return combined_data, results_dir
    else:
        print("No fused files found to combine")
        return None, results_dir

def clean_fused_data(results_dir="fusion_results"):
    """Clean and prepare fused data for analysis"""
    print("\nCleaning fused data...")
    
    fused_data_path = os.path.join(results_dir, 'fused_data.csv')
    if not os.path.exists(fused_data_path):
        print(f"{fused_data_path} not found. Run create_combined_fusion first.")
        return None
    
    # Load fused data
    fused_data = pd.read_csv(fused_data_path)
    print(f"Original shape: {fused_data.shape}")
    
    # Basic cleaning
    # Remove completely empty rows
    fused_data = fused_data.dropna(how='all')
    print(f"After removing empty rows: {fused_data.shape}")
    
    # Remove completely empty columns
    fused_data = fused_data.dropna(axis=1, how='all')
    print(f"After removing empty columns: {fused_data.shape}")
    
    # Clean column names
    fused_data.columns = fused_data.columns.str.strip()
    
    # Save cleaned data in results directory
    clean_path = os.path.join(results_dir, 'fused_respondents_clean.csv')
    fused_data.to_csv(clean_path, index=False)
    print(f"Saved cleaned data: {clean_path}")
    
    return fused_data

def create_excel_friendly_csvs(results_dir="fusion_results"):
    """Create Excel-friendly CSV files - just the data, no summaries"""
    print("\nCreating Excel-friendly CSV files...")
    
    # Create organized directory within results
    excel_dir = os.path.join(results_dir, "excel_friendly_respondents")
    os.makedirs(excel_dir, exist_ok=True)
    
    # Load individual respondent files from fused_respondents directory
    fused_dir = "fused_respondents"
    if not os.path.exists(fused_dir):
        print(f"{fused_dir} directory not found. Run process_individual_respondents first.")
        return None
    
    # Process each fused respondent file
    respondent_files = [f for f in os.listdir(fused_dir) if f.startswith('fused_') and f.endswith('.csv')]
    
    if not respondent_files:
        print("No fused respondent files found.")
        return None
    
    print(f"Found {len(respondent_files)} respondent files to process...")
    
    for file in respondent_files:
        respondent_id = file.replace('fused_', '').replace('.csv', '')
        file_path = os.path.join(fused_dir, file)
        
        print(f"\nProcessing respondent: {respondent_id}")
        
        try:
            # Load the fused data
            df = pd.read_csv(file_path)
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            
            # Create Excel-friendly version
            excel_file = f"{respondent_id}.csv"
            excel_path = os.path.join(excel_dir, excel_file)
            
            # Clean column names for Excel
            df_clean = df.copy()
            df_clean.columns = df_clean.columns.str.strip()
            
            # Add respondent ID column if not present
            if 'respondent_id' not in df_clean.columns:
                df_clean['respondent_id'] = respondent_id
            
            # Save Excel-friendly version
            df_clean.to_csv(excel_path, index=False)
            print(f"  Saved: {excel_file}")
            
        except Exception as e:
            print(f"  Error processing {file}: {e}")
    
    print(f"\n✅ Excel-friendly data files created in: {excel_dir}")
    print(f"📁 Total files created: {len(os.listdir(excel_dir))}")
    
    return excel_dir

def main():
    """Main function to run the complete fusion process"""
    print("=== EEG and Eye Tracking Data Fusion ===")
    print("=" * 50)
    
    # Step 1: Process individual respondents
    print("\n[STEP 1] Processing individual respondents...")
    process_individual_respondents(".")
    
    # Step 2: Create combined fusion
    print("\n[STEP 2] Creating combined fusion...")
    combined_data, results_dir = create_combined_fusion()
    
    # Step 3: Clean fused data
    print("\n[STEP 3] Cleaning fused data...")
    clean_fused_data(results_dir)
    
    # Step 4: Create Excel-friendly files
    print("\n[STEP 4] Creating Excel-friendly files...")
    create_excel_friendly_csvs(results_dir)
    
    # Create final summary
    print(f"\n{'='*60}")
    print(f"📁 FUSION RESULTS CREATED")
    print(f"{'='*60}")
    print(f"📂 Main Results Directory: {results_dir}")
    
    # List all created directories
    if os.path.exists(results_dir):
        subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
        print(f"📁 Subdirectories created:")
        for subdir in subdirs:
            subdir_path = os.path.join(results_dir, subdir)
            file_count = len(os.listdir(subdir_path))
            print(f"   📂 {subdir}/ ({file_count} files)")
    
    print(f"\n✅ All files organized in: {results_dir}")
    print(f"📊 No files created outside the main directory!")
    print(f"{'='*60}")
    
    print("\n=== Fusion process completed ===")
    print(f"📁 Check the '{results_dir}' folder for all results!")

if __name__ == "__main__":
    main()
