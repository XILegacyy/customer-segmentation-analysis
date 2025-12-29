"""
01_exploration.py - Online Retail Dataset Exploration
SIMPLE, GUARANTEED VERSION - Generates reports without plt.show()

Author: Awande Gcabashe
Date: 12/29/2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=" * 70)
print("CUSTOMER SEGMENTATION - DATA EXPLORATION")
print("=" * 70)

# ==================== STEP 1: LOAD DATA ====================
print("\n📁 STEP 1: LOADING DATASET")
print("-" * 40)

# Define the path
data_path = "data/online_retail_II.xlsx"

# Check if file exists
if not os.path.exists(data_path):
    print(f"❌ ERROR: File not found: {data_path}")
    print("Files in data/ directory:")
    print(os.listdir("data/") if os.path.exists("data") else "No data directory")
    exit(1)

print(f"✓ Found file: {data_path}")

# Load the data
try:
    # First, see what sheets exist
    excel_file = pd.ExcelFile(data_path)
    print(f"Excel sheets: {excel_file.sheet_names}")
    
    # Load each sheet
    all_data = []
    for sheet in excel_file.sheet_names:
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet)
        all_data.append(df_sheet)
        print(f"  • {sheet}: {len(df_sheet):,} rows")
    
    # Combine all sheets
    df = pd.concat(all_data, ignore_index=True)
    print(f"✓ Combined total: {len(df):,} rows, {df.shape[1]} columns")
    
except Exception as e:
    print(f"❌ Error loading: {e}")
    exit(1)

# ==================== STEP 2: BASIC INFO ====================
print("\n📊 STEP 2: BASIC INFORMATION")
print("-" * 40)

print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i:2d}. {col}")

# Memory usage
memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
print(f"\nMemory usage: {memory_mb:.2f} MB")

# ==================== STEP 3: MISSING VALUES ====================
print("\n🔍 STEP 3: MISSING VALUES ANALYSIS")
print("-" * 40)

missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Missing_Percent': missing_pct
})

# Show columns with missing values
missing_cols = missing_df[missing_df['Missing_Count'] > 0]

if len(missing_cols) == 0:
    print("✓ No missing values!")
else:
    print(f"Found {len(missing_cols)} columns with missing values:")
    for col in missing_cols.index:
        count = missing_df.loc[col, 'Missing_Count']
        pct = missing_df.loc[col, 'Missing_Percent']
        print(f"  • {col:<15}: {count:>8,} missing ({pct:.2f}%)")

# ==================== STEP 4: DATA TYPES ====================
print("\n📋 STEP 4: DATA TYPES")
print("-" * 40)

print("Column data types:")
for col, dtype in df.dtypes.items():
    print(f"  • {col:<20}: {dtype}")

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}): {', '.join(numeric_cols)}")
print(f"Categorical columns ({len(categorical_cols)}): {', '.join(categorical_cols)}")

# ==================== STEP 5: NUMERIC STATS ====================
print("\n📈 STEP 5: NUMERIC STATISTICS")
print("-" * 40)

if numeric_cols:
    numeric_stats = df[numeric_cols].describe()
    print("Summary statistics for numeric columns:")
    
    # Display key stats for important columns
    for col in numeric_cols:
        if col in ['Quantity', 'Price']:
            print(f"\n{col}:")
            print(f"  • Min: {df[col].min():.2f}")
            print(f"  • Max: {df[col].max():.2f}")
            print(f"  • Mean: {df[col].mean():.2f}")
            print(f"  • Median: {df[col].median():.2f}")
            print(f"  • Std Dev: {df[col].std():.2f}")
            
            # Check for negatives (returns)
            if col == 'Quantity':
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    pct_negative = (negative_count / len(df)) * 100
                    print(f"  • Negative values: {negative_count:,} ({pct_negative:.2f}%) - RETURNS")
else:
    print("No numeric columns found")

# ==================== STEP 6: CATEGORICAL STATS ====================
print("\n🎯 STEP 6: CATEGORICAL STATISTICS")
print("-" * 40)

if categorical_cols:
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"\n{col}:")
        print(f"  • Unique values: {unique_count:,}")
        
        if unique_count <= 10:
            # Show all values if few unique
            value_counts = df[col].value_counts()
            for val, count in value_counts.items():
                pct = (count / len(df)) * 100
                print(f"     {val}: {count:,} ({pct:.1f}%)")
        else:
            # Show top 5 if many unique
            top_5 = df[col].value_counts().head(5)
            print(f"  • Top 5 values:")
            for val, count in top_5.items():
                pct = (count / len(df)) * 100
                print(f"     {val}: {count:,} ({pct:.1f}%)")
else:
    print("No categorical columns found")

# ==================== STEP 7: BUSINESS METRICS ====================
print("\n💰 STEP 7: BUSINESS METRICS")
print("-" * 40)

# Check if we have required columns
if 'InvoiceDate' in df.columns:
    # Convert to datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    print("Time Analysis:")
    print(f"  • Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    print(f"  • Total days: {(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} days")

if 'Quantity' in df.columns and 'Price' in df.columns:
    # Calculate transaction value
    df['TransactionValue'] = df['Quantity'] * df['Price']
    
    print("\nTransaction Analysis:")
    print(f"  • Total transaction value: £{df['TransactionValue'].sum():,.2f}")
    print(f"  • Average transaction value: £{df['TransactionValue'].mean():.2f}")
    
    # Separate purchases and returns
    purchases = df[df['Quantity'] > 0]
    returns = df[df['Quantity'] < 0]
    
    if len(purchases) > 0:
        print(f"  • Purchase transactions: {len(purchases):,}")
        print(f"  • Total purchase value: £{purchases['TransactionValue'].sum():,.2f}")
    
    if len(returns) > 0:
        print(f"  • Return transactions: {len(returns):,} ({len(returns)/len(df)*100:.1f}%)")
        print(f"  • Total return value: £{returns['TransactionValue'].sum():,.2f}")

if 'Customer ID' in df.columns:
    print(f"\nCustomer Analysis:")
    print(f"  • Unique customers: {df['Customer ID'].nunique():,}")
    # Filter out NaN for customer count
    valid_customers = df['Customer ID'].notna().sum()
    print(f"  • Transactions with customer ID: {valid_customers:,} ({valid_customers/len(df)*100:.1f}%)")

if 'Invoice' in df.columns:
    print(f"  • Unique invoices: {df['Invoice'].nunique():,}")

if 'StockCode' in df.columns:
    print(f"  • Unique products: {df['StockCode'].nunique():,}")

# ==================== STEP 8: SAVE REPORT ====================
print("\n💾 STEP 8: SAVING EXPLORATION REPORT")
print("-" * 40)

# Create reports directory if needed
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/images", exist_ok=True)

report_path = "reports/data_exploration_report.txt"

# Create report content
report_content = f"""
{'='*70}
CUSTOMER SEGMENTATION - DATA EXPLORATION REPORT
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: Online Retail II

1. DATASET OVERVIEW
{'-'*40}
Total Rows: {df.shape[0]:,}
Total Columns: {df.shape[1]}
Memory Usage: {memory_mb:.2f} MB

2. COLUMNS
{'-'*40}
"""
for i, col in enumerate(df.columns, 1):
    report_content += f"{i:2d}. {col} ({df[col].dtype})\n"

report_content += f"""
3. MISSING VALUES
{'-'*40}
"""
if len(missing_cols) > 0:
    for col in missing_cols.index:
        count = missing_df.loc[col, 'Missing_Count']
        pct = missing_df.loc[col, 'Missing_Percent']
        report_content += f"{col}: {count:,} missing ({pct:.2f}%)\n"
else:
    report_content += "No missing values\n"

report_content += f"""
4. KEY STATISTICS
{'-'*40}
"""
if 'Quantity' in df.columns and 'Price' in df.columns:
    report_content += f"Quantity: {df['Quantity'].min():.2f} to {df['Quantity'].max():.2f}\n"
    report_content += f"Price: £{df['Price'].min():.2f} to £{df['Price'].max():.2f}\n"

report_content += f"""
5. BUSINESS METRICS
{'-'*40}
"""
if 'InvoiceDate' in df.columns:
    report_content += f"Date Range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}\n"

if 'TransactionValue' in df.columns:
    report_content += f"Total Transaction Value: £{df['TransactionValue'].sum():,.2f}\n"
    report_content += f"Average Transaction Value: £{df['TransactionValue'].mean():.2f}\n"

if 'Customer ID' in df.columns:
    report_content += f"Unique Customers: {df['Customer ID'].nunique():,}\n"

report_content += f"""
6. DATA QUALITY ISSUES
{'-'*40}
1. Missing Customer IDs: {(df['Customer ID'].isna().sum() if 'Customer ID' in df.columns else 0):,} rows
2. Negative Quantities (Returns): {(len(returns) if 'Quantity' in df.columns else 0):,} transactions
3. Missing Descriptions: {(df['Description'].isna().sum() if 'Description' in df.columns else 0):,} rows

7. RECOMMENDATIONS FOR CLEANING
{'-'*40}
1. Remove rows without Customer ID (critical for segmentation)
2. Handle returns separately or remove for purchase analysis
3. Fill missing descriptions with 'Unknown'
4. Check for outliers in Price and Quantity
"""

# Save the report
try:
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ Report saved to: {report_path}")
    print(f"✓ Report size: {os.path.getsize(report_path)/1024:.1f} KB")
    
except Exception as e:
    print(f"❌ Error saving report: {e}")

# ==================== STEP 9: SAVE SAMPLE DATA ====================
print("\n💾 STEP 9: SAVING SAMPLE DATA")
print("-" * 40)

# Save first 1000 rows for quick testing
sample_path = "data/sample_data.csv"
try:
    df.head(1000).to_csv(sample_path, index=False)
    print(f"✓ Sample data saved to: {sample_path}")
except Exception as e:
    print(f"⚠ Could not save sample: {e}")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 70)
print("✅ DATA EXPLORATION COMPLETE!")
print("=" * 70)

print("\nKEY FINDINGS:")
print(f"1. Total transactions: {df.shape[0]:,}")
print(f"2. Missing Customer IDs: {(df['Customer ID'].isna().sum() if 'Customer ID' in df.columns else 'N/A'):,}")
print(f"3. Return transactions: {(len(returns) if 'Quantity' in df.columns else 'N/A'):,}")
print(f"4. Date range: {df['InvoiceDate'].min().date() if 'InvoiceDate' in df.columns else 'N/A'} to "
      f"{df['InvoiceDate'].max().date() if 'InvoiceDate' in df.columns else 'N/A'}")

print("\nNEXT STEPS:")
print("1. Review the report at: reports/data_exploration_report.txt")
print("2. Run 02_cleaning.py to fix data quality issues")

# Display first few rows
print("\n" + "-" * 70)
print("FIRST 3 ROWS OF DATA:")
print("-" * 70)
print(df.head(3))