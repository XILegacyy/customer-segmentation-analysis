"""
02_cleaning.py - Data Cleaning for Customer Segmentation

Cleans the Online Retail dataset by handling:
1. Missing Customer IDs (CRITICAL - remove rows)
2. Negative quantities (returns) - remove for purchase analysis
3. Missing descriptions - fill with 'Unknown'
4. Outliers in Price and Quantity

Author: Awande Gcabashe
Date: 12/29/2025
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=" * 70)
print("CUSTOMER SEGMENTATION PROJECT - DATA CLEANING")
print("=" * 70)

# ==================== STEP 1: LOAD RAW DATA ====================
print("\n📁 STEP 1: LOADING RAW DATA")
print("-" * 40)

def load_raw_data():
    """Load the raw dataset from Excel file."""
    data_path = "data/online_retail_II.xlsx"
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: File not found: {data_path}")
        return None
    
    print(f"Loading from: {data_path}")
    
    try:
        excel_file = pd.ExcelFile(data_path)
        print(f"Found sheets: {excel_file.sheet_names}")
        
        # Load all sheets
        all_data = []
        for sheet in excel_file.sheet_names:
            df_sheet = pd.read_excel(excel_file, sheet_name=sheet)
            all_data.append(df_sheet)
        
        # Combine
        df = pd.concat(all_data, ignore_index=True)
        print(f"✓ Loaded {len(df):,} rows, {df.shape[1]} columns")
        return df
        
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return None

# Load the data
df = load_raw_data()
if df is None:
    exit(1)

initial_rows = len(df)
print(f"Initial dataset: {initial_rows:,} rows")

# ==================== STEP 2: HANDLE MISSING CUSTOMER IDs ====================
print("\n🧹 STEP 2: HANDLING MISSING CUSTOMER IDs")
print("-" * 40)

# Check for missing Customer IDs
missing_customers = df['Customer ID'].isna().sum()
missing_pct = (missing_customers / initial_rows) * 100

print(f"Missing Customer IDs: {missing_customers:,} rows ({missing_pct:.2f}%)")

# STRATEGY: Remove rows without Customer ID (CAN'T segment unknown customers!)
df_clean = df.dropna(subset=['Customer ID']).copy()
rows_removed_customers = initial_rows - len(df_clean)

print(f"✓ Removed {rows_removed_customers:,} rows without Customer ID")
print(f"Rows remaining: {len(df_clean):,}")

# ==================== STEP 3: HANDLE RETURNS (NEGATIVE QUANTITIES) ====================
print("\n🔄 STEP 3: HANDLING RETURNS (NEGATIVE QUANTITIES)")
print("-" * 40)

# Identify returns
returns_mask = df_clean['Quantity'] < 0
returns_count = returns_mask.sum()
returns_pct = (returns_count / len(df_clean)) * 100

print(f"Return transactions: {returns_count:,} ({returns_pct:.2f}%)")

if returns_count > 0:
    # Calculate return value
    df_clean['TransactionValue'] = df_clean['Quantity'] * df_clean['Price']
    returns_value = df_clean[returns_mask]['TransactionValue'].sum()
    
    print(f"Total return value: £{returns_value:,.2f}")
    
    # STRATEGY: Remove returns for purchase analysis (focus on actual purchases)
    df_clean = df_clean[df_clean['Quantity'] > 0].copy()
    rows_removed_returns = returns_count
    
    print(f"✓ Removed {rows_removed_returns:,} return transactions")
    print(f"Rows remaining: {len(df_clean):,}")
else:
    print("✓ No returns found")
    rows_removed_returns = 0

# ==================== STEP 4: HANDLE MISSING DESCRIPTIONS ====================
print("\n📝 STEP 4: HANDLING MISSING DESCRIPTIONS")
print("-" * 40)

missing_desc = df_clean['Description'].isna().sum()
if missing_desc > 0:
    missing_desc_pct = (missing_desc / len(df_clean)) * 100
    print(f"Missing descriptions: {missing_desc:,} ({missing_desc_pct:.2f}%)")
    
    # STRATEGY: Fill with 'Unknown Product'
    df_clean['Description'] = df_clean['Description'].fillna('Unknown Product')
    print(f"✓ Filled {missing_desc:,} missing descriptions with 'Unknown Product'")
else:
    print("✓ No missing descriptions")

# ==================== STEP 5: HANDLE OUTLIERS ====================
print("\n📊 STEP 5: HANDLING OUTLIERS IN PRICE AND QUANTITY")
print("-" * 40)

def handle_column_outliers(column_data, column_name):
    """Cap outliers using IQR method."""
    if len(column_data) == 0:
        return column_data, {}
    
    # Calculate IQR
    Q1 = column_data.quantile(0.25)
    Q3 = column_data.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds (3*IQR is conservative)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Identify outliers
    outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
    outlier_count = len(outliers)
    
    if outlier_count > 0:
        outlier_pct = (outlier_count / len(column_data)) * 100
        print(f"\n{column_name}:")
        print(f"  • Original range: [{column_data.min():.2f}, {column_data.max():.2f}]")
        print(f"  • Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        print(f"  • Outliers found: {outlier_count:,} ({outlier_pct:.2f}%)")
        
        # Cap outliers
        column_clean = column_data.clip(lower_bound, upper_bound)
        
        print(f"  • New range: [{column_clean.min():.2f}, {column_clean.max():.2f}]")
        print(f"  ✓ Outliers capped")
        
        return column_clean, {
            'outlier_count': outlier_count,
            'original_min': column_data.min(),
            'original_max': column_data.max(),
            'new_min': column_clean.min(),
            'new_max': column_clean.max()
        }
    else:
        return column_data, {}

# Handle Quantity outliers
df_clean['Quantity'], qty_outliers = handle_column_outliers(df_clean['Quantity'], 'Quantity')

# Handle Price outliers  
df_clean['Price'], price_outliers = handle_column_outliers(df_clean['Price'], 'Price')

if not qty_outliers and not price_outliers:
    print("✓ No significant outliers found")

# ==================== STEP 6: CREATE NEW FEATURES ====================
print("\n🎯 STEP 6: CREATING NEW FEATURES")
print("-" * 40)

# Ensure InvoiceDate is datetime
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'], errors='coerce')

# Calculate transaction value (with cleaned data)
df_clean['TransactionValue'] = df_clean['Quantity'] * df_clean['Price']

# Create time-based features
df_clean['InvoiceYear'] = df_clean['InvoiceDate'].dt.year
df_clean['InvoiceMonth'] = df_clean['InvoiceDate'].dt.month
df_clean['InvoiceDay'] = df_clean['InvoiceDate'].dt.day
df_clean['InvoiceWeekday'] = df_clean['InvoiceDate'].dt.dayofweek  # Monday=0
df_clean['InvoiceHour'] = df_clean['InvoiceDate'].dt.hour

print("Created features:")
print("  • TransactionValue: Quantity × Price")
print("  • InvoiceYear, InvoiceMonth, InvoiceDay")
print("  • InvoiceWeekday (0=Monday, 6=Sunday)")
print("  • InvoiceHour (0-23)")

# ==================== STEP 7: DATA QUALITY CHECK ====================
print("\n✅ STEP 7: DATA QUALITY VALIDATION")
print("-" * 40)

# Check 1: No missing Customer IDs
missing_customers_final = df_clean['Customer ID'].isna().sum()
print(f"1. Missing Customer IDs: {missing_customers_final} (should be 0)")

# Check 2: No negative quantities
negative_qty_final = (df_clean['Quantity'] < 0).sum()
print(f"2. Negative quantities: {negative_qty_final} (should be 0)")

# Check 3: No missing descriptions
missing_desc_final = df_clean['Description'].isna().sum()
print(f"3. Missing descriptions: {missing_desc_final} (should be 0)")

# Check 4: Date range
date_min = df_clean['InvoiceDate'].min()
date_max = df_clean['InvoiceDate'].max()
date_range_days = (date_max - date_min).days
print(f"4. Date range: {date_min.date()} to {date_max.date()} ({date_range_days} days)")

# Check 5: Customer count
unique_customers = df_clean['Customer ID'].nunique()
print(f"5. Unique customers: {unique_customers:,}")

# Check 6: Transaction statistics
total_value = df_clean['TransactionValue'].sum()
avg_value = df_clean['TransactionValue'].mean()
print(f"6. Total transaction value: £{total_value:,.2f}")
print(f"7. Average transaction value: £{avg_value:.2f}")

if missing_customers_final == 0 and negative_qty_final == 0 and missing_desc_final == 0:
    print("\n✅ ALL CHECKS PASSED - DATA IS CLEAN!")
else:
    print("\n⚠ WARNING: Some data quality issues remain")

# ==================== STEP 8: SAVE CLEANED DATA ====================
print("\n💾 STEP 8: SAVING CLEANED DATA")
print("-" * 40)

# Create data directory if needed
os.makedirs("data", exist_ok=True)

# Save cleaned data
cleaned_path = "data/cleaned_online_retail.csv"
df_clean.to_csv(cleaned_path, index=False)

file_size = os.path.getsize(cleaned_path) / (1024**2)  # MB
print(f"✓ Cleaned data saved to: {cleaned_path}")
print(f"  • Rows: {len(df_clean):,}")
print(f"  • Columns: {df_clean.shape[1]}")
print(f"  • File size: {file_size:.2f} MB")

# Save a smaller sample for testing
sample_path = "data/cleaned_sample_10000.csv"
df_clean.head(10000).to_csv(sample_path, index=False)
print(f"✓ Sample saved to: {sample_path} (10,000 rows)")

# ==================== STEP 9: CREATE CLEANING REPORT ====================
print("\n📄 STEP 9: CREATING CLEANING REPORT")
print("-" * 40)

# Create reports directory
os.makedirs("reports", exist_ok=True)

report_path = "reports/cleaning_report.txt"

# Calculate statistics for report
total_rows_removed = rows_removed_customers + rows_removed_returns
final_rows = len(df_clean)

report_content = f"""
{'='*70}
DATA CLEANING REPORT - CUSTOMER SEGMENTATION
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

1. CLEANING SUMMARY
{'-'*40}
Initial rows: {initial_rows:,}
Final rows: {final_rows:,}
Rows removed: {total_rows_removed:,} ({(total_rows_removed/initial_rows*100):.1f}%)

2. CLEANING STEPS PERFORMED
{'-'*40}
1. Missing Customer IDs: Removed {rows_removed_customers:,} rows
   • Reason: Cannot perform customer segmentation without Customer ID

2. Returns (Negative Quantities): Removed {rows_removed_returns:,} rows
   • Reason: Focus on purchase analysis only

3. Missing Descriptions: Filled {missing_desc:,} rows with 'Unknown Product'

4. Outlier Handling:
   • Quantity: {'Capped outliers' if qty_outliers else 'No significant outliers'}
   • Price: {'Capped outliers' if price_outliers else 'No significant outliers'}

3. DATA QUALITY AFTER CLEANING
{'-'*40}
Missing Customer IDs: {missing_customers_final} ✓
Negative Quantities: {negative_qty_final} ✓
Missing Descriptions: {missing_desc_final} ✓
Unique Customers: {unique_customers:,}
Date Range: {date_min.date()} to {date_max.date()}
Transaction Value: £{total_value:,.2f}

4. NEXT STEPS FOR RFM ANALYSIS
{'-'*40}
1. Calculate Recency: Days since last purchase
2. Calculate Frequency: Number of purchases per customer
3. Calculate Monetary: Total spend per customer
4. Create RFM scores for segmentation

5. FILES CREATED
{'-'*40}
• {cleaned_path} - Full cleaned dataset ({final_rows:,} rows)
• {sample_path} - Sample for testing (10,000 rows)
"""

# Save the report
try:
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ Cleaning report saved to: {report_path}")
    print(f"✓ Report size: {os.path.getsize(report_path)/1024:.1f} KB")
    
    # Show summary
    print(f"\n📋 REPORT SUMMARY:")
    print(f"  • Rows removed: {total_rows_removed:,} ({(total_rows_removed/initial_rows*100):.1f}%)")
    print(f"  • Final dataset: {final_rows:,} rows")
    print(f"  • Unique customers: {unique_customers:,}")
    print(f"  • Date range: {date_range_days} days")
    
except Exception as e:
    print(f"❌ Error saving report: {e}")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 70)
print("✅ DATA CLEANING COMPLETE!")
print("=" * 70)

print(f"\nCLEANING RESULTS:")
print(f"• Initial: {initial_rows:,} rows")
print(f"• Final:   {final_rows:,} rows")
print(f"• Removed: {total_rows_removed:,} rows ({(total_rows_removed/initial_rows*100):.1f}%)")
print(f"• Unique customers: {unique_customers:,}")

print(f"\nNEXT STEPS:")
print(f"1. Review cleaning report: {report_path}")
print(f"2. Check cleaned data: {cleaned_path}")
print(f"3. Run 03_rfm_analysis.py for customer segmentation")

# Show first few rows of cleaned data
print("\n" + "-" * 70)
print("FIRST 3 ROWS OF CLEANED DATA:")
print("-" * 70)
print(df_clean.head(3))