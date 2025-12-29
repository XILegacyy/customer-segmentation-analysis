"""
03_rfm_analysis.py - RFM Analysis for Customer Segmentation

Calculates Recency, Frequency, and Monetary values for each customer
and creates RFM segments for targeted marketing.

Author: Awande Gcabashe
Date: 12/29/2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

print("=" * 70)
print("CUSTOMER SEGMENTATION PROJECT - RFM ANALYSIS")
print("=" * 70)

# ==================== STEP 1: LOAD CLEANED DATA ====================
print("\n📁 STEP 1: LOADING CLEANED DATA")
print("-" * 40)

def load_cleaned_data():
    """Load the cleaned dataset from CSV."""
    data_path = "data/cleaned_online_retail.csv"
    
    if not os.path.exists(data_path):
        print(f"❌ ERROR: File not found: {data_path}")
        print("Run 02_cleaning.py first to create cleaned data")
        return None
    
    print(f"Loading cleaned data from: {data_path}")
    
    try:
        # Load the data
        df = pd.read_csv(data_path, parse_dates=['InvoiceDate'])
        
        print(f"✓ Loaded {len(df):,} transactions")
        print(f"✓ Unique customers: {df['Customer ID'].nunique():,}")
        print(f"✓ Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return None

# Load the data
df = load_cleaned_data()
if df is None:
    exit(1)

# ==================== STEP 2: CALCULATE RFM METRICS ====================
print("\n📊 STEP 2: CALCULATING RFM METRICS")
print("-" * 40)

# Define snapshot date (day after last transaction for recency calculation)
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Snapshot date for recency calculation: {snapshot_date.date()}")

# Group by Customer ID to calculate RFM
print("\nCalculating customer-level metrics...")

rfm_data = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
    'Invoice': 'nunique',  # Frequency (unique invoices)
    'TransactionValue': 'sum'  # Monetary
}).reset_index()

# Rename columns
rfm_data.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

print(f"✓ Calculated RFM for {len(rfm_data):,} customers")
print("\nRFM Metrics Overview:")
print(f"  • Recency: Days since last purchase")
print(f"  • Frequency: Number of unique purchases")
print(f"  • Monetary: Total spend (£)")

# ==================== STEP 3: ANALYZE RFM DISTRIBUTIONS ====================
print("\n📈 STEP 3: ANALYZING RFM DISTRIBUTIONS")
print("-" * 40)

print("RFM Statistics:")
print(rfm_data[['Recency', 'Frequency', 'Monetary']].describe())

print("\nKey Insights:")
print(f"1. Recency range: {rfm_data['Recency'].min()} to {rfm_data['Recency'].max()} days")
print(f"   • Lower = more recent (better)")
print(f"   • Average: {rfm_data['Recency'].mean():.1f} days")
print(f"   • Median: {rfm_data['Recency'].median():.0f} days")

print(f"\n2. Frequency range: {rfm_data['Frequency'].min()} to {rfm_data['Frequency'].max()} purchases")
print(f"   • Higher = more frequent (better)")
print(f"   • Average: {rfm_data['Frequency'].mean():.1f} purchases")
print(f"   • 75% of customers have ≤ {rfm_data['Frequency'].quantile(0.75):.0f} purchases")

print(f"\n3. Monetary range: £{rfm_data['Monetary'].min():.2f} to £{rfm_data['Monetary'].max():.2f}")
print(f"   • Higher = more valuable (better)")
print(f"   • Average spend: £{rfm_data['Monetary'].mean():.2f}")
print(f"   • Total customer value: £{rfm_data['Monetary'].sum():,.2f}")

# ==================== STEP 4: CREATE RFM SCORES ====================
print("\n🎯 STEP 4: CREATING RFM SCORES (1-4)")
print("-" * 40)

print("Scoring logic (1=Lowest, 4=Highest):")
print("  • Recency: Lower days = HIGHER score (4)")
print("  • Frequency: Higher count = HIGHER score (4)")
print("  • Monetary: Higher value = HIGHER score (4)")

# Create quantile-based scores (4 groups each)
rfm_data['R_Score'] = pd.qcut(rfm_data['Recency'], q=4, labels=[4, 3, 2, 1])
rfm_data['F_Score'] = pd.qcut(rfm_data['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4])
rfm_data['M_Score'] = pd.qcut(rfm_data['Monetary'].rank(method='first'), q=4, labels=[1, 2, 3, 4])

# Convert to numeric
rfm_data['R_Score'] = rfm_data['R_Score'].astype(int)
rfm_data['F_Score'] = rfm_data['F_Score'].astype(int)
rfm_data['M_Score'] = rfm_data['M_Score'].astype(int)

# Create combined RFM score (concatenate as string)
rfm_data['RFM_Score'] = rfm_data['R_Score'].astype(str) + rfm_data['F_Score'].astype(str) + rfm_data['M_Score'].astype(str)

print("\nScore Distribution:")
print(f"Recency Scores (R):")
print(rfm_data['R_Score'].value_counts().sort_index())
print(f"\nFrequency Scores (F):")
print(rfm_data['F_Score'].value_counts().sort_index())
print(f"\nMonetary Scores (M):")
print(rfm_data['M_Score'].value_counts().sort_index())

# ==================== STEP 5: CREATE RFM SEGMENTS ====================
print("\n🏷️ STEP 5: CREATING CUSTOMER SEGMENTS")
print("-" * 40)

def assign_segment(row):
    """Assign customer segment based on RFM scores."""
    # Champions: Best customers (high on all dimensions)
    if row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
        return 'Champion'
    
    # Loyal Customers: Frequent but might not be recent or big spenders
    elif row['F_Score'] >= 3:
        if row['R_Score'] >= 3:
            return 'Loyal Customer'
        else:
            return 'At Risk Loyal'
    
    # Recent Customers: Bought recently but not frequently
    elif row['R_Score'] >= 3:
        if row['F_Score'] == 1:
            return 'New Customer'
        else:
            return 'Potential Loyalist'
    
    # Sleeping Customers: Haven't bought in a while
    elif row['R_Score'] == 1:
        return 'Hibernating'
    
    # At Risk: Used to purchase but not recently
    elif row['R_Score'] == 2:
        return 'At Risk'
    
    # Can't Lose: Big spenders but not recent
    elif row['M_Score'] >= 3 and row['R_Score'] <= 2:
        return 'Cant Lose'
    
    # Promising: Recent but low frequency and monetary
    elif row['R_Score'] >= 3 and row['F_Score'] <= 2 and row['M_Score'] <= 2:
        return 'Promising'
    
    # Need Attention: Middle of the road
    else:
        return 'Need Attention'

# Apply segmentation
rfm_data['Segment'] = rfm_data.apply(assign_segment, axis=1)

print("Customer Segments Created:")
segment_counts = rfm_data['Segment'].value_counts()
for segment, count in segment_counts.items():
    pct = (count / len(rfm_data)) * 100
    print(f"  • {segment:<20}: {count:>5,} customers ({pct:.1f}%)")

# ==================== STEP 6: SAVE RFM RESULTS ====================
print("\n💾 STEP 6: SAVING RFM RESULTS")
print("-" * 40)

# Create directories if needed
os.makedirs("data", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Save RFM data
rfm_path = "data/rfm_data.csv"
rfm_data.to_csv(rfm_path, index=False)

print(f"✓ RFM data saved to: {rfm_path}")
print(f"  • Customers: {len(rfm_data):,}")
print(f"  • Columns: {rfm_data.shape[1]}")

# Save segment summary
segment_summary = rfm_data.groupby('Segment').agg({
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'R_Score': 'mean',
    'F_Score': 'mean',
    'M_Score': 'mean'
}).round(2)

segment_summary.columns = ['Count', 'Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 
                           'Avg_R_Score', 'Avg_F_Score', 'Avg_M_Score']
segment_summary = segment_summary.sort_values('Count', ascending=False)

segment_path = "data/segment_summary.csv"
segment_summary.to_csv(segment_path)

print(f"✓ Segment summary saved to: {segment_path}")

# ==================== STEP 7: CREATE RFM REPORT ====================
print("\n📄 STEP 7: CREATING RFM ANALYSIS REPORT")
print("-" * 40)

report_path = "reports/rfm_analysis_report.txt"

report_content = f"""
{'='*70}
RFM ANALYSIS REPORT - CUSTOMER SEGMENTATION
{'='*70}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Customers Analyzed: {len(rfm_data):,}

1. RFM METRICS SUMMARY
{'-'*40}
Recency (Days since last purchase):
  • Range: {rfm_data['Recency'].min()} to {rfm_data['Recency'].max()} days
  • Average: {rfm_data['Recency'].mean():.1f} days
  • Median: {rfm_data['Recency'].median():.0f} days

Frequency (Number of purchases):
  • Range: {rfm_data['Frequency'].min()} to {rfm_data['Frequency'].max()}
  • Average: {rfm_data['Frequency'].mean():.1f} purchases
  • 75th percentile: {rfm_data['Frequency'].quantile(0.75):.0f} purchases

Monetary (Total spend):
  • Range: £{rfm_data['Monetary'].min():.2f} to £{rfm_data['Monetary'].max():.2f}
  • Average: £{rfm_data['Monetary'].mean():.2f}
  • Total customer value: £{rfm_data['Monetary'].sum():,.2f}

2. CUSTOMER SEGMENTS
{'-'*40}
"""

for segment, count in segment_counts.items():
    pct = (count / len(rfm_data)) * 100
    segment_data = rfm_data[rfm_data['Segment'] == segment]
    
    report_content += f"\n{segment}:"
    report_content += f"\n  • Count: {count:,} ({pct:.1f}%)"
    report_content += f"\n  • Avg Recency: {segment_data['Recency'].mean():.1f} days"
    report_content += f"\n  • Avg Frequency: {segment_data['Frequency'].mean():.1f}"
    report_content += f"\n  • Avg Monetary: £{segment_data['Monetary'].mean():.2f}"
    report_content += f"\n  • Typical RFM Score: {segment_data['RFM_Score'].mode().iloc[0] if not segment_data['RFM_Score'].mode().empty else 'N/A'}"

report_content += f"""
\n3. SEGMENT INTERPRETATION & ACTION PLAN
{'-'*40}
1. CHAMPIONS ({segment_counts.get('Champion', 0):,} customers)
   • Action: Reward them, ask for reviews, offer loyalty programs

2. LOYAL CUSTOMERS ({segment_counts.get('Loyal Customer', 0):,} customers)
   • Action: Upsell, offer member-only benefits

3. AT RISK ({segment_counts.get('At Risk', 0):,} customers)
   • Action: Win back with special offers, re-engagement campaigns

4. NEW CUSTOMERS ({segment_counts.get('New Customer', 0):,} customers)
   • Action: Welcome series, onboarding, first-purchase follow-up

5. HIBERNATING ({segment_counts.get('Hibernating', 0):,} customers)
   • Action: Reactivation campaigns, win-back offers

6. PROMISING ({segment_counts.get('Promising', 0):,} customers)
   • Action: Encourage repeat purchases, recommend products

4. FILES GENERATED
{'-'*40}
• {rfm_path} - Complete RFM data for all customers
• {segment_path} - Segment summary with averages
"""

# Save the report
try:
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"✓ RFM report saved to: {report_path}")
    print(f"✓ Report size: {os.path.getsize(report_path)/1024:.1f} KB")
    
except Exception as e:
    print(f"❌ Error saving report: {e}")

# ==================== STEP 8: DISPLAY KEY INSIGHTS ====================
print("\n🔍 STEP 8: KEY BUSINESS INSIGHTS")
print("-" * 40)

# Top customers by monetary value
top_10_customers = rfm_data.nlargest(10, 'Monetary')[['CustomerID', 'Monetary', 'Segment']]
print("Top 10 Customers by Spend:")
for idx, row in top_10_customers.iterrows():
    print(f"  • Customer {row['CustomerID']}: £{row['Monetary']:,.2f} ({row['Segment']})")

# Segment with highest average spend
segment_avg_spend = rfm_data.groupby('Segment')['Monetary'].mean().sort_values(ascending=False)
print(f"\nSegments by Average Spend:")
for segment, avg_spend in segment_avg_spend.head(5).items():
    print(f"  • {segment}: £{avg_spend:,.2f}")

# Customers needing attention
at_risk_count = segment_counts.get('At Risk', 0) + segment_counts.get('Cant Lose', 0)
print(f"\n⚠ Customers Needing Attention: {at_risk_count:,}")
if at_risk_count > 0:
    print(f"  • Potential revenue at risk")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 70)
print("✅ RFM ANALYSIS COMPLETE!")
print("=" * 70)

print(f"\nRESULTS SUMMARY:")
print(f"• Total customers analyzed: {len(rfm_data):,}")
print(f"• Segments created: {len(segment_counts)}")
print(f"• Top segment: {segment_counts.index[0]} ({segment_counts.iloc[0]:,} customers)")
print(f"• Total customer value: £{rfm_data['Monetary'].sum():,.2f}")

print(f"\nFILES CREATED:")
print(f"1. {rfm_path} - RFM data for {len(rfm_data):,} customers")
print(f"2. {segment_path} - Segment summary")
print(f"3. {report_path} - Detailed analysis report")

print(f"\nNEXT STEPS:")
print(f"1. Review the RFM report: {report_path}")
print(f"2. Run 04_clustering.py for advanced segmentation using K-Means")
print(f"3. Create targeted marketing campaigns for each segment")

# Show sample of RFM data
print("\n" + "-" * 70)
print("SAMPLE OF RFM DATA (First 5 customers):")
print("-" * 70)
print(rfm_data.head())