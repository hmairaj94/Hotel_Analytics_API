import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def generate_analytics(data_path):
    """
    Generate analytics from hotel booking data
    
    Args:
        data_path: Path to the CSV file with hotel booking data
    
    Returns:
        JSON with analytics data
    """

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    df['month_year'] = df['reservation_status_date'].dt.strftime('%Y-%m')
    df['revenue'] = df.apply(
        lambda row: row['adr'] * (row['stays_in_weekend_nights'] + row['stays_in_week_nights']) 
        if row['is_canceled'] == 0 else 0, 
        axis=1
    )
    
    # 1. Revenue trends over time
    revenue_by_month = df.groupby('month_year')['revenue'].sum()
    
    plt.figure(figsize=(12, 6))
    revenue_by_month.plot(kind='bar')
    plt.title('Revenue Trends by Month')
    plt.xlabel('Month')
    plt.ylabel('Revenue ($)')
    plt.tight_layout()
    plt.savefig('visualizations/revenue_trends.png')
    plt.close()
    
    # 2. Cancellation rate
    total_bookings = len(df)
    cancelled_bookings = df['is_canceled'].sum()
    cancellation_rate = (cancelled_bookings / total_bookings) * 100
    
    plt.figure(figsize=(8, 8))
    plt.pie([cancelled_bookings, total_bookings - cancelled_bookings], 
            labels=['Cancelled', 'Completed'], 
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff'])
    plt.title('Booking Cancellation Rate')
    plt.savefig('visualizations/cancellation_rate.png')
    plt.close()
    
    # 3. Geographical distribution (Cancellations by country)
    cancellation_by_country = df[df['is_canceled'] == 1].groupby('country').size()
    cancellation_by_country = cancellation_by_country.sort_values(ascending=False)
    
    # Visualization (top 10 countries)
    plt.figure(figsize=(12, 6))
    cancellation_by_country.head(10).plot(kind='bar')
    plt.title('Top 10 Countries by Cancellation Count')
    plt.xlabel('Country')
    plt.ylabel('Number of Cancellations')
    plt.tight_layout()
    plt.savefig('visualizations/cancellation_by_country.png')
    plt.close()
    
    # 4. Booking Lead time distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(df['lead_time'], bins=50)
    plt.title('Booking Lead Time Distribution')
    plt.xlabel('Lead Time (days)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('visualizations/lead_time_distribution.png')
    plt.close()
    
    # 5. Additional Analytics - Average Daily Rate by Hotel Type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='hotel', y='adr', data=df)
    plt.title('Average Daily Rate by Hotel Type')
    plt.xlabel('Hotel Type')
    plt.ylabel('ADR ($)')
    plt.tight_layout()
    plt.savefig('visualizations/adr_by_hotel.png')
    plt.close()
    
    # 6. Stay Duration Analysis
    df['total_nights'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
    plt.figure(figsize=(12, 6))
    sns.histplot(df['total_nights'][df['total_nights'] < 15], bins=15)  # Filter outliers
    plt.title('Distribution of Stay Duration')
    plt.xlabel('Number of Nights')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('visualizations/stay_duration.png')
    plt.close()
    
    analytics_data = {
        "revenue_trends": revenue_by_month.to_dict(),
        "cancellation_rate": float(cancellation_rate),
        "cancellation_by_country": cancellation_by_country.to_dict(),
        "average_lead_time": float(df['lead_time'].mean()),
        "lead_time_median": float(df['lead_time'].median()),
        "lead_time_percentiles": {
            "25%": float(df['lead_time'].quantile(0.25)),
            "50%": float(df['lead_time'].quantile(0.5)),
            "75%": float(df['lead_time'].quantile(0.75)),
            "90%": float(df['lead_time'].quantile(0.9))
        },
        "average_adr": float(df['adr'].mean()),
        "total_revenue": float(df['revenue'].sum()),
        "average_stay_duration": float(df['total_nights'].mean()),
        "total_bookings": int(total_bookings),
        "total_cancellations": int(cancelled_bookings),
        "generation_timestamp": datetime.now().isoformat()
    }
    
    with open('precomputed_analytics.json', 'w') as f:
        json.dump(analytics_data, f, indent=2)
    
    print("Analytics generated successfully and saved to precomputed_analytics.json")
    print(f"Visualizations saved to the 'visualizations' directory")
    
    return analytics_data

if __name__ == "__main__":
    import os
    
    os.makedirs('visualizations', exist_ok=True)
    
    analytics = generate_analytics('cleaned_hotel_bookings.csv')
    
    print("\nAnalytics Summary:")
    print(f"Total Bookings: {analytics['total_bookings']}")
    print(f"Cancellation Rate: {analytics['cancellation_rate']:.2f}%")
    print(f"Average ADR: ${analytics['average_adr']:.2f}")
    print(f"Total Revenue: ${analytics['total_revenue']:.2f}")
    print(f"Average Lead Time: {analytics['average_lead_time']:.2f} days")