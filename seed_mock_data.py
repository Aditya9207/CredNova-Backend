import os
import pymongo
from bson import ObjectId

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/crednova")
client = pymongo.MongoClient(MONGO_URI)
db = client["crednova"]
applications_coll = db["credit_applications"]

# 1. Define rich mock statement analysis
MOCK_STATEMENT_ANALYSIS = {
    "available": True,
    "monthly": [
        {"key": "2025-03", "label": "Mar 2025", "credits": 88500.0, "debits": 64200.0},
        {"key": "2025-04", "label": "Apr 2025", "credits": 92000.0, "debits": 71800.0},
        {"key": "2025-05", "label": "May 2025", "credits": 89000.0, "debits": 62400.0},
        {"key": "2025-06", "label": "Jun 2025", "credits": 95000.0, "debits": 68900.0},
        {"key": "2025-07", "label": "Jul 2025", "credits": 91500.0, "debits": 65300.0},
        {"key": "2025-08", "label": "Aug 2025", "credits": 98000.0, "debits": 74100.0},
    ],
    "totals": {
        "total_credits_inr": 554000.0,
        "total_debits_inr": 406700.0,
        "net_savings_inr": 147300.0,
        "months_in_chart": 6,
    },
    "balance": {
        "peak_inr": 184500.0,
        "peak_month_label": "Aug 2025",
        "closing_inr": 158200.0,
    },
    "rental": {
        "credits_inr": 25000.0,
        "months_with_rent": 6,
        "avg_monthly_inr": 25000.0,
    },
    "insights": [
        "Healthy monthly savings rate averaging 26.5% of total inflow over 6 months.",
        "Consistent primary salary/business inflow detected on the 1st week of each month.",
        "UPI velocity demonstrates high active digital footprint (average 42 transactions/month).",
        "No overdraft fees or negative balance dips recorded across all 6 reporting periods.",
        "Rental income stability observed with 6 recurring credits averaging \u20b925,000.",
        "Low cash withdrawal ratio (< 15%), indicating verifiable digital expenditure trails.",
    ],
}

# 2. Rich mock CSV with realistic transactions that match _SPEND_RULES in insights_service
MOCK_TRANSACTIONS_CSV = """date,remarks,debits,credits,balance
01-03-2025,Salary Credit - Infosys Ltd,0.0,88500.0,115000.0
03-03-2025,Tata Power Electricity Bill,4200.0,0.0,110800.0
05-03-2025,Swiggy Bangalore Order,1250.0,0.0,109550.0
08-03-2025,Amazon Retail India Shopping,5800.0,0.0,103750.0
12-03-2025,Uber India Commute,650.0,0.0,103100.0
15-03-2025,Zerodha Broking SIP Investment,10000.0,0.0,93100.0
18-03-2025,Apollo Pharmacy Medical,1450.0,0.0,91650.0
22-03-2025,Netflix Entertainment Subscription,649.0,0.0,91001.0
25-03-2025,Blinkit Daily Grocery,2800.0,0.0,88201.0
28-03-2025,HPCL Petrol Fuel Station,2500.0,0.0,85701.0
01-04-2025,Salary Credit - Infosys Ltd,0.0,92000.0,177701.0
04-04-2025,Bescom Electric Bill,3800.0,0.0,173901.0
07-04-2025,Zomato Online Delivery,1400.0,0.0,172501.0
10-04-2025,Flipkart Online Electronics,12500.0,0.0,160001.0
14-04-2025,Ola Cabs Ride,550.0,0.0,159451.0
18-04-2025,Groww Mutual Fund SIP,10000.0,0.0,149451.0
22-04-2025,BookMyShow Movie Tickets,1100.0,0.0,148351.0
26-04-2025,BigBasket Monthly Supermarket,4200.0,0.0,144151.0
01-05-2025,Salary Credit - Infosys Ltd,0.0,89000.0,233151.0
03-05-2025,Airtel Broadband Fiber Internet,1499.0,0.0,231652.0
06-05-2025,Domino's Pizza Dining,950.0,0.0,230702.0
11-05-2025,Myntra Fashion Apparel,3600.0,0.0,227102.0
15-05-2025,Zerodha Broking SIP Investment,10000.0,0.0,217102.0
20-05-2025,BPCL Petrol Pump Fuel,2800.0,0.0,214302.0
25-05-2025,1mg Healthcare Diagnostic,2100.0,0.0,212202.0
01-06-2025,Salary Credit - Infosys Ltd,0.0,95000.0,307202.0
05-06-2025,Rental Income Credit Flat 402,0.0,25000.0,332202.0
08-06-2025,Tata Power Electricity,4100.0,0.0,328102.0
12-06-2025,Swiggy Dineout Restaurant,3200.0,0.0,324902.0
16-06-2025,Croma Electronics Appliance,8400.0,0.0,316502.0
20-06-2025,Uber Airport Trip,1200.0,0.0,315302.0
25-06-2025,Spotify Annual Subscription,1189.0,0.0,314113.0
01-07-2025,Salary Credit - Infosys Ltd,0.0,91500.0,405613.0
05-07-2025,Rental Income Credit Flat 402,0.0,25000.0,430613.0
09-07-2025,Amazon Prime Shopping,4600.0,0.0,426013.0
14-07-2025,KFC Food Order,890.0,0.0,425123.0
18-07-2025,Zerodha Broking SIP Investment,10000.0,0.0,415123.0
24-07-2025,Indian Oil Petrol,2400.0,0.0,412723.0
01-08-2025,Salary Credit - Infosys Ltd,0.0,98000.0,510723.0
05-08-2025,Rental Income Credit Flat 402,0.0,25000.0,535723.0
07-08-2025,Gas & Utility Pipeline Bill,1650.0,0.0,534073.0
12-08-2025,Swiggy Food Gourmet,2100.0,0.0,531973.0
16-08-2025,Ajio Fashion Shopping,5400.0,0.0,526573.0
20-08-2025,PVR Cinemas Gold Class,1800.0,0.0,524773.0
25-08-2025,PharmEasy Medicines,1600.0,0.0,523173.0
"""

def seed():
    # Update latest application 6ad5f236e815f3434965ed1e
    target_id = ObjectId("6ad5f236e815f3434965ed1e")
    doc = applications_coll.find_one({"_id": target_id})
    if doc:
        print("Found target application 6ad5f236e815f3434965ed1e, updating with mock data...")
        applications_coll.update_one(
            {"_id": target_id},
            {
                "$set": {
                    "statement.analysis": MOCK_STATEMENT_ANALYSIS,
                    "transactions_csv": MOCK_TRANSACTIONS_CSV,
                    "transactions_csv_available": True,
                    "statement.metrics.statement_row_count": 45,
                    "statement.metrics.monthly_upi": 42.0,
                    "statement.metrics.cash_transaction_ratio": 0.08,
                    "statement.metrics.months_analyzed": 6.0,
                    "statement.row_count": 45,
                }
            }
        )
        print("Target application updated successfully!")

    # Also update any applications with missing or unavailable statement analysis
    cursor = applications_coll.find({
        "$or": [
            {"statement.analysis": {"$exists": False}},
            {"statement.analysis.available": False},
            {"statement.analysis.monthly": {"$size": 0}},
        ]
    })
    count = 0
    for app in cursor:
        applications_coll.update_one(
            {"_id": app["_id"]},
            {
                "$set": {
                    "statement.analysis": MOCK_STATEMENT_ANALYSIS,
                    "transactions_csv": MOCK_TRANSACTIONS_CSV,
                    "transactions_csv_available": True,
                }
            }
        )
        count += 1
    print(f"Updated {count} applications with mock analysis and transactions.")

if __name__ == "__main__":
    seed()
