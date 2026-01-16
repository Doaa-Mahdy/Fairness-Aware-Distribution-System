import pandas as pd
import datetime
import os

FEEDBACK_LOG = "database/live_experience.csv"

import pandas as pd
import datetime
import os

FEEDBACK_LOG = "database/live_experience.csv"

def log_human_edit(recipient_id, ai_suggested, human_edited, features, group_id=1, max_budget=10000, min_allocation=50, max_allocation=1000, min_cases=5):
    """
    Captures the 'Lesson' from the human worker.
    Features should be a dict with keys matching the JSON format.
    """
    # Ensure database directory exists
    os.makedirs(os.path.dirname(FEEDBACK_LOG), exist_ok=True)
    
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "recipientid": recipient_id,
        # Map from JSON keys to CSV column names
        "case_status": features.get("Case_Status", 0),
        "case_reopened": features.get("Case_Reopened", 0),
        "case_isactive": features.get("Case_IsActive", 1),
        "demo_familysize": features.get("Demo_FamilySize", 1),
        "demo_deceasedcount": features.get("Demo_DeceasedCount", 0),
        "demo_eduburden": features.get("Demo_EduBurden", 0),
        "demo_maritalvuln": features.get("Demo_MaritalVuln", 0),
        "med_disability": features.get("Med_Disability", 0),
        "med_chronic": features.get("Med_Chronic", 0),
        "med_urgent": features.get("Med_Urgent", 0),
        "med_count": features.get("Med_Count", 0),
        "house_isrented": features.get("House_IsRented", 0),
        "house_rent": features.get("House_Rent", 0),
        "house_infra": features.get("House_Infra", 0),
        "house_elec": features.get("House_Elec", 1),
        "house_ratio": features.get("House_Ratio", 1.0),
        "fin_balance": features.get("Fin_Balance", 0),
        "fin_status": features.get("Fin_Status", 0),
        "hist_lastmonth": features.get("Hist_LastMonth", 0),
        "xgboost_suggestion": features.get("XGBoost_Suggestion", ai_suggested),
        # Add required group columns
        "group_id": group_id,
        "max_budget": max_budget,
        "min_allocation": min_allocation,
        "max_allocation": max_allocation,
        "min_cases": min_cases,
        "amount_allocated": human_edited  # Use human edit as the training target
    }
    
    df = pd.DataFrame([entry])
    # Append to the log file that train_live.py will read
    df.to_csv(FEEDBACK_LOG, mode='a', header=not os.path.exists(FEEDBACK_LOG), index=False)
    
    return entry
