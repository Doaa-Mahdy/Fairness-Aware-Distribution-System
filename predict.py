import os
import json
import numpy as np
import xgboost as xgb
from stable_baselines3 import PPO
from datetime import datetime

# --- HELPER FUNCTIONS ---
def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj

# --- MODEL LOADING ---
MODEL_PATH = "models/fairness_rl_model.zip"
XGB_MODEL_PATH = "models/charity_xgboost.json"

RL_MODEL = PPO.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
XGB_MODEL = xgb.Booster()
if os.path.exists(XGB_MODEL_PATH):
    XGB_MODEL.load_model(XGB_MODEL_PATH)

def build_env_features(recipient_data: dict) -> list:
    """Maps incoming JSON to env feature order used during training."""
    case = recipient_data.get("CaseMetadata", {})
    demo = recipient_data.get("Demographics", {})
    med = recipient_data.get("MedicalProfile", {})
    house = recipient_data.get("HousingAndLiving", {})
    fin = recipient_data.get("FinancialLiquidity", {})
    hist = recipient_data.get("FinancialHistory", {})

    return [
        case.get("Status", 0),
        case.get("ReopenedCount", 0),
        1 if case.get("IsActive", True) else 0,
        demo.get("FamilySize", 1),
        demo.get("DeceasedParentCount", 0),
        demo.get("EducationBurden", 0),
        demo.get("MaritalVulnerability", 0),
        med.get("DisabilityWeight", 0),
        med.get("ChronicConditionWeight", 0),
        1 if med.get("RequiresUrgentCare", False) else 0,
        med.get("MedicationCount", 0),
        1 if house.get("IsRented", False) else 0,
        house.get("MonthlyRent", 0),
        house.get("InfrastructureDeficit", 0),
        1 if house.get("HasElectricity", True) else 0,
        house.get("OvercrowdingRatio", 1.0),
        fin.get("CurrentCardBalance", 0),
        fin.get("CardStatus", 0),
        hist.get("TotalReceivedLastMonth", 0),
        recipient_data.get("xgboost_suggestion", 0)
    ]

def predict_from_payload(payload):
    params = payload.get("params", {})
    global_budget = float(params.get("budget", 0))
    min_alloc = float(params.get("min_allocation", 50.0))
    max_alloc_default = float(params.get("max_allocation", global_budget))
    min_people = int(params.get("min_people_to_help", 1))
    
    raw_data = payload.get("data", [])
    scored_recipients = []
    
    # --- STEP 1: SCORING ---
    for rec in raw_data:
        features = build_env_features(rec)
        # Assuming XGB model handles the 19 features (excluding xgb_suggestion)
        xgb_val = (float(XGB_MODEL.predict(xgb.DMatrix(np.array([features[:-1]])))[0])/10) if XGB_MODEL else 0.0
        rec["xgboost_suggestion"] = float(xgb_val)
        scored_recipients.append({"data": rec, "xgb_score": float(xgb_val), "final_allocation": 0.0})

    # Sort by priority (XGB Score) so we help the most vulnerable first
    scored_recipients.sort(key=lambda x: x['xgb_score'], reverse=True)

    # --- STEP 2: PASS 1 (Initial RL Guess) ---
    remaining_budget = global_budget
    for item in scored_recipients:
        rec_data = item["data"]
        # Build state
        features = np.array(build_env_features(rec_data), dtype=np.float32)
        # Match env observation: features (20) + [B_max, min_alloc, max_alloc]
        constraints = np.array([global_budget, min_alloc, max_alloc_default], dtype=np.float32)
        state = np.concatenate([features, constraints])

        if RL_MODEL:
            action, _ = RL_MODEL.predict(state, deterministic=True)
            # Map [-1, 1] to [min_alloc, max_alloc]
            alloc = min_alloc + (action[0] + 1) * 0.5 * (max_alloc_default - min_alloc)
            alloc = float(alloc)  # Convert numpy to Python float
        else:
            alloc = min_alloc

        # Initial constraint check
        alloc = float(np.clip(alloc, min_alloc, max_alloc_default))
        if remaining_budget >= min_alloc:
            actual = float(min(alloc, remaining_budget))
            item["final_allocation"] = actual
            remaining_budget -= actual
        else:
            item["final_allocation"] = 0.0 # Not enough budget left for this person

    # --- STEP 3: NEED-BASED SURPLUS DISTRIBUTION ---
    # Distribute remaining budget proportionally to vulnerability scores
    max_iterations = 10
    iteration = 0
    while remaining_budget > 1 and iteration < max_iterations:
        iteration += 1
        distributed_this_round = 0
        
        # Calculate total need (sum of scores for those with room to grow)
        eligible = [item for item in scored_recipients if item["final_allocation"] >= 0 and (max_alloc_default - item["final_allocation"]) > 1]
        if not eligible:
            break
        
        total_need = sum(item["xgb_score"] for item in eligible)
        if total_need <= 0:
            break
        
        for item in eligible:
            if remaining_budget <= 1:
                break
            
            current = item["final_allocation"]
            potential_extra = max_alloc_default - current
            
            # Allocate proportional to their need (xgb_score / total_need)
            need_ratio = item["xgb_score"] / total_need
            share = min(potential_extra, remaining_budget * need_ratio)
            
            if share > 1:
                item["final_allocation"] = float(item["final_allocation"] + share)
                remaining_budget = float(remaining_budget - share)
                distributed_this_round += share
        
        if distributed_this_round < 1:
            break

    # --- STEP 4: FINAL ASSEMBLY ---
    final_output = []
    cases_served = 0
    for item in scored_recipients:
        alloc = float(round(item["final_allocation"], 2))
        if alloc >= min_alloc: cases_served += 1
        final_output.append({
            "RecipientId": item["data"].get("RecipientId"),
            "xgb_reference": float(round(item["xgb_score"], 2)),
            "rl_allocation": alloc,
            "met_min": bool(alloc >= min_alloc)
        })

    return {
        "allocations": final_output,
        "summary": {
            "total_budget": float(global_budget),
            "total_allocated": float(round(global_budget - remaining_budget, 2)),
            "people_helped": int(cases_served),
            "min_target_met": bool(cases_served >= min_people)
        },
        "timestamp": datetime.now().isoformat()
    }

def predict_from_payload_serializable(payload):
    """Wrapper that ensures output is JSON-serializable"""
    result = predict_from_payload(payload)
    return convert_to_serializable(result)
if __name__ == "__main__":
    # Test with your provided JSON data
    with open("in/example_predict.json", "r") as f:
        data = json.load(f)
    print(json.dumps(predict_from_payload(data), indent=2))