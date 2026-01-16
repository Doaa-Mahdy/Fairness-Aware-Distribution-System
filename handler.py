"""
RunPod Serverless Handler for Fairness-Aware RL Allocation System
Supports: predict, feedback logging, and live model training
"""

import json
import traceback
import os
from predict import predict_from_payload
from feedback import log_human_edit
from train_live import update_model
import runpod


# ======================================================
# SERVERLESS HANDLER (PREDICT / FEEDBACK / TRAIN)
# ======================================================
def handler(job):
    """
    RunPod Serverless entry point
    
    Expected request formats:
    
    1. PREDICT:
    {
      "input": {
        "operation": "predict",
        "params": {
          "budget": 5000,
          "min_allocation": 50.0,
          "max_allocation": 1000.0,
          "min_people_to_help": 5
        },
        "data": [
          {
            "RecipientId": "R1",
            "CaseMetadata": {...},
            "Demographics": {...},
            ...
          }
        ]
      }
    }
    
    2. FEEDBACK:
    {
      "input": {
        "operation": "feedback",
        "run_id": "...",
        "group_id": 1,
        "max_budget": 10000.0,
        "min_allocation": 200.0,
        "max_allocation": 800.0,
        "min_cases": 8,
        "edits": [
          {
            "RecipientId": "R1",
            "Human_Final_Value": 550.0,
            "AI_Suggested_Value": 408.57,
            "features": {...}
          }
        ]
      }
    }
    
    3. TRAIN:
    {
      "input": {
        "operation": "train"
      }
    }
    """
    try:
        job_input = job.get("input", {}) if isinstance(job, dict) else {}
        
        operation = job_input.get("operation", "predict").lower()
        
        # ----------------------------------------------
        # OPERATION: PREDICT
        # ----------------------------------------------
        if operation == "predict":
            return _handle_predict(job_input)
        
        # ----------------------------------------------
        # OPERATION: FEEDBACK
        # ----------------------------------------------
        elif operation == "feedback":
            return _handle_feedback(job_input)
        
        # ----------------------------------------------
        # OPERATION: TRAIN
        # ----------------------------------------------
        elif operation == "train":
            return _handle_train(job_input)
        
        else:
            return _error(f"Invalid operation: {operation}. Use: predict | feedback | train")
    
    except Exception as e:
        return {
            "statusCode": 500,
            "error": str(e),
            "trace": traceback.format_exc()
        }


# ======================================================
# PREDICT HANDLER
# ======================================================
def _handle_predict(job_input):
    """
    Process allocation prediction request
    """
    try:
        # Validate input structure
        if "params" not in job_input or "data" not in job_input:
            return _error("Missing 'params' or 'data' field for predict operation")
        
        params = job_input["params"]
        recipients = job_input["data"]
        
        # Validate params
        required_params = ["budget", "min_allocation", "max_allocation", "min_people_to_help"]
        missing = [p for p in required_params if p not in params]
        if missing:
            return _error(f"Missing required params: {missing}")
        
        if not isinstance(recipients, list) or len(recipients) == 0:
            return _error("Data must be a non-empty list of recipients")
        
        # Call prediction pipeline
        results = predict_from_payload(job_input)
        
        return {
            "statusCode": 200,
            "operation": "predict",
            "count": len(recipients),
            "results": results
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "operation": "predict",
            "error": str(e),
            "trace": traceback.format_exc()
        }


# ======================================================
# FEEDBACK HANDLER
# ======================================================
def _handle_feedback(job_input):
    """
    Log human corrections for model improvement
    """
    try:
        # Validate feedback structure
        required_fields = ["run_id", "group_id", "max_budget", "min_allocation", 
                          "max_allocation", "min_cases", "edits"]
        missing = [f for f in required_fields if f not in job_input]
        if missing:
            return _error(f"Missing required fields for feedback: {missing}")
        
        edits = job_input["edits"]
        if not isinstance(edits, list) or len(edits) == 0:
            return _error("Edits must be a non-empty list")
        
        # Extract group constraints
        group_id = job_input["group_id"]
        max_budget = job_input["max_budget"]
        min_allocation = job_input["min_allocation"]
        max_allocation = job_input["max_allocation"]
        min_cases = job_input["min_cases"]
        
        logged_count = 0
        errors = []
        
        # Process each edit
        for idx, edit in enumerate(edits):
            try:
                recipient_id = edit.get("RecipientId")
                human_value = edit.get("Human_Final_Value")
                ai_value = edit.get("AI_Suggested_Value")
                features = edit.get("features", {})
                
                if not recipient_id or human_value is None or ai_value is None:
                    errors.append({
                        "index": idx,
                        "error": "Missing RecipientId, Human_Final_Value, or AI_Suggested_Value"
                    })
                    continue
                
                # Log to database
                log_human_edit(
                    recipient_id=recipient_id,
                    ai_suggested=ai_value,
                    human_edited=human_value,
                    features=features,
                    group_id=group_id,
                    max_budget=max_budget,
                    min_allocation=min_allocation,
                    max_allocation=max_allocation,
                    min_cases=min_cases
                )
                logged_count += 1
            
            except Exception as e:
                errors.append({
                    "index": idx,
                    "recipient": edit.get("RecipientId", "unknown"),
                    "error": str(e)
                })
        
        return {
            "statusCode": 200,
            "operation": "feedback",
            "run_id": job_input["run_id"],
            "logged": logged_count,
            "failed": len(errors),
            "errors": errors if errors else None
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "operation": "feedback",
            "error": str(e),
            "trace": traceback.format_exc()
        }


# ======================================================
# TRAIN HANDLER
# ======================================================
def _handle_train(job_input):
    """
    Trigger live model training on logged feedback
    """
    try:
        # Check if feedback exists
        feedback_path = "database/live_experience.csv"
        if not os.path.exists(feedback_path):
            return _error("No feedback data found. Log feedback first before training.")
        
        # Trigger training
        update_model()
        
        return {
            "statusCode": 200,
            "operation": "train",
            "message": "Model training completed successfully",
            "note": "Restart service to load updated model for predictions"
        }
    
    except Exception as e:
        return {
            "statusCode": 500,
            "operation": "train",
            "error": str(e),
            "trace": traceback.format_exc()
        }


# ======================================================
# HELPER
# ======================================================
def _error(msg):
    """Return standardized error response"""
    return {"statusCode": 400, "error": msg}


# 🔥 REQUIRED for RunPod serverless
runpod.serverless.start({
    "handler": handler
})
