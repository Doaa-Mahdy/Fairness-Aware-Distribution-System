import json
from feedback import log_human_edit

# Load example feedback data
with open("in/example_feedback.json", "r") as f:
    feedback_data = json.load(f)

print("Testing feedback logging system...")
print(f"Processing {len(feedback_data['edits'])} human edits from run: {feedback_data['run_id']}")

# Extract group-level constraints
group_id = feedback_data.get('group_id', 1)
max_budget = feedback_data.get('max_budget', 10000.0)
min_allocation = feedback_data.get('min_allocation', 50.0)
max_allocation = feedback_data.get('max_allocation', 1000.0)
min_cases = feedback_data.get('min_cases', 5)

print(f"\nGroup Constraints:")
print(f"  Group ID: {group_id}")
print(f"  Max Budget: ${max_budget:,.2f}")
print(f"  Min Allocation: ${min_allocation}")
print(f"  Max Allocation: ${max_allocation}")
print(f"  Min Cases: {min_cases}")

# Process each edit
for edit in feedback_data['edits']:
    recipient_id = edit['RecipientId']
    ai_suggested = edit['AI_Suggested_Value']
    human_edited = edit['Human_Final_Value']
    features = edit['features']
    
    print(f"\nLogging edit for {recipient_id}:")
    print(f"  AI suggested: ${ai_suggested:.2f}")
    print(f"  Human edited to: ${human_edited:.2f}")
    print(f"  Difference: ${human_edited - ai_suggested:+.2f}")
    
    # Log the feedback with group constraints
    log_human_edit(
        recipient_id, 
        ai_suggested, 
        human_edited, 
        features,
        group_id=group_id,
        max_budget=max_budget,
        min_allocation=min_allocation,
        max_allocation=max_allocation,
        min_cases=min_cases
    )

print("\n✓ All feedback entries logged successfully!")
print(f"Check 'database/live_experience.csv' for the logged data")
