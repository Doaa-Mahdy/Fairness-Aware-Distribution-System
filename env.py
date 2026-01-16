import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FairnessEnv(gym.Env):
    def __init__(self, df):
        super(FairnessEnv, self).__init__()
        self.df = df
        self.group_ids = df['group_id'].unique()
        self.current_group_idx = 0
        
        # Define features to observe (matching your list)
        self.feature_cols = [
            'case_status', 'case_reopened', 'case_isactive', 'demo_familysize',
            'demo_deceasedcount', 'demo_eduburden', 'demo_maritalvuln', 'med_disability',
            'med_chronic', 'med_urgent', 'med_count', 'house_isrented', 'house_rent',
            'house_infra', 'house_elec', 'house_ratio', 'fin_balance', 'fin_status',
            'hist_lastmonth', 'xgboost_suggestion'
        ]
        
        # Action: Amount to allocate (Normalized between -1 and 1 for PPO stability)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        
        # Observation: The features + constraints
        # Total features = len(self.feature_cols) + 3 (max_budget, min_alloc, max_alloc)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.feature_cols) + 3,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Select a random group for training variety
        self.group_id = np.random.choice(self.group_ids)
        self.group_data = self.df[self.df['group_id'] == self.group_id].copy()
        self.group_data = self.group_data.reset_index(drop=True)
        
        self.current_step = 0
        self.total_recipients = len(self.group_data)
        self.allocations = np.zeros(self.total_recipients)
        
        # Constraints from data
        self.B_max = self.group_data['max_budget'].iloc[0]
        self.min_alloc = self.group_data['min_allocation'].iloc[0]
        self.max_alloc = self.group_data['max_allocation'].iloc[0]
        self.remaining_budget = self.B_max
        
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.group_data.iloc[self.current_step]
        features = row[self.feature_cols].values.astype(np.float32)
        constraints = np.array([self.B_max, self.min_alloc, self.max_alloc], dtype=np.float32)
        return np.concatenate([features, constraints])

    def step(self, action):
        # Rescale action from [-1, 1] to [min_allocation, max_allocation]
        scaled_action = self.min_alloc + (action[0] + 1) * 0.5 * (self.max_alloc - self.min_alloc)
        # Do not exceed remaining budget
        scaled_action = min(scaled_action, self.remaining_budget)
        self.remaining_budget -= scaled_action
        self.allocations[self.current_step] = scaled_action
        
        self.current_step += 1
        done = self.current_step >= self.total_recipients
        truncated = False
        
        reward = 0
        if done:
            reward = self._calculate_reward()
            
        return self._get_obs() if not done else np.zeros(self.observation_space.shape), reward, done, truncated, {}

    def _calculate_reward(self):
        """Reward scaled to keep magnitudes small for PPO stability."""
        allocs = self.allocations
        total_spent = float(np.sum(allocs))
        util = total_spent / max(1.0, self.B_max)

        cases_meeting_min = int(np.sum(allocs >= self.min_alloc))
        invalid_low_cases = int(np.sum((allocs > 0) & (allocs < self.min_alloc)))
        min_cases_required = self.group_data['min_cases'].iloc[0]

        reward = 0.0
        reward += 1.5 * cases_meeting_min
        reward += 2.0 if cases_meeting_min >= min_cases_required else -1.0 * (min_cases_required - cases_meeting_min)
        reward += 1.0 * util

        # Smooth penalty if <90% spent
        reward -= 2.0 * max(0.0, 0.10 - util)

        # Penalty for invalid low allocations
        reward -= 0.5 * invalid_low_cases

        # Overspend hard penalty
        if total_spent > self.B_max:
            reward -= 10.0 * ((total_spent - self.B_max) / self.B_max + 1.0)

        # Nudge toward max cap: penalize gap to max_alloc on average
        max_gap = np.mean(np.clip(self.max_alloc - allocs, 0, None)) if len(allocs) else 0.0
        reward -= 0.2 * (max_gap / max(1.0, self.max_alloc))

        return reward