
"""
This module provides a centralized way to access the agent tracker
across all modules in the Athena application.
"""

from agent_tracker import AgentTracker, RewardCalculator
from typing import Optional
import streamlit as st

# Global tracker instance for non-Streamlit contexts
_global_tracker: Optional[AgentTracker] = None
_global_calc: Optional[RewardCalculator] = None


def get_tracker() -> AgentTracker:
    """
    Get the agent tracker instance.
    
    In Streamlit context: Uses session state
    In non-Streamlit context: Uses global instance
    
    Returns:
        AgentTracker instance
    """
    try:
        # Try Streamlit session state first
        if 'agent_tracker' not in st.session_state:
            st.session_state.agent_tracker = AgentTracker()
        return st.session_state.agent_tracker
    except:
        # Fallback to global instance
        global _global_tracker
        if _global_tracker is None:
            _global_tracker = AgentTracker()
        return _global_tracker


def get_calc() -> RewardCalculator:
    """
    Get the reward calculator instance.
    
    In Streamlit context: Uses session state
    In non-Streamlit context: Uses global instance
    
    Returns:
        RewardCalculator instance
    """
    try:
        # Try Streamlit session state first
        if 'reward_calc' not in st.session_state:
            st.session_state.reward_calc = RewardCalculator()
        return st.session_state.reward_calc
    except:
        # Fallback to global instance
        global _global_calc
        if _global_calc is None:
            _global_calc = RewardCalculator()
        return _global_calc


def reset_tracker():
    """Reset the tracker (useful for testing or starting fresh)"""
    try:
        # Reset Streamlit session state
        if 'agent_tracker' in st.session_state:
            st.session_state.agent_tracker.reset_episode()
    except:
        # Reset global instance
        global _global_tracker
        if _global_tracker:
            _global_tracker.reset_episode()


# Convenience functions for common operations
def log_action(action_name: str, **params):
    """Quick action logging"""
    tracker = get_tracker()
    return tracker.log_action(action_name, **params)


def add_reward(value: float, reason: str):
    """Quick reward adding"""
    tracker = get_tracker()
    return tracker.add_reward(value, reason)


def log_with_timing(action_name: str, **params):
    """
    Context manager for logging actions with automatic timing
    
    Usage:
        with log_with_timing("search_papers", query="AI"):
            # Do work here
            papers = search(...)
        # Automatically logs completion time
    """
    import time
    from contextlib import contextmanager
    
    @contextmanager
    def timing_context():
        tracker = get_tracker()
        calc = get_calc()
        
        start = time.time()
        action = tracker.log_action(action_name, **params)
        
        try:
            yield tracker, calc
            # Success
            duration = time.time() - start
            tracker.add_reward(calc.task_completion(True), f"Completed: {action_name}")
            tracker.add_reward(calc.response_time(duration), f"Time: {duration:.2f}s")
        except Exception as e:
            # Failure
            tracker.add_reward(calc.error_penalty(), f"Error in {action_name}: {str(e)}")
            raise
    
    return timing_context()