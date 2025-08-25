"""
Logging utility for Emobird system

Captures all model interactions, prompts, responses, and system behavior for analysis.
"""

import json
import os
import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class EmobirdLogger:
    """
    Logger for capturing all Emobird model interactions and system behavior.
    """
    
    def __init__(self, log_dir: str = "logs/testing"):
        """Initialize the logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = timestamp
        self.log_file = self.log_dir / f"session_{timestamp}.json"
        
        # Initialize session log
        self.session_data = {
            "session_id": self.session_id,
            "start_time": datetime.datetime.now().isoformat(),
            "interactions": [],
            "summary": {
                "total_interactions": 0,
                "by_component": {},
                "by_type": {}
            }
        }
        
        print(f"📝 Logging session started: {self.session_id}")
        print(f"📁 Log file: {self.log_file}")
    
    def log_interaction(self, component: str, interaction_type: str, 
                       prompt: str, response: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Log a model interaction.
        
        Args:
            component: Component name (e.g., 'scenario_generator', 'factor_generator')
            interaction_type: Type of interaction (e.g., 'abstract_generation', 'factor_generation')
            prompt: Input prompt sent to model
            response: Raw response from model
            metadata: Additional metadata about the interaction
        """
        
        interaction = {
            "timestamp": datetime.datetime.now().isoformat(),
            "component": component,
            "type": interaction_type,
            "prompt": prompt,
            "response": response,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "metadata": metadata or {}
        }
        
        self.session_data["interactions"].append(interaction)
        
        # Update summary
        self.session_data["summary"]["total_interactions"] += 1
        
        # Count by component
        if component not in self.session_data["summary"]["by_component"]:
            self.session_data["summary"]["by_component"][component] = 0
        self.session_data["summary"]["by_component"][component] += 1
        
        # Count by type
        if interaction_type not in self.session_data["summary"]["by_type"]:
            self.session_data["summary"]["by_type"][interaction_type] = 0
        self.session_data["summary"]["by_type"][interaction_type] += 1
        
        # Print progress
        print(f"📊 Logged: {component}.{interaction_type} ({len(self.session_data['interactions'])} total)")
        
        # Save to file periodically
        self._save_session()
    
    def log_analysis_result(self, user_situation: str, result: Dict[str, Any]):
        """
        Log the complete analysis result for a situation.
        
        Args:
            user_situation: Original user input
            result: Complete analysis result from Emobird
        """
        
        # Derive num_factors and workflow steps robustly across schema versions
        meta = result.get('metadata', {}) if isinstance(result, dict) else {}
        num_factors = None
        # 1) Prefer explicit metadata field if it's an int
        mf = meta.get('num_factors')
        if isinstance(mf, int):
            num_factors = mf
        else:
            # 2) Try 'factors' field (can be list of definitions or dict of values)
            f = result.get('factors') if isinstance(result, dict) else None
            if isinstance(f, dict):
                num_factors = len(f)
            elif isinstance(f, list):
                num_factors = len(f)
            else:
                # 3) Legacy key
                num_factors = len(result.get('factors_definition', [])) if isinstance(result, dict) else 0
        # 4) Ensure an integer
        if not isinstance(num_factors, int):
            try:
                num_factors = int(num_factors)
            except Exception:
                num_factors = 0

        workflow_steps = meta.get('workflow_steps')
        if not workflow_steps:
            workflow_steps = meta.get('processing_steps', []) or []

        analysis_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "user_situation": user_situation,
            "result": result,
            "analysis_metadata": {
                "situation_length": len(user_situation),
                "num_factors": num_factors,
                "num_emotions": len(result.get('emotions', {})),
                "top_emotion": max(result.get('emotions', {}).items(), key=lambda x: x[1])[0] if result.get('emotions') else None,
                "workflow_steps": workflow_steps
            }
        }
        
        # Save individual analysis result
        analysis_file = self.log_dir / f"analysis_{self.session_id}_{len(self.session_data.get('analyses', []))}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_log, f, indent=2, ensure_ascii=False)
        
        # Add to session data
        if 'analyses' not in self.session_data:
            self.session_data['analyses'] = []
        self.session_data['analyses'].append(analysis_log)
        
        print(f"🎯 Analysis result logged: {analysis_file.name}")
    
    def log_error(self, component: str, error_type: str, error_message: str, 
                  context: Optional[Dict[str, Any]] = None):
        """
        Log an error or warning.
        
        Args:
            component: Component where error occurred
            error_type: Type of error
            error_message: Error message
            context: Additional context about the error
        """
        
        error_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "component": component,
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }
        
        if 'errors' not in self.session_data:
            self.session_data['errors'] = []
        self.session_data['errors'].append(error_log)
        
        print(f"❌ Error logged: {component}.{error_type} - {error_message}")
        self._save_session()
    
    def _save_session(self):
        """Save the current session data to file."""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save session log: {e}")
    
    def close_session(self):
        """Close the logging session and save final summary."""
        self.session_data["end_time"] = datetime.datetime.now().isoformat()
        
        # Calculate session duration
        start_time = datetime.datetime.fromisoformat(self.session_data["start_time"])
        end_time = datetime.datetime.fromisoformat(self.session_data["end_time"])
        duration = (end_time - start_time).total_seconds()
        self.session_data["duration_seconds"] = duration
        
        # Final save
        self._save_session()
        
        # Create summary file
        summary_file = self.log_dir / f"summary_{self.session_id}.json"
        summary_data = {
            "session_id": self.session_id,
            "duration_seconds": duration,
            "summary": self.session_data["summary"],
            "analyses_count": len(self.session_data.get('analyses', [])),
            "errors_count": len(self.session_data.get('errors', []))
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"📊 Session closed: {duration:.1f}s, {self.session_data['summary']['total_interactions']} interactions")
        print(f"📄 Summary saved: {summary_file.name}")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session."""
        return {
            "session_id": self.session_id,
            "log_file": str(self.log_file),
            "interactions_count": len(self.session_data["interactions"]),
            "analyses_count": len(self.session_data.get('analyses', [])),
            "errors_count": len(self.session_data.get('errors', []))
        }


# Global logger instance
_global_logger: Optional[EmobirdLogger] = None

def get_logger() -> EmobirdLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = EmobirdLogger()
    return _global_logger

def set_logger(logger: EmobirdLogger):
    """Set the global logger instance."""
    global _global_logger
    _global_logger = logger

def close_logger():
    """Close the global logger instance."""
    global _global_logger
    if _global_logger:
        _global_logger.close_session()
        _global_logger = None
