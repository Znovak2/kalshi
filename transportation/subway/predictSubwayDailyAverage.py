#!/usr/bin/env python3
"""
NYC Subway Daily Average Prediction - Master Automation Script

This script automates the complete subway ridership forecasting workflow:
1. Phase 1: Generate base predictions using the hybrid model
2. Phase 2: Calculate bias corrections and model evaluation
3. Phase 3: Apply hybrid strategy with bias calibration
4. Phase 4: Calculate weekly averages with actual data integration

The script orchestrates all phases and outputs the final daily average prediction
to 'daily_average_subway_prediction.csv' in the transportation/subway directory.

Usage:
    python predictSubwayDailyAverage.py [--verbose] [--skip-phases PHASES]

Examples:
    python predictSubwayDailyAverage.py                    # Run full workflow
    python predictSubwayDailyAverage.py --verbose          # Detailed output
    python predictSubwayDailyAverage.py --skip-phases 1,2  # Skip phases 1 and 2
"""

import os
import sys
import argparse
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional


class SubwayPredictionWorkflow:
    """Orchestrates the complete subway prediction workflow."""
    
    def __init__(self, base_dir: str, verbose: bool = False):
        self.base_dir = Path(base_dir)
        self.verbose = verbose
        
        # Define phase directories
        self.phase1_dir = self.base_dir / "phase1"
        self.phase2_dir = self.base_dir / "phase2"
        self.phase3_dir = self.base_dir / "phase3"
        self.phase4_dir = self.base_dir / "phase4"
        
        # Define key files
        self.predictions_file = self.phase3_dir / "subway_predictions.csv"
        self.final_output_file = self.base_dir / "daily_average_subway_prediction.csv"
        
    def log(self, message: str, level: str = "INFO") -> None:
        """Log messages with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {level}:"
        if self.verbose or level in ["ERROR", "SUCCESS"]:
            print(f"{prefix} {message}")
    
    def run_python_script(self, script_path: Path, description: str, args: Optional[List[str]] = None) -> bool:
        """Run a Python script and return success status."""
        if not script_path.exists():
            self.log(f"Script not found: {script_path}", "ERROR")
            return False
        
        self.log(f"Running {description}...")
        self.log(f"Script: {script_path}")
        
        try:
            # Change to the script's directory
            script_dir = script_path.parent
            original_dir = os.getcwd()
            os.chdir(script_dir)
            
            # Build command
            cmd = [sys.executable, str(script_path.name)]
            if args:
                cmd.extend(args)
            
            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
            
            # Return to original directory
            os.chdir(original_dir)
            
            if result.returncode == 0:
                self.log(f"✓ {description} completed successfully", "SUCCESS")
                if self.verbose and result.stdout:
                    self.log(f"Output: {result.stdout.strip()}")
                return True
            else:
                self.log(f"✗ {description} failed (exit code {result.returncode})", "ERROR")
                if result.stderr:
                    self.log(f"Error: {result.stderr.strip()}", "ERROR")
                if result.stdout:
                    self.log(f"Output: {result.stdout.strip()}")
                return False
                
        except Exception as e:
            self.log(f"✗ {description} failed with exception: {e}", "ERROR")
            return False
    
    def run_phase1(self) -> bool:
        """Run Phase 1: Base prediction generation."""
        script = self.phase3_dir / "hybridPrediction.py"
        return self.run_python_script(script, "Phase 1: Hybrid Prediction Generation")
    
    def run_phase2(self) -> bool:
        """Run Phase 2: Model evaluation and bias correction."""
        # Run MSAE test for evaluation
        msae_script = self.phase2_dir / "msae_test.py"
        success1 = self.run_python_script(msae_script, "Phase 2a: MSAE Testing")
        
        # Run MSAE analysis for bias calculation
        analysis_script = self.phase2_dir / "msae_analysis.py"
        success2 = self.run_python_script(analysis_script, "Phase 2b: MSAE Analysis")
        
        return success1 and success2
    
    def run_phase3(self) -> bool:
        """Run Phase 3: Apply hybrid strategy (already done in phase1)."""
        self.log("✓ Phase 3: Hybrid strategy integrated with Phase 1", "SUCCESS")
        return True
    
    def run_phase4(self) -> bool:
        """Run Phase 4: Weekly averaging with actual data integration."""
        script = self.phase4_dir / "weeklyAverage.py"
        args = ["--save-to-main"]
        if not self.verbose:
            args.append("--quiet")
        return self.run_python_script(script, "Phase 4: Weekly Average Calculation", args)
    
    def move_final_output(self) -> bool:
        """Verify the final output was created correctly."""
        try:
            if not self.final_output_file.exists():
                self.log(f"Final output file not found: {self.final_output_file}", "ERROR")
                return False
            
            self.log(f"✓ Final output verified at: {self.final_output_file}", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"✗ Failed to verify final output: {e}", "ERROR")
            return False
    
    def cleanup_intermediate_files(self) -> None:
        """Clean up intermediate prediction files from wrong locations."""
        # Remove subway_predictions.csv from base directory if it exists
        old_predictions = self.base_dir / "subway_predictions.csv"
        if old_predictions.exists():
            old_predictions.unlink()
            self.log("Cleaned up old predictions file from base directory")
    
    def run_workflow(self, skip_phases: Optional[List[int]] = None) -> bool:
        """Run the complete workflow."""
        skip_phases = skip_phases or []
        
        self.log("=" * 60)
        self.log("NYC SUBWAY DAILY AVERAGE PREDICTION WORKFLOW")
        self.log("=" * 60)
        
        # Phase execution mapping
        phases = {
            1: ("Phase 1: Base Predictions", self.run_phase1),
            2: ("Phase 2: Model Evaluation", self.run_phase2), 
            3: ("Phase 3: Hybrid Strategy", self.run_phase3),
            4: ("Phase 4: Weekly Averaging", self.run_phase4)
        }
        
        success = True
        
        # Execute phases
        for phase_num, (description, phase_func) in phases.items():
            if phase_num in skip_phases:
                self.log(f"Skipping {description}")
                continue
                
            self.log(f"\n--- {description} ---")
            if not phase_func():
                self.log(f"Workflow failed at {description}", "ERROR")
                success = False
                break
        
        # Move final output if all phases succeeded
        if success:
            self.log(f"\n--- Final Output Processing ---")
            success = self.move_final_output()
        
        # Cleanup
        self.cleanup_intermediate_files()
        
        # Final status
        self.log(f"\n" + "=" * 60)
        if success:
            self.log("WORKFLOW COMPLETED SUCCESSFULLY", "SUCCESS")
            self.log(f"Daily average prediction saved to: {self.final_output_file.name}")
        else:
            self.log("WORKFLOW FAILED", "ERROR")
        self.log("=" * 60)
        
        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the complete NYC Subway daily average prediction workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predictSubwayDailyAverage.py                    # Run full workflow
  python predictSubwayDailyAverage.py --verbose          # Detailed output  
  python predictSubwayDailyAverage.py --skip-phases 1,2  # Skip phases 1 and 2
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--skip-phases",
        type=str,
        help="Comma-separated list of phase numbers to skip (e.g., '1,2')"
    )
    
    args = parser.parse_args()
    
    # Parse skip phases
    skip_phases = []
    if args.skip_phases:
        try:
            skip_phases = [int(x.strip()) for x in args.skip_phases.split(",")]
        except ValueError:
            print("Error: Invalid skip-phases format. Use comma-separated integers.")
            sys.exit(1)
    
    # Get the script directory (transportation/subway)
    script_dir = Path(__file__).parent
    
    # Create and run workflow
    workflow = SubwayPredictionWorkflow(script_dir, verbose=args.verbose)
    success = workflow.run_workflow(skip_phases=skip_phases)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
