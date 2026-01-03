# run_optimizer.py
# Blessing Continuous Optimizer - Daily Runner
# Uruchom: python run_optimizer.py
# Przerwij: Ctrl+C (zapisze stan automatycznie)
# Wznów: python run_optimizer.py (kontynuuje od przerwania)

import sys
from pathlib import Path
import logging
import argparse

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from optimization.blessing_continuous_optimizer import BlessingContinuousOptimizer


def main():
    parser = argparse.ArgumentParser(description='Blessing Continuous Optimizer')
    parser.add_argument('--max-hours', type=float, default=None,
                       help='Max runtime in hours (default: infinite)')
    parser.add_argument('--max-backtests', type=int, default=None,
                       help='Max backtests (default: infinite)')
    parser.add_argument('--gpu', action='store_true', default=True,
                       help='Use GPU acceleration (default: True)')
    parser.add_argument('--no-gpu', action='store_false', dest='gpu',
                       help='Disable GPU')
    parser.add_argument('--checkpoint-interval', type=int, default=10,
                       help='Save state every N backtests (default: 10)')
    parser.add_argument('--state-file', type=str,
                       default='data/state/blessing_optimizer_state.pkl',
                       help='Path to state file')
    parser.add_argument('--results-dir', type=str,
                       default='data/results/continuous',
                       help='Results directory')
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Trading symbol (default: EURUSD)')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date (default: 2024-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date (default: 2024-12-31)')

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/optimizer.log'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    # Print banner
    print("="*70)
    print(" "*15 + "BLESSING CONTINUOUS OPTIMIZER")
    print("="*70)
    print(f"GPU:              {'ENABLED' if args.gpu else 'DISABLED'}")
    print(f"Max runtime:      {args.max_hours}h" if args.max_hours else "Max runtime:      INFINITE")
    print(f"Max backtests:    {args.max_backtests}" if args.max_backtests else "Max backtests:    INFINITE")
    print(f"Checkpoint:       Every {args.checkpoint_interval} backtests")
    print(f"State file:       {args.state_file}")
    print(f"Results dir:      {args.results_dir}")
    print(f"Symbol:           {args.symbol}")
    print(f"Date range:       {args.start_date} to {args.end_date}")
    print("="*70)
    print("\nPress Ctrl+C to stop (state will be saved automatically)")
    print("Run again to resume from last checkpoint\n")
    print("="*70 + "\n")

    # Create optimizer
    optimizer = BlessingContinuousOptimizer(
        state_file=args.state_file,
        results_dir=args.results_dir,
        checkpoint_interval=args.checkpoint_interval,
        use_gpu=args.gpu,
        verbose=True,
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Run
    try:
        optimizer.run_continuous(
            max_runtime_hours=args.max_hours,
            max_backtests=args.max_backtests
        )

        logger.info("\n" + "="*70)
        logger.info("[COMPLETE] Optimization finished successfully!")
        logger.info("="*70)

    except KeyboardInterrupt:
        logger.info("\n" + "="*70)
        logger.info("[STOPPED] User interrupted - state saved")
        logger.info("[INFO] Run 'python run_optimizer.py' again to resume")
        logger.info("="*70)

    except Exception as e:
        logger.error(f"\n[ERROR] Optimization failed: {e}")
        logger.error("[INFO] State was saved before error")
        logger.error("[INFO] Fix the issue and run again to resume")
        raise


if __name__ == "__main__":
    main()
