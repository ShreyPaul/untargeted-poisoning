#!/usr/bin/env python3
"""
Sybil-based untargeting poisoning attack simulation
"""

import argparse
import sys
from datetime import datetime
import json

def main():
    """Main function for SPoiL aggressive attack simulation"""
    parser = argparse.ArgumentParser(
        description='SPoiL Aggressive Sybil Attack Simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Options: --rounds, --start-round, --honest-clients, --sybil-clients, --num-sybil-per-malicious, --poison-ratio
        """
    )
    parser.add_argument(
        '--rounds', 
        type=int, 
        default=15,
        help='Number of training rounds (default: 15)'
    )
    parser.add_argument(
        '--start-round', 
        type=int, 
        default=2,
        help='Attack start round (default: 2)'
    )
    parser.add_argument(
        '--honest-clients', 
        type=int, 
        default=5,
        help='Number of honest clients (default: 5)'
    )
    parser.add_argument(
        '--sybil-clients', 
        type=int, 
        default=3,
        help='Number of Sybil clients (default: 3)'
    )
    parser.add_argument(
        '--num-sybil-per-malicious',
        type=int,
        default=64,
        help='Number of Sybil nodes per malicious client (default: 64)'
    )
    parser.add_argument(
        '--amplification-factor',
        type=float,
        default=5.0,
        help='Gradient amplification factor for Sybil models (default: 5.0)'
    )
    parser.add_argument(
        '--poison-ratio', 
        type=float, 
        default=1.0,
        help='Poison ratio (default: 1.0)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default=None,
        help='Output filename (default: sybil_attack_results_TIMESTAMP.json)'
    )
    parser.add_argument(
        '--quiet', 
        action='store_true',
        help='Quiet mode, reduce output'
    )
    parser.add_argument(
        '--setup-only', 
        action='store_true',
        help='Only perform environment setup check'
    )
    args = parser.parse_args()
    
    try:
        from setup import quick_setup
        setup_success, config = quick_setup()
        if not setup_success:
            print("Environment setup failed, cannot continue")
            sys.exit(1)
        if args.setup_only:
            print("Environment setup check complete!")
            return
    except Exception as e:
        print(f"Error during environment setup: {e}")
        print("Please check that all dependencies are installed correctly")
        sys.exit(1)
    
    from environment import FederatedLearningEnvironment
    fl_env = FederatedLearningEnvironment(
        num_honest_clients=args.honest_clients,
        num_sybil_clients=args.sybil_clients,
        poison_ratio=args.poison_ratio
    )
    
    try:
        from attack import SybilVirtualDataAttackOrchestrator, ATTACK_SCENARIOS, create_attack_orchestrator
        attack_orchestrator = create_attack_orchestrator(
            fl_env,
            num_sybil_per_malicious=args.num_sybil_per_malicious,
            amplification_factor=args.amplification_factor
        )
        total_rounds = args.rounds
        start_round = args.start_round
        if not args.quiet:
            print(f"Attack config:")
            print(f"   Total rounds: {total_rounds}")
            print(f"   Attack start round: {start_round}")
            print(f"   Attack method: label_flipping")
        results = attack_orchestrator.run_attack_simulation(
            total_rounds=total_rounds,
            attack_start_round=start_round,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Attack execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    output_file = args.output or f"sybil_attack_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUser interrupted, program exiting")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 