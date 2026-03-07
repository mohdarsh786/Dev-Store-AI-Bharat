"""
Production Ingestion Runner

Runs the ingestion pipeline with infrastructure detection
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator_production import ProductionOrchestrator


async def check_infrastructure():
    """Check if infrastructure is available"""
    try:
        from clients.database import DatabaseClient
        from clients.redis_client import RedisClient
        
        # Try to connect to database
        db = DatabaseClient()
        db_health = db.health_check()
        
        # Try to connect to Redis
        redis = RedisClient()
        await redis.connect()
        redis_health = await redis.ping()
        await redis.disconnect()
        
        if db_health['status'] == 'healthy' and redis_health:
            print("✓ Infrastructure available (PostgreSQL + Redis)")
            return True
        else:
            print("✗ Infrastructure not fully available")
            return False
            
    except Exception as e:
        print(f"✗ Infrastructure not available: {e}")
        return False


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run production ingestion pipeline')
    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['huggingface', 'openrouter', 'github', 'kaggle'],
        help='Sources to run (default: all)'
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level'
    )
    parser.add_argument(
        '--force-json',
        action='store_true',
        help='Force JSON-only mode (skip infrastructure)'
    )
    parser.add_argument(
        '--check-only',
        action='store_true',
        help='Only check infrastructure availability'
    )
    
    args = parser.parse_args()
    
    # Check infrastructure
    print("Checking infrastructure availability...")
    has_infrastructure = await check_infrastructure()
    
    if args.check_only:
        sys.exit(0 if has_infrastructure else 1)
    
    # Determine mode
    use_infrastructure = has_infrastructure and not args.force_json
    
    if use_infrastructure:
        print("\n🚀 Running in PRODUCTION mode (with infrastructure)")
    else:
        print("\n📝 Running in JSON mode (no infrastructure)")
    
    # Run orchestrator
    orchestrator = ProductionOrchestrator(
        sources=args.sources,
        log_level=args.log_level,
        use_infrastructure=use_infrastructure
    )
    
    result = await orchestrator.run()
    
    # Print summary
    print("\n" + "=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Status: {result['status']}")
    print(f"Duration: {result.get('duration_seconds', 0):.2f} seconds")
    print(f"\nStatistics:")
    for key, value in result.get('stats', {}).items():
        print(f"  {key}: {value}")
    
    if result.get('source_stats'):
        print(f"\nBy Source:")
        for source, stats in result['source_stats'].items():
            print(f"  {source}: {stats}")
    
    # Exit with appropriate code
    sys.exit(0 if result['status'] in ['completed', 'skipped'] else 1)


if __name__ == '__main__':
    asyncio.run(main())
