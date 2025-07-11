# fix_imports.py - แก้ไข import paths ในไฟล์ทั้งหมด
"""
ไฟล์นี้จะแก้ไข import statements ให้ถูกต้องสำหรับโครงสร้างโปรเจค
"""

import os
import re

def fix_file_imports(file_path, replacements):
    """แก้ไข imports ในไฟล์"""
    if not os.path.exists(file_path):
        print(f"⚠️  File not found: {file_path}")
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        for old_import, new_import in replacements.items():
            content = content.replace(old_import, new_import)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed imports in: {file_path}")
            return True
        else:
            print(f"📝 No changes needed: {file_path}")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """แก้ไข import paths ทั้งหมด"""
    print("🔧 Fixing Import Paths...")
    print("=" * 50)
    
    # 1. Fix src/rl/training_pipeline.py
    training_pipeline_fixes = {
        "from ppo_agent import PPOAgent, PPOConfig, create_ppo_agent": 
        "from .ppo_agent import PPOAgent, PPOConfig, create_ppo_agent"
    }
    
    fix_file_imports("src/rl/training_pipeline.py", training_pipeline_fixes)
    
    # 2. Fix src/rl/evaluation_visualization.py  
    evaluation_fixes = {
        "from ppo_agent import PPOAgent, PPOConfig, create_ppo_agent":
        "from .ppo_agent import PPOAgent, PPOConfig, create_ppo_agent"
    }
    
    fix_file_imports("src/rl/evaluation_visualization.py", evaluation_fixes)
    
    # 3. Fix train_ppo_forex.py (main script)
    main_script_fixes = {
        "from src.rl.trading_environment import create_trading_environment":
        "from src.rl.trading_environment import create_trading_environment",
        "from src.rl.ppo_agent import PPOConfig, create_ppo_agent":
        "from src.rl.ppo_agent import PPOConfig, create_ppo_agent", 
        "from src.rl.training_pipeline import TrainingConfig, create_training_pipeline":
        "from src.rl.training_pipeline import TrainingConfig, create_training_pipeline",
        "from src.rl.evaluation_visualization import create_evaluation_system":
        "from src.rl.evaluation_visualization import create_evaluation_system"
    }
    
    fix_file_imports("train_ppo_forex.py", main_script_fixes)
    
    # 4. Create/fix __init__.py files
    init_files = [
        "src/__init__.py",
        "src/rl/__init__.py", 
        "src/data/__init__.py",
        "src/features/__init__.py"
    ]
    
    print("\n📝 Creating __init__.py files...")
    for init_file in init_files:
        os.makedirs(os.path.dirname(init_file), exist_ok=True)
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('# Package initialization\n')
            print(f"✅ Created: {init_file}")
        else:
            print(f"📝 Already exists: {init_file}")
    
    # 5. Fix specific issues in trading_environment.py
    trading_env_fixes = {
        "from trading_environment import create_trading_environment":
        "from .trading_environment import create_trading_environment"
    }
    
    fix_file_imports("src/rl/trading_environment.py", trading_env_fixes)
    
    print("\n🎯 Import fixes completed!")
    print("Now try running: python train_ppo_forex.py --help")

if __name__ == "__main__":
    main()