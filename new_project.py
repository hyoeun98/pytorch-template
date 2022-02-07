import sys
from pathlib import Path
from shutil import copytree, ignore_patterns


# This script initializes new pytorch project with the template files.
# Run `python3 new_project.py ../MyNewProject` then new project named 
# MyNewProject will be made
current_dir = Path() # 현재 경로
assert (current_dir / 'new_project.py').is_file(), 'Script should be executed in the pytorch-template directory' # 현재 경로에 new_project_py가 있을 경우
assert len(sys.argv) == 2, 'Specify a name for the new project. Example: python3 new_project.py MyNewProject' # 인자가 2개 주어질 경우

project_name = Path(sys.argv[1]) # 첫번째 인자를 project_name으로 설정
target_dir = current_dir / project_name # current_path/project_name

ignore = [".git", "data", "saved", "new_project.py", "LICENSE", ".flake8", "README.md", "__pycache__"] # 해당 파일을 ignore
copytree(current_dir, target_dir, ignore=ignore_patterns(*ignore)) # current_dir의 파일,폴더를 target_dir에 복사
print('New project initialized at', target_dir.absolute().resolve()) # absolute == resolve