import os
import shutil


def clean_bytecode():
    """Remove all Python bytecode files and __pycache__ directories"""
    removed_files = 0
    removed_dirs = 0

    # Walk through all directories
    for root, dirs, files in os.walk("."):
        # Remove .pyc files
        for file in files:
            if file.endswith(".pyc") or file.endswith(".pyo"):
                try:
                    filepath = os.path.join(root, file)
                    print(f"Removing file: {filepath}")
                    os.remove(filepath)
                    removed_files += 1
                except Exception as e:
                    print(f"Error removing {file}: {str(e)}")

        # Remove __pycache__ directories
        for dir in dirs[:]:  # Create a copy to avoid modifying the list during iteration
            if dir == "__pycache__":
                try:
                    dirpath = os.path.join(root, dir)
                    print(f"Removing directory: {dirpath}")
                    shutil.rmtree(dirpath)
                    removed_dirs += 1
                    dirs.remove(dir)  # Remove from the list to avoid further descent
                except Exception as e:
                    print(f"Error removing {dirpath}: {str(e)}")

    print(
        f"Cleanup completed. Removed {removed_files} .pyc/.pyo files and {removed_dirs} __pycache__ directories."
    )


if __name__ == "__main__":
    clean_bytecode()
