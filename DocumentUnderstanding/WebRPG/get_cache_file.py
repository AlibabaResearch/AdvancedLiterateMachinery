
import os
import argparse

def update_cache(cache_file, target_folder):
    if not os.path.exists(cache_file):
        with open(cache_file, 'w') as f:
            for files in os.listdir(target_folder):
                path = os.path.join(target_folder, files)
                f.write(path + '\n')

def main():
    parser = argparse.ArgumentParser(description="Update the cache with file paths.")
    parser.add_argument('--cache_file', help="Path to the cache file")
    parser.add_argument('--target_folder', help="Path to the target folder containing the files")
    
    args = parser.parse_args()

    update_cache(args.cache_file, args.target_folder)

if __name__ == '__main__':
    main()
