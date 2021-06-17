import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Update the bash script',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--models_path', type=str, required=True, help='Path to models folder')
    parser.add_argument('--bash_path', type=str, required=True, help='Path to the bash script')

    args = parser.parse_args()
    models_path = args.models_path
    bash_path = args.bash_path

    with open(bash_path, 'r+') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if 'timeout 118m' in line:
                cmd = line.strip().split(' ')
                if list(Path(models_path).glob('*.h5')):
                    model_path_arg = '--model_path=' + str(sorted(list(Path(models_path).glob('*.h5')))[-1])
                    for idx, arg in enumerate(cmd):
                        if '--model_path' in arg:
                            cmd[idx] = model_path_arg
                    cmd.append(model_path_arg)
                lines[index] = ' '.join(cmd)

        f.seek(0)
        f.writelines(lines)
        f.truncate()
