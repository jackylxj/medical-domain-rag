"""Optional cloud download; skip if the model already exists."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='/mnt/workspace/models/Qwen2-7B-Instruct')
    parser.add_argument('--revision', default=None, help='Optional ModelScope revision')
    args = parser.parse_args()
    from modelscope import snapshot_download
    print(snapshot_download(model_id='Qwen/Qwen2-7B-Instruct',
                            local_dir=args.output, revision=args.revision))


if __name__ == '__main__':
    main()
