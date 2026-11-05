"""Build an upload archive from an explicit source allowlist."""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ('app', 'scripts', 'configs', 'requirements', 'docs', 'tests', 'experiments/retrieval_v1', 'experiments/retrieval_v2')
FILES = ('README.md', '.gitignore', '.gitattributes', '.env.example',
         'requirements.txt', 'requirements.in', 'run.py', 'experiments/retrieval_v1/questions.json',
         'experiments/retrieval_v1/validation_v1.json', 'experiments/retrieval_v2/questions.json',
         'experiments/evidence_rules_validation/cases.json',
         'experiments/evidence_rules_validation/README.md',
         'experiments/evidence_sft_v1/seed.jsonl', 'experiments/evidence_sft_v1/README.md',
         'experiments/evidence_sft_v1/review.md', 'experiments/evidence_sources_v2/registry.json',
         'experiments/evidence_sources_v2/snapshot_manifest.json', 'experiments/evidence_sources_v2/README.md',
         'experiments/evidence_sft_v2/batch1.json', 'experiments/evidence_sft_v2/README.md',
         'docs/experiments/evidence_candidate_baseline_v1_review.json',
         'docs/experiments/evidence_candidate_prompt_v1_review.json',
         'docs/release_acceptance.json', 'docs/release_results.json',
         'docs/demo_sample.json', 'docs/demo.html', 'docs/demo.png',
         'docs/experiments/medical_v1-data-check.json')
EXTENSIONS = {'.py', '.sh', '.yaml', '.yml', '.md', '.txt', '.in', '.html', '.conf'}
PRIVATE_DOCS = {'docs/cv_project1_audit.md'}


def source_paths():
    paths = [ROOT / name for name in FILES]
    for name in DIRECTORIES:
        paths.extend(p for p in (ROOT / name).rglob('*')
                     if p.is_file() and p.suffix in EXTENSIONS and '__pycache__' not in p.parts)
    return sorted(p for p in set(paths)
                  if p.relative_to(ROOT).as_posix() not in PRIVATE_DOCS
                  and 'superpowers' not in p.relative_to(ROOT).parts)


def main():
    paths = source_paths()
    destination = ROOT / 'dist/RAGDemo-source.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            archive.write(path, 'RAGDemo/' + path.relative_to(ROOT).as_posix())
    print(destination)
    print(f'{len(set(paths))} source files; {destination.stat().st_size:,} bytes')


if __name__ == '__main__':
    main()
