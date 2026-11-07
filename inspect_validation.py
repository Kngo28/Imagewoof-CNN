"""Inspect a saved run using its own submission.py and official predict interface."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--examples', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.examples < 1 or args.examples > 64:
        parser.error('--examples must be between 1 and 64')

    # Use the architecture saved with these weights, even if the root code changes.
    spec = importlib.util.spec_from_file_location('saved_submission', args.run / 'submission.py')
    submission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submission)
    mapping = {}
    for line in (args.data / 'classes.txt').read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            index, name = line.split(maxsplit=1)
            mapping[name.strip()] = int(index)
    if sorted(mapping.values()) != list(range(10)):
        raise ValueError('Expected ten official class IDs 0-9')
    names = sorted(mapping, key=mapping.get)
    if hasattr(submission, 'CLASS_NAMES') and list(submission.CLASS_NAMES) != names:
        raise ValueError('Saved model class order differs from classes.txt')
    folders = {p.name for p in (args.data / 'val').iterdir() if p.is_dir()}
    if folders != set(names):
        raise ValueError('Validation folders differ from classes.txt')
    samples = [(path, mapping[name]) for name in names
               for path in sorted((args.data / 'val' / name).rglob('*.png'))]
    if len(samples) != 1295:
        raise ValueError(f'Expected 1295 validation PNGs, found {len(samples)}')

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = submission.load_model(args.run / 'model.pth')
    matrix = np.zeros((10, 10), dtype=np.int64)
    rows = []
    for start in range(0, len(samples), 64):
        batch = samples[start:start + 64]
        pixels = []
        for path, _ in batch:
            with Image.open(path) as image:
                if image.size != (128, 128) or image.mode != 'RGB':
                    raise ValueError(f'Expected RGB 128x128: {path}')
                pixels.append(np.array(image))
        predictions = submission.predict(model, np.stack(pixels))
        if (not isinstance(predictions, np.ndarray) or predictions.shape != (len(batch),)
                or not np.issubdtype(predictions.dtype, np.integer)
                or np.any((predictions < 0) | (predictions >= 10))):
            raise ValueError('predict returned invalid class IDs')
        for (path, actual), predicted in zip(batch, predictions):
            predicted = int(predicted)
            matrix[actual, predicted] += 1
            rows.append({'image': str(path.resolve()), 'true_id': actual,
                         'predicted_id': predicted, 'true_class': names[actual],
                         'predicted_class': names[predicted], 'correct': actual == predicted})

    output = args.run / 'analysis'
    output.mkdir(exist_ok=True)
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['true / predicted', *names])
        writer.writerows([[name, *row] for name, row in zip(names, matrix)])

    support = matrix.sum(axis=1)
    percentages = np.divide(matrix * 100., support[:, None],
                            out=np.zeros((10, 10)), where=support[:, None] != 0)
    labels = [name.replace('_', ' ') for name in names]
    for filename, values, normalized in [
        ('confusion_matrix.png', matrix, False),
        ('confusion_matrix_normalized.png', percentages, True),
    ]:
        fig, ax = plt.subplots(figsize=(12, 10), layout='constrained')
        heat = ax.imshow(values, cmap='Blues', vmin=0,
                         vmax=100 if normalized else max(1, int(matrix.max())))
        fig.colorbar(heat, ax=ax, label='Percent of true class' if normalized else 'Images')
        ax.set(xticks=range(10), yticks=range(10), xticklabels=labels,
               yticklabels=labels, xlabel='Predicted breed', ylabel='True breed',
               title='Validation confusion matrix' + (' (% within each true breed)' if normalized else ' (counts)'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        threshold = heat.norm.vmax / 2
        for actual in range(10):
            for predicted in range(10):
                value = values[actual, predicted]
                ax.text(predicted, actual, f'{value:.1f}' if normalized else str(value),
                        ha='center', va='center', fontsize=8,
                        color='white' if value > threshold else 'black')
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)

    errors = [row for row in rows if not row['correct']]
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(errors), size=min(args.examples, len(errors)), replace=False)
    columns = 4
    fig, axes = plt.subplots(max(1, (len(chosen) + columns - 1) // columns), columns,
                             figsize=(14, max(1, (len(chosen) + columns - 1) // columns) * 3.3),
                             squeeze=False, layout='constrained')
    for ax in axes.flat:
        ax.axis('off')
    for ax, index in zip(axes.flat, chosen):
        row = errors[index]
        with Image.open(row['image']) as image:
            ax.imshow(image)
        ax.set_title(f"True: {row['true_class'].replace('_', ' ')}\n"
                     f"Pred: {row['predicted_class'].replace('_', ' ')}", fontsize=10)
    fig.suptitle(f'Random validation mistakes (seed {args.seed})' if errors else 'No validation mistakes')
    fig.savefig(output / 'misclassified_examples.png', dpi=160)
    plt.close(fig)

    pairs = sorted([(int(matrix[i, j]), names[i], names[j]) for i in range(10)
                    for j in range(10) if i != j and matrix[i, j] > 0], reverse=True)
    summary = {'images': len(samples), 'correct': int(matrix.trace()),
               'accuracy': float(matrix.trace() / matrix.sum()), 'mistakes': len(errors),
               'per_class_recall': {name: float(percentages[i, i] / 100) for i, name in enumerate(names)},
               'top_confusions': [{'count': count, 'true': actual, 'predicted': predicted}
                                  for count, actual, predicted in pairs[:10]]}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Validation accuracy: {summary['accuracy']:.2%} ({summary['correct']}/{len(samples)})")
    print(f'Mistakes: {len(errors)}')
    for count, actual, predicted in pairs[:5]:
        print(f'  {actual} -> {predicted}: {count}')
    print(f'Reports saved to {output.resolve()}')


if __name__ == '__main__':
    main()
