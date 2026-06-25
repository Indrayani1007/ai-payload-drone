import os

for folder in ['train', 'valid', 'test']:
    img_dir = f'./dataset/{folder}/images'
    lbl_dir = f'./dataset/{folder}/labels'

    if not os.path.exists(lbl_dir):
        continue

    images = sorted(os.listdir(img_dir))
    labels = sorted(os.listdir(lbl_dir))

    print(f'{folder}: {len(images)} images, {len(labels)} labels')

    # First rename all labels to temp names
    for i, lbl in enumerate(labels):
        os.rename(
            os.path.join(lbl_dir, lbl),
            os.path.join(lbl_dir, f'tmp_{i:05d}.txt')
        )

    # Now rename temp files to match image names
    tmp_labels = sorted(os.listdir(lbl_dir))
    for i, tmp in enumerate(tmp_labels):
        img_name = os.path.splitext(images[i])[0]
        os.rename(
            os.path.join(lbl_dir, tmp),
            os.path.join(lbl_dir, f'{img_name}.txt')
        )

    # Count matches
    matched = 0
    for f in os.listdir(img_dir):
        name = os.path.splitext(f)[0]
        if os.path.exists(os.path.join(lbl_dir, name + '.txt')):
            matched += 1

    print(f'  Matched: {matched}/{len(images)}')

print()
print('Sample check:')
for f in sorted(os.listdir('./dataset/train/images'))[:3]:
    name = os.path.splitext(f)[0]
    lbl_path = f'./dataset/train/labels/{name}.txt'
    status = 'MATCH' if os.path.exists(lbl_path) else 'MISSING'
    print(f'  {f} -> {name}.txt -> {status}')