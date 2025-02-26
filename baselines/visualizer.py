# Visualize training samples
sample_batch = next(iter(dl_train))

fig, axes = plt.subplots(8, 8, figsize=(15, 15))
for i in range(8):
    for j in range(8):
        idx = i * 8 + j
        if idx < len(sample_batch[0][0]):
            original_img = sample_batch[0][0][idx].permute(1, 2, 0).numpy()
            augmented_img = sample_batch[0][1][idx].permute(1, 2, 0).numpy()
            
            # Normalize the images to be between 0 and 1
            original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
            augmented_img = (augmented_img - augmented_img.min()) / (augmented_img.max() - augmented_img.min())
            
            axes[i, j].imshow(np.concatenate((original_img, augmented_img), axis=1))
            axes[i, j].axis('off')

plt.tight_layout()
plt.show()


# Visualize validation samples
batch_val = next(iter(dl_valid))

sample_task = batch_val[0]
print(sample_task.shape)
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

for i in range(2):
    for j in range(10):
        img = sample_task[i * 10 + j].permute(1, 2, 0).numpy()
        axes[i, j].imshow(img)
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()