use std::collections::{
    HashMap,
    HashSet,
};

use burn::data::dataset::{
    InMemDataset,
    vision::ImageLoaderError,
};

/// Scan a folder of ``$ROOT/$CLASS/$IMG.{jpg,png}`` into an `InMemDataset`.
pub fn image_dataset_for_folder<P>(root: P) -> anyhow::Result<InMemDataset<(String, usize)>>
where
    P: AsRef<std::path::Path>,
{
    // Glob all images with extensions
    let walker = globwalk::GlobWalkerBuilder::from_patterns(root.as_ref(), &["*.{jpg,png}"])
        .follow_links(true)
        .sort_by(|p1, p2| p1.path().cmp(p2.path())) // order by path
        .build()
        .map_err(|e| anyhow::anyhow!("Failed to scan folder: {}", e))?;

    // Get all dataset items
    let mut items = Vec::new();
    let mut classes = HashSet::new();
    for img in walker {
        let img = img?;
        let image_path = img.path().to_path_buf();

        // Label name is represented by the parent folder name
        let label = image_path
            .parent()
            .ok_or_else(|| {
                ImageLoaderError::IOError("Could not resolve image parent folder".to_string())
            })?
            .file_name()
            .ok_or_else(|| {
                ImageLoaderError::IOError("Could not resolve image parent folder name".to_string())
            })?
            .to_string_lossy()
            .into_owned();

        classes.insert(label.clone());

        items.push((image_path, label))
    }

    let mut classes = classes.into_iter().collect::<Vec<_>>();
    classes.sort();

    let mut class_to_index = HashMap::new();
    class_to_index.extend(
        classes
            .iter()
            .enumerate()
            .map(|(i, class)| (class.clone(), i)),
    );

    let items = items
        .into_iter()
        .map(|(path, label)| {
            let class = class_to_index[&label];
            (path.to_string_lossy().to_string(), class)
        })
        .collect::<Vec<_>>();

    Ok(InMemDataset::new(items))
}
