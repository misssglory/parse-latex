use anyhow::{Context, Result};
use clap::Parser;
use image::{GenericImageView, ImageBuffer, Rgba};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "crop_union_formulas",
    about = "Find common union region of black text across PNGs and crop all images to that rectangle."
)]
struct Args {
    /// Folder with source PNG images
    input_dir: PathBuf,
    /// Folder to save cropped images
    output_dir: PathBuf,
}

#[derive(Debug, Clone, Copy)]
struct BBox {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

impl BBox {
    fn new(x: u32, y: u32) -> Self {
        BBox {
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
        }
    }

    fn union(&self, other: &BBox) -> BBox {
        BBox {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }
}

fn is_png(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("png"))
        .unwrap_or(false)
}

fn collect_pngs(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in WalkDir::new(dir).min_depth(1).max_depth(1) {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && is_png(path) {
            files.push(path.to_path_buf());
        }
    }
    Ok(files)
}

/// Determine per-image bounding box of black pixels (alpha>0, rgb=0,0,0)
fn find_bbox_for_image(path: &Path) -> Result<Option<BBox>> {
    let img = image::open(path)
        .with_context(|| format!("Failed to open image: {}", path.display()))?;
    let (w, h) = img.dimensions();

    let rgba = img.to_rgba8();

    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut found = false;

    for y in 0..h {
        for x in 0..w {
            let p: Rgba<u8> = rgba.get_pixel(x, y).to_owned();
            let [r, g, b, a] = p.0;

            // black text on transparent bg
            if a > 0 && r == 0 && g == 0 && b == 0 {
                if !found {
                    min_x = x;
                    max_x = x;
                    min_y = y;
                    max_y = y;
                    found = true;
                } else {
                    if x < min_x {
                        min_x = x;
                    }
                    if y < min_y {
                        min_y = y;
                    }
                    if x > max_x {
                        max_x = x;
                    }
                    if y > max_y {
                        max_y = y;
                    }
                }
            }
        }
    }

    if found {
        Ok(Some(BBox {
            min_x,
            min_y,
            max_x,
            max_y,
        }))
    } else {
        Ok(None)
    }
}

fn find_global_bbox(paths: &[PathBuf]) -> Result<Option<BBox>> {
    // Parallel per-image bbox
    let per_image: Vec<Option<BBox>> = paths
        .par_iter()
        .map(|p| find_bbox_for_image(p).unwrap_or(None))
        .collect();

    // Reduce to global bbox
    let mut global: Option<BBox> = None;
    for bbox_opt in per_image.into_iter().flatten() {
        global = Some(match global {
            None => bbox_opt,
            Some(acc) => acc.union(&bbox_opt),
        });
    }
    Ok(global)
}

fn crop_with_bbox(input_paths: &[PathBuf], output_dir: &Path, bbox: BBox) -> Result<()> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output dir: {}", output_dir.display()))?;

    input_paths.par_iter().try_for_each(|path| -> Result<()> {
        let img = image::open(path)
            .with_context(|| format!("Failed to open image for cropping: {}", path.display()))?;
        let (w, h) = img.dimensions();

        // clamp bbox just in case
        let left = bbox.min_x.min(w - 1);
        let top = bbox.min_y.min(h - 1);
        let right = bbox.max_x.min(w - 1);
        let bottom = bbox.max_y.min(h - 1);

        // width/height of cropped region
        let cw = right - left + 1;
        let ch = bottom - top + 1;

        let rgba = img.to_rgba8();
        let mut cropped: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::new(cw, ch);

        for y in 0..ch {
            for x in 0..cw {
                let src_x = left + x;
                let src_y = top + y;
                let p = rgba.get_pixel(src_x, src_y);
                cropped.put_pixel(x, y, *p);
            }
        }

        let out_path = output_dir.join(
            path.file_name()
                .ok_or_else(|| anyhow::anyhow!("Invalid file name: {}", path.display()))?,
        );

        cropped
            .save(&out_path)
            .with_context(|| format!("Failed to save cropped image: {}", out_path.display()))?;

        Ok(())
    })?;

    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.input_dir.is_dir() {
        anyhow::bail!(
            "Input dir does not exist or is not a directory: {}",
            args.input_dir.display()
        );
    }

    let pngs = collect_pngs(&args.input_dir)?;
    if pngs.is_empty() {
        anyhow::bail!("No PNG files found in input directory");
    }

    let global_bbox = find_global_bbox(&pngs)?;
    let global_bbox = match global_bbox {
        Some(b) => b,
        None => anyhow::bail!("No black pixels found in any image; nothing to crop"),
    };

    crop_with_bbox(&pngs, &args.output_dir, global_bbox)?;
    Ok(())
}