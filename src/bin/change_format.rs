use anyhow::{Context, Result};
use clap::Parser;
use image::{GrayImage, Luma};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "change_format",
    about = "Convert RGBA PNG images to 1-channel grayscale PNGs with transparent background replaced by white."
)]
struct Args {
    /// Folder with source PNG images
    input_dir: PathBuf,
    /// Folder to save converted images
    output_dir: PathBuf,
}

fn is_png(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("png"))
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
    files.sort();
    Ok(files)
}

fn rgba_to_gray_white_bg(path: &Path, output_dir: &Path) -> Result<()> {
    let img = image::open(path)
        .with_context(|| format!("Failed to open image: {}", path.display()))?
        .to_rgba8();

    let (w, h) = img.dimensions();
    let mut out = GrayImage::new(w, h);

    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x, y);
            let [r, g, b, a] = p.0;

            let gray = if a == 0 {
                255u8
            } else {
                let rf = r as f32 / 255.0;
                let gf = g as f32 / 255.0;
                let bf = b as f32 / 255.0;
                let af = a as f32 / 255.0;

                let comp_r = rf * af + 1.0 * (1.0 - af);
                let comp_g = gf * af + 1.0 * (1.0 - af);
                let comp_b = bf * af + 1.0 * (1.0 - af);

                let y_lin = 0.299 * comp_r + 0.587 * comp_g + 0.114 * comp_b;
                (y_lin * 255.0).round().clamp(0.0, 255.0) as u8
            };

            out.put_pixel(x, y, Luma([gray]));
        }
    }

    let file_name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("Invalid file name: {}", path.display()))?;
    let out_path = output_dir.join(file_name);

    out.save(&out_path)
        .with_context(|| format!("Failed to save image: {}", out_path.display()))?;

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

    fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("Failed to create output dir: {}", args.output_dir.display()))?;

    let pngs = collect_pngs(&args.input_dir)?;
    if pngs.is_empty() {
        anyhow::bail!("No PNG files found in input directory");
    }

    pngs.par_iter()
        .try_for_each(|path| rgba_to_gray_white_bg(path, &args.output_dir))?;

    Ok(())
}