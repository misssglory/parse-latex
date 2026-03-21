# Translate Latex from images to formula

- [Paper](https://arxiv.org/pdf/1609.04938)


## Dataset preprocess stage

- [Dataset link](https://zenodo.org/records/56198#.V2p0KTXT6eA) 
- Scripts crop images to least common union rectangle and convert to 1-channel format in order to save disk space
- Rust is used in order to preprocess with CPU multithreading

```bash
# Consider dataset is in dataset folder
cd dataset
unzip 56198.zip
tar -xvf formula_images.tar.gz
rm 56198.zip
rm formula_images.tar.gz
cd -
mkdir -p dataset/crop
mkdir -p dataset/formatted
cargo run --release --bin crop_union_formulas -- dataset/formula_images dataset/crop
cargo run --release --bin change_format -- dataset/crop dataset/formatted
rm -r dataset/crop
rm -r dataset/formula_images
```

## Nix environment setup for ROCm (incomplete)
```
nix build .#rocm-cmake
nix build .#libdivide
nix build .#hipsparselt
...
```



## Starting input prompt
```
here is paper to implement image to text attention model for latex formulas translation from images. Dataset description:
- im2latex_[...].lst files
- Each line is separate generated image per formula
- Line structure: "[formula_idx] [image_name] [render_type]"
- formula_idx is the line number of the formula in im2latex_formulas.lst
- image_name is the name of rendered image (without '.png')
- render_type is the name of used rendering settings (in image2latex.py)
- Dataset is split to train, validation and test
- train is for training the model
- validation is for things testing accuracy while training / checking overfitting
- test set is for final evaluation
- im2latex_formulas.lst
- Each line is a separate formula (formulas were stripped from "\\\\\\\\n")
- formula_images.tar.gz
- tar file of the rendered images
- Each image is an image of full "A4" page, where _only_ the formula is rendered
- -> Lots of empty space

here is dataset structure:
ls dataset
im2latex_formulas.lst im2latex_test.lst im2latex_train.lst im2latex_validate.lst formatted readme.txt

ls dataset/formatted | head -n 5
100009e256.png
10007a9101.png
1000a29807.png
1000f2a2d2.png
10011a6bf8.png

Result images are postprocessed and are of following shape:
file dataset/formatted/100009e256.png
dataset/formatted/100009e256.png: PNG image data, 1545 x 1488, 8-bit grayscale, non-interlaced
file dataset/formatted/10007a9101.png

dataset/formatted/10007a9101.png: PNG image data, 1545 x 1488, 8-bit grayscale, non-interlaced

Use keras3 with tensorflow backend for implementation, use loguru for logging. Separate training and evaluation functions. Every N epochs (set in cli args with default value 5) draw 5 samples of input images, same images with visualized attention, ground truth formulas, predicted formulas, char-wise diff between them and status if predicted formula compiles to latex or not
```

## Output

Below is a solid implementation plan and a full Keras 3 code skeleton for an image-to-LaTeX model based on the paper’s core idea: a CNN encoder over the image, a row encoder to preserve spatial order, and an autoregressive decoder with visual attention over the encoded feature grid. The paper explicitly argues that row-wise recurrent encoding is important for OCR-like structured markup generation, and that cropping the formula region plus downsampling materially improves efficiency on IM2LATEX-style data. It also evaluates output primarily by rendered-image agreement rather than only token accuracy, because multiple LaTeX strings can render to the same visual formula.

Architecture
Your dataset matches the paper’s setup closely: one rendered image per formula instance, a separate formula list, and train/validation/test split files that map image names to formula indices. The paper’s encoder uses a CNN to produce a 2D feature grid, then runs an RNN across each row of that grid so the decoder can attend to location-aware features; this row encoder was reported as important for final exact-match performance. For a practical Keras 3 implementation, I would keep that idea but use a lighter GRU/LSTM row encoder, Bahdanau attention, and a token-level decoder with teacher forcing during training.

Because your images are still large grayscale pages with lots of empty space, the most important preprocessing step is dynamic trimming to the tight formula bounding box, then adding a small pad and resizing or bucketing for batching. The original paper cropped to the formula area, padded by 8 pixels on each side, then downsampled to half resolution and grouped similar image sizes for batching, specifically to reduce attention cost and memory use. If you skip this, attention over 1545 x 1488 images will be unnecessarily expensive and training will be much slower.

Project layout
A clean layout is useful because you asked for separate training and evaluation functions plus periodic visual diagnostics.

```text
im2latex_keras/
├── train.py
├── model.py
├── data.py
├── utils.py
├── viz.py
├── metrics.py
├── requirements.txt
└── outputs/
    ├── checkpoints/
    ├── samples/
    └── logs/
```
