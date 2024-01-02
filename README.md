# neural_image

A neural network learning the mapping between pixel coordinates and pixel color
of an image. In practice, the 2D input is not used directly, but converted to a
set of sines and cosines of different frequencies (Fourier features), allowing
the network to learn high-frequency details much more easily.

## Example

The output image was obtained using 256 input Fourier features, 4 leaky ReLU
hidden layers of sizes {512, 512, 128, 128}, and a sigmoid 3-channel output
layer.
The network was trained using stochastic gradient descent for 100 epochs with a
learning rate of 0.01.

![Input](images/input.png)
![Output](images/output.png)

## TODO

- Clean up the Fourier features code

## Build

All dependencies are handled
by [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake).

```
git clone https://github.com/mschertenleib/neural_image.git
cd neural_image
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target neural_image
```

## Usage

```
SYNOPSIS
        neural_image.exe -h
        neural_image.exe -i <input> -o <output> [-W <width>] [-H <height>] [-a <layer_sizes>...]
                         [-g] [-e <epochs>] [-l <learning_rate>]

OPTIONS
        -h, --help  Show this message and exit
        -i, --input <input>
                    The input image (JPEG, PNG, TGA, BMP, PSD, GIF, HDR, PIC, PNM)

        -o, --output <output>
                    The output image (PNG)

        -W, --width <width>
                    The width of the output image

        -H, --height <height>
                    The height of the output image

        -a, --arch <layer_sizes>
                    Sizes of the network layers (includes the input size but excludes the output
                    size)

        -g, --gray  Force grayscale for the output image (by default, the output will be either RGB
                    or grayscale depending on the input)

        -e, --epochs <epochs>
                    Number of training epochs

        -l, --learning_rate <learning_rate>
                    Learning rate
```

## External libraries

- [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) for
  matrix/vector operations
- [stb_image and stb_image_write](https://github.com/nothings/stb) for image
  reading/writing
- [clipp](https://github.com/muellan/clipp) for command line argument parsing

## Inspiration & References

- https://www.youtube.com/watch?v=TkwXa7Cvfr8
- https://www.youtube.com/watch?v=eqIMsdYPaNs
- K. He, X. Zhang, S. Ren and J. Sun, "Delving Deep into Rectifiers: Surpassing
  Human-Level Performance on ImageNet Classification", 2015,
  doi: https://doi.org/10.48550/arXiv.1502.01852
- M. Tancik, P. P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U.
  Singhal, R. Ramamoorthi, J. T. Barron and R. Ng, "Fourier Features Let
  Networks Learn High Frequency Functions in Low Dimensional Domains", 2020,
  doi: https://doi.org/10.48550/arXiv.2006.10739

## License

This software is released under [MIT License](LICENSE).
