# neural_image

## Build

All dependencies are handled
by [CPM.cmake](https://github.com/cpm-cmake/CPM.cmake).

```
git clone https://github.com/mschertenleib/neural_image.git
cd neural_image
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target neural_image
```

## External libraries

- [Eigen](https://github.com/libeigen/eigen)
- [SDL](https://github.com/libsdl-org/SDL)
- [stb_image](https://github.com/nothings/stb)

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
