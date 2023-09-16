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

## License

This software is released under [MIT License](LICENSE).
