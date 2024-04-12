# Showcase

[![Showcase gif](./showcase.gif)](./showcase.mp4)

# Why?

At the time of writing, 12.04.2024 <sup>(dd.MM.yyyy)</sup>, it's kinda hard to configure [ort](https://github.com/pykeio/ort) and [opencv-rust](twistedfall/opencv-rust/) inside one Rust project. This repo serves as a guide/starter to save you precious time debugging! [There is even an issue saying that this is impossible lol.](https://github.com/pykeio/ort/issues/145)

# Setup

First of all, you need to download and compile opencv. At the time of writing, the version `4.9.0` is the latest stable.

- If you're on Linux
  - Either you [build from source](https://docs.opencv.orgu/4.9.0/d7/d9f/tutorial_linux_install.html#tutorial_linux_install_quick_build_core)
  - Or you use your package manager to install opencv
- If you're on Windows
  - You can also [build from source](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#autotoc_md1007), but it will be pain in the ass.
  - Or you'll [download the prebuilt files](https://sourceforge.net/projects/opencvlibrary/files/4.9.0/), [extract them](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_prebuilt) somewhere where it makes sense (like `C:\Program Files\opencv-4.9.0`) and [run the env management script](https://docs.opencv.org/4.x/d3/d52/tutorial_windows_install.html#tutorial_windows_install_path).
- When you're building from source, I would suggest to build it somewhere where you don't accidentaly don't delete it, since we'll need that location later.

Once you're done with that, you need to add the following to you system path (on Linux, add to `$PATH` Linux's equivalents)

```sh
C:\Program Files\opencv-4.9.0\build\x64\vc16\lib\
C:\Program Files\opencv-4.9.0\build\include\
C:\Program Files\opencv-4.9.0\build\x64\vc16\bin
```

<small>Btw, you NEED to add these to your path, if you use the `OPENCV_LINK_LIBS` / `OPENCV_LINK_PATHS` / `OPENCV_INCLUDE_PATHS`, you will get runtime linking errors (at least on windows).</small>

Now we can add the dependencies to our Rust project.

```toml
[dependencies]
opencv = "0.89"
ort = "2.0.0-rc.1"
```

And that's it! Now you can go ahead and run `OPENCV_LINK_LIBS='opencv_world490' OPENCV_MSVC_CRT='static' cargo r` to compile your program. Yes, the arguments before the command are required. Alternatively, you can set them as enviromental variables.

# Other

- To ensure that VSCode's autocomplete is working properly, copy settings from `./vscode/settings.json` to your repository.
